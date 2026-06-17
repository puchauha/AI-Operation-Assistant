"""
supervisor_agent.py
===================
Business Operations Support AI Agent

This agent investigates customer complaints by:
  1. Selecting relevant knowledge files (YAML) based on the complaint
  2. Invoking enterprise diagnostic tools to gather data
  3. Reasoning over the data to produce a structured analysis
  4. Formatting the analysis into a human-readable Decision Card
  5. Reading the human's free-text response and classifying their intent
  6. Re-investigating further if the human wants more data
  7. Accepting a human's direct correction to the root cause / fix
  8. Writing the finalized outcome back to ServiceNow

The agent follows a "Human as Collaborator" model — it investigates and
surfaces findings, but a human always makes the final decision. The human
responds in plain free text (not buttons), and the agent classifies what
the human actually meant before deciding what to do next. Depending on
that classification, the agent either re-investigates with more context,
accepts a direct correction with no further investigation, or finalises
the analysis as-is — and the final outcome is always written back to
ServiceNow as an internal, customer-invisible audit trail.

Methods (in call order):
    1. select_knowledge_base()         — find relevant YAML knowledge files
    2. investigate_and_analyse()       — run tools, reason, return structured dict
    3. present_to_human()              — format dict into a readable Decision Card
    4. classify_human_feedback()       — read human's free-text reply, classify intent
    5. re_investigate_with_feedback()  — continue investigating when human wants more data
    6. apply_human_override()          — accept human's direct RCA/fix correction, no re-investigation
    7. finalize_to_servicenow()        — write the final outcome to ServiceNow work notes

Usage:
    agent     = SupervisorAgent()
    kb        = agent.select_knowledge_base(user_query)
    analysis  = agent.investigate_and_analyse(user_query, kb["selected_yamls"], tools)
    card      = agent.present_to_human(user_query, analysis)
    print(card)

    # ... human reads the card in Slack and replies with free text ...

    feedback = agent.classify_human_feedback(human_message, analysis)

    if feedback["intent"] == "approve":
        agent.finalize_to_servicenow(
            servicenow_client, sys_id, analysis, "approved", human_agent_id
        )

    elif feedback["intent"] == "request_more_data":
        analysis = agent.re_investigate_with_feedback(
            analysis, feedback["extracted_data_request"],
            kb["selected_yamls"], tools
        )
        card = agent.present_to_human(user_query, analysis)
        # ... present updated card, wait for next reply ...

    elif feedback["intent"] == "correct_rca_fix":
        analysis = agent.apply_human_override(
            analysis,
            feedback["extracted_root_cause"],
            feedback["extracted_fix"],
        )
        agent.finalize_to_servicenow(
            servicenow_client, sys_id, analysis, "human_override", human_agent_id
        )
"""

# ── Standard library imports ──────────────────────────────────────────────────
import os       # used to build file paths that work on any operating system
import json     # used to parse and produce JSON — the format LLMs return data in
import textwrap # used to wrap long text lines so they fit neatly in the Decision Card
import yaml     # used to read YAML knowledge files from the knowledge base directory

# ── Third-party imports ───────────────────────────────────────────────────────
# LangChain is the framework we use to talk to LLMs (OpenAI GPT models)
# and to define tools the LLM can call during investigation.
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

# python-dotenv reads the .env file and loads values into environment variables
# so secrets like OPENAI_API_KEY are never hardcoded in source code
import dotenv

# Load .env file once at module level — this runs when the file is first imported.
# Any variable defined in .env (e.g. OPENAI_API_KEY=sk-...) becomes available
# via os.getenv() or is picked up automatically by LangChain's ChatOpenAI.
dotenv.load_dotenv()


# ── Module-level constants ────────────────────────────────────────────────────

# Default path to the knowledge base directory.
# os.path.abspath() converts a relative path to an absolute one.
# os.path.dirname(__file__) gives the directory where THIS file lives.
# We then navigate up two levels (.., ..) to reach the project root,
# and into the knowledge_base folder where the YAML files are stored.
# This means the path always resolves correctly regardless of which
# directory the developer runs the code from.
_DEFAULT_KNOWLEDGE_BASE_DIR = os.path.join(
    os.path.abspath(os.path.dirname(__file__)),
    "..", "..", "knowledge_base"
)

# Width of the Decision Card in characters.
# Used to keep all sections consistently aligned.
_CARD_WIDTH = 64


class SupervisorAgent:
    """
    The main AI agent class.

    All investigation logic lives here as methods.
    The agent is built incrementally — one method at a time,
    each tested before the next is added.

    Attributes:
        router_llm    : cheap, fast LLM used for classification tasks
        reasoning_llm : capable LLM used for investigation and analysis
        knowledge_base_dir: path to the folder containing YAML knowledge files
    """

    def __init__(self, knowledge_base_dir: str = _DEFAULT_KNOWLEDGE_BASE_DIR):
        """
        Initialise the agent.

        Args:
            knowledge_base_dir: path to the YAML knowledge base directory.
                                Defaults to the project-level knowledge_base/
                                folder. Can be overridden in tests or for
                                different deployment environments.

        Note:
            ChatOpenAI reads OPENAI_API_KEY from the environment automatically.
            No need to pass the key explicitly — load_dotenv() above handles it.
        """
        # gpt-4o-mini is used for the router — it is cheaper and faster than
        # gpt-4o, which is sufficient for the classification task of selecting
        # which YAML files are relevant. It does not need deep reasoning ability.
        self.router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # gpt-4o is used for investigation and analysis — it needs stronger
        # reasoning ability to interpret tool results and produce accurate findings.
        # temperature=0 means the model gives deterministic, factual responses
        # rather than creative or varied ones.
        self.reasoning_llm = ChatOpenAI(model="gpt-4o", temperature=0)

        # Store the knowledge base path on the instance so all methods
        # can access it via self.knowledge_base_dir without needing to
        # pass it as a parameter to every method.
        self.knowledge_base_dir = knowledge_base_dir

    # ═══════════════════════════════════════════════════════════════════
    # METHOD 1 — select_knowledge_base
    #
    # Purpose:
    #   Given a free-text user complaint, find all YAML knowledge files
    #   in the knowledge base that are relevant to the incident.
    #
    # How it works:
    #   - Reads the metadata block from every YAML file in knowledge_base/
    #   - Builds a summary of all available files for the LLM to read
    #   - Asks the router LLM to select relevant files based on the query
    #   - Returns selected filenames with a reason for each selection
    #
    # Design decisions:
    #   - Only metadata is read here — full YAML content is loaded later
    #     in investigate_and_analyse() only for selected files
    #   - Multiple YAMLs can be selected — incidents often span domains
    #   - The router LLM selects by semantic meaning, not filename matching
    # ═══════════════════════════════════════════════════════════════════

    # This is the instruction we give the router LLM.
    # It tells the LLM exactly what its job is, what input it will receive,
    # and precisely what format to return the answer in.
    # Using a class-level variable (prefixed with _) keeps it private to
    # this class and avoids recreating the string on every method call.
    _SELECTOR_SYSTEM_PROMPT = """\
You are an incident knowledge router for a hi-tech B2B manufacturing operations system.

You will be given:
  1. A user complaint or incident description
  2. A list of available YAML knowledge files, each with incident type,
     business domain, summary, scenarios, and tags

Your task:
  - Select ALL YAML files relevant to investigating the user complaint
  - A complaint may span multiple domains — select all that apply
  - Do NOT select files that are clearly unrelated
  - For each selected file, provide a one-line reason

Respond ONLY in this exact JSON format — no preamble, no markdown fences:
{
  "selected_yamls": [
    {"filename": "<filename>", "reason": "<one line reason>"},
    ...
  ],
  "overall_reasoning": "<one sentence summary of the selection>"
}

If no YAML is relevant, return:
{
  "selected_yamls": [],
  "overall_reasoning": "<why nothing matched>"
}
"""

    def _load_yaml_metadata(self) -> list[dict]:
        """
        Scan the knowledge base directory and read only the metadata block
        from each YAML file. Returns a list of metadata dictionaries.

        We read only metadata here (not the full file) because:
        - The router LLM only needs summaries to make selection decisions
        - Loading full content for all files would waste tokens and cost money
        - Full content is loaded later only for the files that are selected

        Returns:
            List of dicts, one per valid YAML file, containing:
            filename, incident_type, business_domain, summary, scenarios, tags
        """
        # This list will be populated with one dict per YAML file
        entries = []

        # os.listdir() returns all filenames in the directory as a plain list.
        # sorted() sorts them alphabetically so the order is consistent
        # across different operating systems and runs.
        for filename in sorted(os.listdir(self.knowledge_base_dir)):

            # Skip any file that is not a YAML file.
            # This guards against README files, .DS_Store, etc.
            if not filename.endswith(".yaml"):
                continue

            # Build the full file path by joining the directory and filename.
            # os.path.join() handles path separators correctly on all platforms
            # (backslash on Windows, forward slash on Mac/Linux).
            filepath = os.path.join(self.knowledge_base_dir, filename)

            try:
                # Open and parse the YAML file.
                # yaml.safe_load() converts the YAML text into a Python dict.
                # We use 'safe_load' rather than 'load' for security —
                # safe_load cannot execute arbitrary Python code.
                with open(filepath, "r") as f:
                    content = yaml.safe_load(f)

                # Extract only the metadata section from the parsed YAML dict.
                # .get("metadata", {}) means: give me the "metadata" key,
                # and if it doesn't exist, give me an empty dict instead of
                # raising a KeyError.
                meta = content.get("metadata", {})

                # If a YAML file has no metadata block, we cannot use it for
                # routing. Warn the developer and skip to the next file.
                if not meta:
                    print(f"  [WARN] {filename} has no metadata block — skipping")
                    continue

                # Build a flat dict with just the fields we need for routing.
                # This is cleaner than passing the entire raw YAML dict around.
                entries.append({
                    "filename":        filename,
                    "incident_type":   meta.get("incident_type",   ""),
                    "business_domain": meta.get("business_domain", ""),
                    "summary":         meta.get("summary",         "").strip(),
                    "scenarios":       meta.get("scenarios",       []),
                    "tags":            meta.get("tags",            []),
                })

            except Exception as e:
                # If any file fails to load (e.g. invalid YAML syntax),
                # log the error and continue — don't crash the whole agent.
                print(f"  [ERROR] Could not read {filename}: {e}")

        return entries

    def _build_metadata_context(self, entries: list[dict]) -> str:
        """
        Format the list of metadata dicts into a single readable text block
        that the router LLM can understand.

        We format it as labelled text rather than raw JSON because LLMs
        read and reason over structured text more reliably than dense JSON
        for classification tasks.

        Args:
            entries: list of metadata dicts from _load_yaml_metadata()

        Returns:
            Multi-line string — one block per YAML file, ready for the prompt
        """
        lines = []

        for entry in entries:
            # Each file gets a clearly labelled block so the LLM can
            # distinguish between files and compare them easily
            lines.append(f"FILE: {entry['filename']}")
            lines.append(f"  Incident Type  : {entry['incident_type']}")
            lines.append(f"  Business Domain: {entry['business_domain']}")
            lines.append(f"  Summary        : {entry['summary']}")
            # join() converts the list of scenarios into a single string
            # separated by semicolons so it reads naturally in a prompt
            lines.append(f"  Scenarios      : {'; '.join(entry['scenarios'])}")
            lines.append(f"  Tags           : {', '.join(entry['tags'])}")
            lines.append("")  # blank line between files for readability

        # "\n".join() combines all lines into one string with line breaks
        return "\n".join(lines)

    def _parse_llm_json(self, raw: str) -> dict:
        """
        Parse a JSON string returned by the LLM.

        Some LLMs wrap their JSON response in markdown code fences
        (```json ... ```) even when instructed not to. This method
        strips those fences before parsing so we always get clean JSON.

        Args:
            raw: raw string content from the LLM response

        Returns:
            Parsed dict

        Raises:
            ValueError: if the string cannot be parsed as JSON even
                        after fence stripping, with the raw content
                        included for debugging
        """
        # Remove leading and trailing whitespace
        raw = raw.strip()

        # Check if the response starts with a markdown code fence.
        # Some models return ```json\n{...}\n``` despite instructions.
        if raw.startswith("```"):
            # split("```") breaks the string at every occurrence of ```.
            # The JSON content will be in position [1] — between the fences.
            raw = raw.split("```")[1]
            # Remove the optional "json" language tag that follows the fence
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        # Attempt to parse the cleaned string as JSON.
        # If it still fails, raise a clear error with the raw content
        # so the developer can see exactly what the LLM returned.
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"LLM returned content that could not be parsed as JSON.\n"
                f"Parse error: {e}\n"
                f"Raw content:\n{raw}"
            )

    def select_knowledge_base(self, user_query: str) -> dict:
        """
        Given a free-text user complaint, return all relevant YAML
        knowledge files from the knowledge base.

        This is the first method called in the pipeline. Its output
        is passed directly as input to investigate_and_analyse().

        Args:
            user_query: the raw complaint text from the user,
                        e.g. "My order hasn't arrived in 3 days"

        Returns:
            {
                "selected_yamls": [
                    {"filename": "order_delay.yaml", "reason": "..."},
                    ...
                ],
                "overall_reasoning": "one sentence explaining the selection",
                "available_count":   total number of YAML files found,
                "selected_count":    number of files selected as relevant
            }
        """
        # Load metadata from all YAML files in the knowledge base
        entries = self._load_yaml_metadata()

        # If no YAML files were found, return early with a clear message.
        # This guards against misconfiguration or an empty knowledge base.
        if not entries:
            return {
                "selected_yamls":    [],
                "overall_reasoning": "No YAML files found in knowledge base.",
                "available_count":   0,
                "selected_count":    0,
            }

        # Format the metadata into a readable block for the LLM prompt
        metadata_context = self._build_metadata_context(entries)

        # Call the router LLM with two messages:
        # - SystemMessage: sets the LLM's role and output format rules
        # - HumanMessage: provides the actual data and the user's complaint
        response = self.router_llm.invoke([
            SystemMessage(content=self._SELECTOR_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"AVAILABLE KNOWLEDGE FILES:\n\n{metadata_context}"
                f"USER COMPLAINT:\n\"{user_query}\"\n\n"
                "Select all relevant YAML files."
            ))
        ])

        # response.content is the raw text string the LLM returned.
        # Parse it into a Python dict using our shared helper.
        result = self._parse_llm_json(response.content)

        # Add counts so the caller can make decisions without
        # having to compute len() themselves
        result["available_count"] = len(entries)
        result["selected_count"]  = len(result.get("selected_yamls", []))

        return result

    # ═══════════════════════════════════════════════════════════════════
    # METHOD 2 — investigate_and_analyse
    #
    # Purpose:
    #   Investigate the user complaint using YAML knowledge and enterprise
    #   tools. Returns a structured dict — not formatted text.
    #   Formatting is the responsibility of present_to_human() (method 3).
    #
    # How it works:
    #   - Loads full YAML content for each file selected by method 1
    #   - Seeds the reasoning LLM with the complaint + YAML context
    #   - Runs a tool loop: LLM calls tools → results fed back → LLM continues
    #   - When the LLM stops calling tools, it produces the final JSON analysis
    #   - Returns the parsed JSON as a Python dict
    #
    # Design decisions:
    #   - Tools are passed in as a parameter — the agent doesn't hardcode them.
    #     This means new tools can be added without changing the agent.
    #   - bind_tools() is called fresh each iteration — not stored on self —
    #     so each investigation is fully independent.
    #   - The loop is capped at MAX_ITERATIONS to prevent runaway API calls
    #     if the LLM enters an unexpected tool-calling loop.
    #   - LLM returns JSON, not formatted text — clean separation of concerns.
    # ═══════════════════════════════════════════════════════════════════

    # Instruction for the reasoning LLM.
    # Tells it to investigate using tools, then return a structured JSON dict.
    # The JSON structure matches exactly what present_to_human() expects.
    _REASONING_SYSTEM_PROMPT = """\
You are an enterprise AI operations assistant conducting an operational investigation
for a hi-tech B2B manufacturing company connected to partners via RosettaNet.

You have access to diagnostic tools. Use them to gather enterprise data relevant
to the customer complaint and the YAML operational knowledge provided.

Investigation approach:
  - Read the YAML knowledge context carefully — it tells you what data to collect
  - Call tools in logical sequence based on what the YAML says to check
  - Use findings from one tool to decide whether another tool is needed
  - Do not call tools that are clearly irrelevant to the complaint

Once investigation is complete, respond ONLY in this exact JSON format —
no preamble, no markdown fences:
{
  "operational_issue": "<one line description of the most likely issue>",
  "findings": [
    "<finding 1 — specific data point from tool results>",
    "<finding 2>",
    "<finding 3>"
  ],
  "reasoning_steps": [
    "<step 1 — what you checked and what it indicated>",
    "<step 2>",
    "<step 3>"
  ],
  "root_cause": "<specific root cause based on evidence>",
  "short_term_fix": "<concrete actionable fix drawn from YAML short_term_fix guidance>"
}
"""

    def _load_full_yaml_content(self, selected_yamls: list[dict]) -> str:
        """
        Load the complete content of each selected YAML file and combine
        them into a single knowledge context string for the reasoning LLM.

        This is called in method 2 after method 1 has identified which
        files are relevant. We only load the selected files — not all files —
        to keep the LLM prompt focused and avoid unnecessary token usage.

        Args:
            selected_yamls: list of {"filename": ..., "reason": ...} dicts
                            as returned by select_knowledge_base()

        Returns:
            Multi-line string with full YAML content of all selected files,
            each clearly labelled with its filename
        """
        sections = []

        for item in selected_yamls:
            filename = item["filename"]
            filepath = os.path.join(self.knowledge_base_dir, filename)

            try:
                with open(filepath, "r") as f:
                    content = yaml.safe_load(f)

                # Convert the parsed YAML dict back to a formatted JSON string
                # for the LLM prompt. JSON is more compact and unambiguous
                # than YAML for LLM consumption.
                sections.append(
                    f"=== KNOWLEDGE BASE: {filename} ===\n"
                    f"{json.dumps(content, indent=2)}\n"
                )

            except Exception as e:
                # Log the error but continue — if one file fails,
                # we can still investigate with the remaining files
                print(f"  [ERROR] Could not load {filename}: {e}")

        return "\n".join(sections)

    def investigate_and_analyse(
        self,
        user_query:     str,
        selected_yamls: list[dict],
        tools:          list,
    ) -> dict:
        """
        Investigate the user complaint using YAML knowledge and enterprise tools.
        The reasoning LLM decides which tools to call based on YAML guidance.

        This is the second method called in the pipeline. Its output is
        passed directly as input to present_to_human().

        Args:
            user_query:     original free-text complaint from the user
            selected_yamls: list of {"filename": ..., "reason": ...} dicts
                            from select_knowledge_base()
            tools:          list of LangChain @tool decorated functions.
                            The reasoning LLM reads their descriptions and
                            decides which ones to call and with what arguments.

        Returns:
            Structured dict with keys:
            {
                "operational_issue": str   — one-line issue description,
                "findings":          list  — data points from tool results,
                "reasoning_steps":   list  — how the LLM reached its conclusion,
                "root_cause":        str   — specific root cause,
                "short_term_fix":    str   — actionable fix from YAML guidance
            }
        """
        # Build a name→tool lookup dict so we can invoke a tool by name
        # when the LLM tells us which tool it wants to call.
        # {t.name: t} creates a dict where the key is the tool's name
        # (from the @tool decorator) and the value is the tool object itself.
        tool_by_name = {t.name: t for t in tools}

        # Load the full content of all selected YAML files into one string
        knowledge_context = self._load_full_yaml_content(selected_yamls)

        # Build the initial conversation for the reasoning LLM.
        # LLMs work with a list of messages — each message has a role
        # (system, human, AI, tool) and content.
        # SystemMessage sets the LLM's behaviour and output format.
        # HumanMessage provides the actual data to investigate.
        messages = [
            SystemMessage(content=self._REASONING_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"CUSTOMER COMPLAINT:\n{user_query}\n\n"
                f"OPERATIONAL KNOWLEDGE:\n{knowledge_context}\n\n"
                "Begin your investigation. Call relevant tools to gather "
                "enterprise data, then produce your final JSON analysis."
            ))
        ]

        # Dictionary to accumulate all tool results across iterations.
        # Used in the safety fallback if the iteration limit is reached.
        accumulated_tool_results = {}

        # Safety ceiling — prevents runaway tool-calling loops.
        # In normal operation, 2-3 iterations are sufficient:
        # iteration 1: call tools, iteration 2: produce final analysis.
        MAX_ITERATIONS = 6

        for iteration in range(1, MAX_ITERATIONS + 1):
            print(f"  [reasoning LLM — iteration {iteration}]")

            # bind_tools() tells the LLM about the available tools by
            # converting each @tool function's signature and docstring
            # into a JSON schema the OpenAI API understands.
            # We call bind_tools() fresh each iteration rather than
            # storing the bound LLM on self, so each call is independent.
            response = self.reasoning_llm.bind_tools(tools).invoke(messages)

            # Append the LLM's response to the conversation history.
            # This is how the LLM "remembers" what it already decided —
            # we pass the full history back on every call.
            messages.append(response)

            # If the LLM returned no tool calls, it has finished investigating
            # and its response content is the final JSON analysis.
            if not response.tool_calls:
                print(f"  Investigation complete after {iteration} iteration(s)")

                # Parse the JSON response using our shared helper.
                # If parsing fails, raise a clear error rather than
                # returning garbage data to the caller.
                try:
                    return self._parse_llm_json(response.content)
                except ValueError as e:
                    raise ValueError(
                        f"investigate_and_analyse: reasoning LLM returned "
                        f"content that could not be parsed as JSON.\n{e}"
                    )

            # The LLM wants to call one or more tools.
            # response.tool_calls is a list — the LLM can request
            # multiple tools in a single iteration.
            for tc in response.tool_calls:
                tool_name = tc["name"]      # name of the tool to call
                tool_args = tc.get("args", {})  # arguments to pass to the tool
                call_id   = tc["id"]        # unique ID linking this call to its result

                print(f"  → Tool called: {tool_name} | args: {tool_args}")

                # Look up the tool object by name in our registry dict
                tool = tool_by_name.get(tool_name)

                if not tool:
                    # The LLM requested a tool that isn't in our registry.
                    # Return an error result rather than crashing.
                    result = {"error": f"Tool '{tool_name}' not registered"}
                else:
                    try:
                        # tool.invoke() calls the actual tool function
                        # with the arguments the LLM provided
                        result = tool.invoke(tool_args)
                        print(f"    Result: {result}")
                    except Exception as e:
                        result = {"error": str(e)}
                        print(f"    Error: {e}")

                # Store the result for the fallback dict
                accumulated_tool_results[tool_name] = result

                # Feed the tool result back to the LLM as a ToolMessage.
                # ToolMessage is a specific message type that the OpenAI API
                # requires — it must include the tool_call_id so the API can
                # match this result to the tool call that requested it.
                messages.append(
                    ToolMessage(
                        tool_call_id=call_id,
                        content=json.dumps(result)
                    )
                )

        # If we reach here, the LLM called tools on every iteration without
        # ever producing a final answer — this should not happen in normal
        # operation. Return a structured fallback dict rather than crashing.
        print(f"  [WARN] Reached iteration limit of {MAX_ITERATIONS}")
        return {
            "operational_issue": "Investigation incomplete — iteration limit reached",
            "findings":          [json.dumps(accumulated_tool_results)],
            "reasoning_steps":   ["Exceeded maximum tool call iterations"],
            "root_cause":        "Unable to determine within iteration limit",
            "short_term_fix":    "Escalate to human operator for manual investigation"
        }

    # ═══════════════════════════════════════════════════════════════════
    # METHOD 3 — present_to_human
    #
    # Purpose:
    #   Format the structured analysis dict from method 2 into a clean,
    #   readable Decision Card for the human collaborator.
    #
    # How it works:
    #   - Takes the user query and analysis dict as input
    #   - Formats each section with consistent headers, emojis, and alignment
    #   - Returns the complete formatted string — no side effects
    #
    # Design decisions:
    #   - Pure presentation — no LLM calls, no business logic
    #   - Returns a string rather than printing directly, so the caller
    #     decides where the output goes (console, Slack, email, etc.)
    #   - Text wrapping applied so long lines don't overflow the card
    #   - Slack integration point is explicitly marked with a comment
    # ═══════════════════════════════════════════════════════════════════

    def present_to_human(self, user_query: str, analysis: dict) -> str:
        """
        Format the analysis dict into a readable Decision Card for human review.

        This is the third and final method in the pipeline. The output is
        ready to display to the human operations agent who will approve,
        override, or escalate the recommended action.

        Args:
            user_query: original free-text complaint from the user.
                        Included in the card so the reviewer has full context.
            analysis:   structured dict returned by investigate_and_analyse().
                        Expected keys: operational_issue, findings,
                        reasoning_steps, root_cause, short_term_fix.

        Returns:
            Formatted Decision Card string.

            Slack integration point:
                Pass this string to src/integrations/slack as a
                Block Kit section block to send it as a Slack message.
        """

        # ── Private formatting helpers ────────────────────────────────
        # These are small functions defined inside present_to_human()
        # because they are only needed here and would clutter the class
        # if defined at class level.

        def _section_header(title: str, emoji: str) -> str:
            """
            Produce a section header with an emoji, title, and underline.
            Example:
                📊  ENTERPRISE FINDINGS
                ────────────────────────────────────────────────────────────
            """
            underline = "─" * _CARD_WIDTH
            return f"\n  {emoji}  {title}\n  {underline}"

        def _wrap(text: str, indent: int = 2) -> str:
            """
            Wrap a long text string so it doesn't overflow the card width.
            textwrap.fill() breaks the text at word boundaries.
            subsequent_indent adds spaces to align continuation lines
            with the start of the first line.
            """
            return textwrap.fill(
                text,
                width=_CARD_WIDTH,
                initial_indent=" " * indent,
                subsequent_indent=" " * indent
            )

        def _bullet_list(items: list) -> str:
            """
            Format a list of strings as a bulleted list.
            Each item is wrapped to fit the card width.
            Example:
                •  Warehouse stock available: 18 units.
                •  Pending allocation requests: 34.
            """
            lines = []
            for item in items:
                # Wrap with extra indent for continuation lines
                # so they align with the text after the bullet
                wrapped = textwrap.fill(
                    f"•  {item}",
                    width=_CARD_WIDTH,
                    initial_indent="  ",
                    subsequent_indent="     "
                )
                lines.append(wrapped)
            return "\n".join(lines)

        def _numbered_list(items: list) -> str:
            """
            Format a list of strings as a numbered list.
            Example:
                1.  Checked warehouse stock — 18 units available.
                2.  Checked allocation queue — order at position 14.
            """
            lines = []
            for i, item in enumerate(items):
                # enumerate() gives us both the index (i) and the value (item)
                # i+1 because we want numbering to start at 1, not 0
                wrapped = textwrap.fill(
                    f"{i + 1}.  {item}",
                    width=_CARD_WIDTH,
                    initial_indent="  ",
                    subsequent_indent="      "
                )
                lines.append(wrapped)
            return "\n".join(lines)

        # ── Build the card section by section ─────────────────────────
        # We build a list of strings and join them at the end.
        # This is more efficient than concatenating strings one by one
        # with += because Python strings are immutable — each += creates
        # a new string object in memory.
        card = []

        # ── Header ───────────────────────────────────────────────────
        # Box-drawing characters (╔ ═ ╗ ║ ╚ ╝) create a visible border
        # that makes the card stand out in terminal output and Slack.
        card.append("╔" + "═" * (_CARD_WIDTH + 2) + "╗")
        card.append(
            "║" +
            "   AI OPERATIONS ASSISTANT — DECISION CARD".center(_CARD_WIDTH + 2) +
            "║"
        )
        card.append("╚" + "═" * (_CARD_WIDTH + 2) + "╝")

        # ── Customer Complaint ────────────────────────────────────────
        card.append(_section_header("CUSTOMER COMPLAINT", "📋"))
        # Display the query exactly as the user typed it — no wrapping.
        # This is a verbatim quote and must not be line-broken.
        # The human reviewer needs to see the original complaint unchanged.
        card.append(f'  "{user_query}"')

        # ── Operational Issue ─────────────────────────────────────────
        card.append(_section_header("MOST PROBABLE OPERATIONAL ISSUE", "🔍"))
        card.append(_wrap(analysis.get("operational_issue", "N/A")))

        # ── Enterprise Findings ───────────────────────────────────────
        card.append(_section_header("ENTERPRISE FINDINGS", "📊"))
        # .get() with a default empty list prevents a crash if the key
        # is missing from the analysis dict
        card.append(_bullet_list(analysis.get("findings", [])))

        # ── Reasoning Steps ───────────────────────────────────────────
        card.append(_section_header("AI REASONING STEPS", "🧠"))
        card.append(_numbered_list(analysis.get("reasoning_steps", [])))

        # ── Root Cause ────────────────────────────────────────────────
        card.append(_section_header("PROBABLE ROOT CAUSE", "🎯"))
        card.append(_wrap(analysis.get("root_cause", "N/A")))

        # ── Short Term Fix ────────────────────────────────────────────
        card.append(_section_header("SUGGESTED SHORT-TERM FIX", "🛠 "))
        card.append(_wrap(analysis.get("short_term_fix", "N/A")))

        # ── Status & Human Action Buttons ─────────────────────────────
        # The double line visually separates the analysis from the action zone
        card.append("\n" + "═" * (_CARD_WIDTH + 2))
        card.append("  ⏳  AWAITING HUMAN REVIEW")
        card.append("═" * (_CARD_WIDTH + 2))
        card.append(
            "\n  [ ✅ APPROVE ]  [ ✏️  OVERRIDE ]  "
            "[ 🔺 ESCALATE ]  [ ❓ MORE DATA ]"
        )

        # ── Footer ────────────────────────────────────────────────────
        card.append("\n  " + "─" * _CARD_WIDTH)
        card.append("  AI Operations Assistant  |  Human as Collaborator Model")
        card.append("  " + "─" * _CARD_WIDTH)

        # Join all sections with newlines to produce the final string.
        # "\n".join() is equivalent to putting \n between every item in the list.
        return "\n".join(card)

    # ═══════════════════════════════════════════════════════════════════
    # METHOD 4 — classify_human_feedback
    #
    # Purpose:
    #   After present_to_human() shows the Decision Card in Slack, the
    #   human agent replies in their own words — free text, not a button
    #   click. This method reads that free text and figures out what the
    #   human actually means, so the rest of the system knows what to do
    #   next (update ServiceNow, re-investigate, or accept a correction).
    #
    # Why free text instead of buttons?
    #   Buttons force the human to pre-categorise their own feedback
    #   before they can even give it. Free text lets a human write
    #   naturally — e.g. "data's fine but the RCA is wrong, it's actually
    #   X" — and lets the AI do the work of figuring out the category.
    #   This is the same "Human as Companion" idea applied to the
    #   response side, not just the analysis side.
    #
    # How it works:
    #   - We send the router LLM three things: the human's message, the
    #     AI's own original analysis (for context), and a strict
    #     instruction to classify the message into one of four buckets.
    #   - The four buckets (the "intent") are:
    #       "approve"            — human is happy with everything
    #       "request_more_data"  — human wants the AI to investigate further
    #       "correct_rca_fix"    — human says data is fine, but the
    #                              root cause / fix is wrong, and gives
    #                              the correct version themselves
    #       "unclear"            — the AI genuinely cannot tell what the
    #                              human means; in this case the AI must
    #                              NOT guess and must ask a clarifying
    #                              question instead
    #   - The router LLM is the same cheap, fast model used in
    #     select_knowledge_base() (gpt-4o-mini) — classifying intent is
    #     a simple task and does not need the more expensive reasoning LLM.
    #
    # Design decisions:
    #   - This method does NOT take any action itself (it doesn't update
    #     ServiceNow or re-run investigation). It only classifies. The
    #     caller (e.g. a Slack event handler) decides what to do with the
    #     classification. This keeps the method simple, predictable, and
    #     easy to test in isolation — exactly like select_knowledge_base()
    #     only selects YAMLs, it doesn't load their full content itself.
    #   - When intent is "correct_rca_fix", we ask the LLM to extract the
    #     human's corrected root cause and fix into separate fields, so
    #     the caller doesn't have to re-parse free text again later.
    #   - When intent is "request_more_data", we extract what additional
    #     information the human is asking for, again so the caller has
    #     structured data to act on rather than raw text.
    #   - If the LLM is not confident about its classification, it must
    #     say so via the "confidence" field rather than silently guessing.
    #     A low-confidence "approve" is dangerous — we would rather know
    #     to treat it as "unclear" and ask the human to confirm.
    # ═══════════════════════════════════════════════════════════════════

    # This is the instruction we give the router LLM for classifying
    # human feedback. Notice the structure is very similar to
    # _SELECTOR_SYSTEM_PROMPT above — same pattern, different task.
    _FEEDBACK_CLASSIFIER_SYSTEM_PROMPT = """\
You are reading a human operations agent's reply to an AI-generated
incident analysis. Your job is to classify what the human means and
extract any structured information from their reply.

You will be given:
  1. The AI's original analysis (operational issue, root cause, fix)
  2. The human's free-text reply

Classify the human's reply into exactly ONE of these four intents:

  "approve"
      The human accepts the AI's analysis as-is. No corrections, no
      requests for more data. Example: "Looks good, go ahead."

  "request_more_data"
      The human wants the AI to investigate further or gather additional
      information before a decision can be made. The original findings
      are not being disputed — the human just wants more. Example:
      "Can you also check the invoice details for this order?"

  "correct_rca_fix"
      The human accepts that the AI's enterprise findings (the data
      collected) are correct, but says the root cause or the suggested
      fix is wrong, and tells you what the correct one should be.
      Example: "Data is right, but this isn't a warehouse backlog issue —
      it's actually a payment hold. Fix should be to release the hold."

  "unclear"
      You cannot confidently tell which of the above three the human
      means. Do NOT guess. It is better to say "unclear" and let the
      system ask the human to clarify than to misclassify and take the
      wrong action automatically.

Respond ONLY in this exact JSON format — no preamble, no markdown fences:
{
  "intent": "approve" | "request_more_data" | "correct_rca_fix" | "unclear",
  "confidence": "high" | "medium" | "low",
  "extracted_data_request": "<what the human is asking for, or null>",
  "extracted_root_cause": "<human's corrected root cause, or null>",
  "extracted_fix": "<human's corrected fix, or null>",
  "reasoning": "<one sentence explaining why you chose this intent>"
}

Rules:
  - extracted_data_request is only filled when intent is "request_more_data"
  - extracted_root_cause and extracted_fix are only filled when intent
    is "correct_rca_fix" — if the human only corrected one of the two
    (e.g. only the fix, not the root cause), leave the other one null
  - If confidence is "low", strongly prefer setting intent to "unclear"
    rather than guessing one of the other three
  - All fields must always be present in the JSON, even if their value is null
"""

    def classify_human_feedback(
        self,
        human_message: str,
        original_analysis: dict,
    ) -> dict:
        """
        Read a human agent's free-text reply to a Decision Card and
        classify what they mean.

        This is the fourth method in the pipeline, called after
        present_to_human() has shown the Decision Card and the human
        has typed a response (in Slack, or wherever the card was shown).

        Args:
            human_message:
                The exact free text the human typed as their reply.
                Example: "data is correct but RCA is wrong, it's actually
                a payment hold, fix should be to release it"

            original_analysis:
                The structured dict that investigate_and_analyse() returned
                for this incident. We pass this in so the classifier LLM
                has the full context of what the AI originally said —
                without this, "the RCA is wrong" would be meaningless to
                the LLM, since it would not know what the original RCA was.

        Returns:
            A dict with this exact shape:
            {
                "intent": "approve" | "request_more_data" |
                          "correct_rca_fix" | "unclear",
                "confidence": "high" | "medium" | "low",
                "extracted_data_request": str or None,
                "extracted_root_cause":   str or None,
                "extracted_fix":          str or None,
                "reasoning": str
            }

            The caller is responsible for deciding what to actually DO
            with this classification (e.g. call a different method for
            each intent type). This method only classifies — it does not
            take any action itself.
        """

        # ── Step 1: Build the context block the LLM needs ──────────────
        # We show the LLM the AI's own original analysis so it has
        # something concrete to compare the human's reply against.
        # json.dumps with indent=2 makes this readable in the prompt,
        # the same way we format YAML content for the reasoning LLM
        # in _load_full_yaml_content().
        original_analysis_text = json.dumps(original_analysis, indent=2)

        # ── Step 2: Call the router LLM ─────────────────────────────────
        # We reuse self.router_llm (gpt-4o-mini) — the same cheap LLM
        # used in select_knowledge_base(). Classifying a human's intent
        # from a short message is a simple task; it does not need the
        # more expensive reasoning LLM that investigate_and_analyse() uses.
        response = self.router_llm.invoke([
            SystemMessage(content=self._FEEDBACK_CLASSIFIER_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"AI'S ORIGINAL ANALYSIS:\n{original_analysis_text}\n\n"
                f"HUMAN'S REPLY:\n\"{human_message}\"\n\n"
                "Classify the human's reply."
            ))
        ])

        # ── Step 3: Parse the LLM's JSON response ───────────────────────
        # We reuse the exact same _parse_llm_json() helper that
        # select_knowledge_base() and investigate_and_analyse() already
        # use. This is exactly why that helper was pulled out into its
        # own private method earlier — so every place that needs to
        # parse LLM JSON output behaves identically and we do not
        # duplicate the markdown-fence-stripping logic three times.
        try:
            result = self._parse_llm_json(response.content)
        except ValueError as e:
            # If even the classifier itself returns unparseable JSON,
            # we do not want the whole pipeline to crash. Instead we
            # return a safe "unclear" result so the caller asks the
            # human to clarify, rather than the system throwing an
            # unhandled exception in front of a real user.
            print(f"  [WARN] classify_human_feedback: could not parse "
                  f"LLM response — defaulting to 'unclear'. Error: {e}")
            return {
                "intent": "unclear",
                "confidence": "low",
                "extracted_data_request": None,
                "extracted_root_cause": None,
                "extracted_fix": None,
                "reasoning": "Classifier LLM returned unparseable output."
            }

        # ── Step 4: Defensive defaults ───────────────────────────────────
        # Even though we instruct the LLM to always include every field,
        # LLMs occasionally omit a field despite instructions. Using
        # .get(key, default) here means a missing field becomes a safe
        # default value instead of causing a KeyError later when some
        # other part of the code does result["extracted_root_cause"].
        result.setdefault("intent", "unclear")
        result.setdefault("confidence", "low")
        result.setdefault("extracted_data_request", None)
        result.setdefault("extracted_root_cause", None)
        result.setdefault("extracted_fix", None)
        result.setdefault("reasoning", "")

        # ── Step 5: Safety rule — low confidence forces "unclear" ───────
        # This mirrors a rule we wrote directly into the prompt above,
        # but we also enforce it here in code. Prompts are not 100%
        # reliable — an LLM might still return "approve" with "low"
        # confidence despite being told not to. Enforcing the rule in
        # Python code as well, not just in the prompt text, means the
        # system behaves safely even if the LLM does not follow
        # instructions perfectly. This is a defence-in-depth pattern:
        # never rely on a prompt alone for a safety-critical rule.
        if result["confidence"] == "low" and result["intent"] != "unclear":
            print(f"  [INFO] Low confidence ({result['intent']}) — "
                  f"overriding to 'unclear' for safety")
            result["intent"] = "unclear"

        return result

    # ═══════════════════════════════════════════════════════════════════
    # METHOD 5 — re_investigate_with_feedback
    #
    # Purpose:
    #   Called when classify_human_feedback() returned intent =
    #   "request_more_data". The human did NOT say anything is wrong with
    #   the original findings — they just want the AI to dig deeper before
    #   a decision is made. This method re-runs the investigation, but
    #   this time the human's specific request is added to the context
    #   so the reasoning LLM knows exactly what extra information to go
    #   find, rather than repeating the same investigation from scratch.
    #
    # Why this is a SEPARATE method from apply_human_override() (method 6):
    #   This is one of the most important design decisions in this phase
    #   of the project. There are two very different things a human can
    #   mean when they don't simply approve:
    #
    #     (a) "I need MORE INFORMATION before I can decide"
    #         → the AI needs to go back to its tools and investigate again
    #         → this is what re_investigate_with_feedback() (this method) does
    #
    #     (b) "The information you already have is fine, but your
    #          CONCLUSION from it is wrong, and here is the correct one"
    #         → the AI does NOT need to re-investigate — the human has
    #           already told it the right answer
    #         → this is what apply_human_override() (the next method) does
    #
    #   If we used a single generic method for both cases, we would risk
    #   the AI pointlessly re-running tool calls in case (b) — wasting
    #   API calls and, worse, potentially "re-discovering" its own
    #   original (wrong) conclusion and contradicting the human's
    #   correction. Keeping these as two distinct methods makes the
    #   correct behaviour the only possible behaviour, rather than
    #   relying on internal logic to decide what to do every time.
    #
    # How it works:
    #   - We take the ORIGINAL analysis dict and the human's specific
    #     request for more data (as a plain string).
    #   - We re-run essentially the same tool-calling loop as
    #     investigate_and_analyse(), but the very first message to the
    #     reasoning LLM now includes the original findings AND the
    #     human's new request, so the LLM treats this as "continue
    #     investigating", not "start over from nothing".
    #   - The same YAML knowledge and the same tools are reused — the
    #     incident type has not changed, only the depth of investigation.
    #
    # Design decisions:
    #   - We deliberately reuse the exact same tool-calling loop pattern
    #     as investigate_and_analyse() — same MAX_ITERATIONS safety cap,
    #     same ToolMessage feedback pattern — so the two methods behave
    #     consistently and any future bug fix to the loop logic is easy
    #     to apply to both in the same way.
    #   - The output shape is IDENTICAL to investigate_and_analyse()'s
    #     output (the same five keys). This means present_to_human() can
    #     be called on the result of THIS method exactly the same way it
    #     is called on the result of investigate_and_analyse() — no
    #     special-casing needed in present_to_human() at all.
    # ═══════════════════════════════════════════════════════════════════

    # Instruction for the reasoning LLM when it is continuing an
    # investigation rather than starting a fresh one. Notice this is
    # very similar to _REASONING_SYSTEM_PROMPT but explicitly tells the
    # LLM that this is a follow-up round, not a first pass.
    _REINVESTIGATION_SYSTEM_PROMPT = """\
You are an enterprise AI operations assistant continuing an operational
investigation for a hi-tech B2B manufacturing company connected to
partners via RosettaNet.

You already produced an initial analysis. A human operations agent has
reviewed it and is asking for additional information before they can
make a decision. They are NOT disputing your existing findings — they
simply want you to investigate further.

You have access to diagnostic tools. Use them to gather the additional
enterprise data the human is asking for, and incorporate it alongside
your original findings.

Investigation approach:
  - Read your ORIGINAL analysis below — do not contradict it without
    new evidence; you are extending it, not starting over
  - Read the human's specific request carefully — call tools that would
    answer exactly what they asked for
  - Combine the original findings with any new findings into one
    complete, updated analysis

Once the additional investigation is complete, respond ONLY in this
exact JSON format — no preamble, no markdown fences:
{
  "operational_issue": "<one line description of the most likely issue>",
  "findings": [
    "<finding 1 — include original findings plus any new ones>",
    "<finding 2>",
    "<finding 3>"
  ],
  "reasoning_steps": [
    "<step 1 — what you checked and what it indicated>",
    "<step 2>",
    "<step 3>"
  ],
  "root_cause": "<specific root cause based on all evidence so far>",
  "short_term_fix": "<concrete actionable fix drawn from YAML short_term_fix guidance>"
}
"""

    def re_investigate_with_feedback(
        self,
        original_analysis: dict,
        data_request:      str,
        selected_yamls:    list[dict],
        tools:             list,
    ) -> dict:
        """
        Re-run investigation when the human wants more data, without
        treating it as a brand new investigation from scratch.

        Called after classify_human_feedback() returns
        intent == "request_more_data". The data_request string should
        come from that result's "extracted_data_request" field.

        Args:
            original_analysis:
                The dict previously returned by investigate_and_analyse()
                (or by a prior call to this same method, if the human
                has asked for more data more than once).

            data_request:
                Plain text describing what additional information the
                human wants. Example: "Can you also check the invoice
                details for this order?"

            selected_yamls:
                Same list of {"filename": ..., "reason": ...} dicts used
                in the original investigation. We reuse the same
                knowledge base files — the incident type has not changed.

            tools:
                Same list of LangChain @tool functions available to the
                reasoning LLM. The LLM decides which ones to call based
                on the human's specific request.

        Returns:
            A dict with the exact same five keys as investigate_and_analyse():
            {
                "operational_issue": str,
                "findings":          list,
                "reasoning_steps":   list,
                "root_cause":        str,
                "short_term_fix":    str
            }
            This dict can be passed directly into present_to_human() —
            no special handling is needed because the shape matches.
        """
        # Build a name→tool lookup dict, exactly like investigate_and_analyse()
        tool_by_name = {t.name: t for t in tools}

        # Reload the full YAML content — same knowledge base files as before.
        # We reload rather than caching from the original call because this
        # method may be called much later, possibly even as a separate
        # Slack event handler invocation where the original in-memory
        # context no longer exists.
        knowledge_context = self._load_full_yaml_content(selected_yamls)

        # Convert the original analysis dict to a readable JSON string so
        # we can show it to the LLM as context for what it already found.
        original_analysis_text = json.dumps(original_analysis, indent=2)

        # Build the initial conversation. Notice this is different from
        # investigate_and_analyse()'s first message — here we explicitly
        # include the ORIGINAL analysis and the human's new request,
        # so the LLM understands this is a continuation, not a fresh start.
        messages = [
            SystemMessage(content=self._REINVESTIGATION_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"YOUR ORIGINAL ANALYSIS:\n{original_analysis_text}\n\n"
                f"HUMAN'S REQUEST FOR MORE DATA:\n\"{data_request}\"\n\n"
                f"OPERATIONAL KNOWLEDGE:\n{knowledge_context}\n\n"
                "Continue your investigation. Call tools to gather the "
                "additional information requested, then produce your "
                "updated final JSON analysis."
            ))
        ]

        # The tool-calling loop below is intentionally structured the
        # same way as investigate_and_analyse()'s loop. Keeping the same
        # pattern in both places means a developer who already
        # understands one of these methods can read the other quickly,
        # and any future bug fix to how we handle tool calls can be
        # applied consistently to both.
        accumulated_tool_results = {}
        MAX_ITERATIONS = 6

        for iteration in range(1, MAX_ITERATIONS + 1):
            print(f"  [re-investigation — iteration {iteration}]")

            response = self.reasoning_llm.bind_tools(tools).invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                print(f"  Re-investigation complete after {iteration} iteration(s)")
                try:
                    return self._parse_llm_json(response.content)
                except ValueError as e:
                    raise ValueError(
                        f"re_investigate_with_feedback: reasoning LLM "
                        f"returned content that could not be parsed as "
                        f"JSON.\n{e}"
                    )

            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc.get("args", {})
                call_id   = tc["id"]

                print(f"  → Tool called: {tool_name} | args: {tool_args}")

                tool = tool_by_name.get(tool_name)

                if not tool:
                    result = {"error": f"Tool '{tool_name}' not registered"}
                else:
                    try:
                        result = tool.invoke(tool_args)
                        print(f"    Result: {result}")
                    except Exception as e:
                        result = {"error": str(e)}
                        print(f"    Error: {e}")

                accumulated_tool_results[tool_name] = result

                messages.append(
                    ToolMessage(
                        tool_call_id=call_id,
                        content=json.dumps(result)
                    )
                )

        # Safety fallback if the LLM never stops calling tools.
        # We deliberately preserve the ORIGINAL analysis's root cause
        # and fix in this fallback rather than returning generic
        # placeholder text, because falling back to nothing would
        # actually lose the information we already had before this
        # re-investigation attempt even started.
        print(f"  [WARN] Reached iteration limit of {MAX_ITERATIONS} "
              f"during re-investigation")
        return {
            "operational_issue": original_analysis.get(
                "operational_issue", "Unable to complete re-investigation"
            ),
            "findings": original_analysis.get("findings", []) + [
                f"Additional investigation requested ('{data_request}') "
                f"could not be completed within the iteration limit."
            ],
            "reasoning_steps": original_analysis.get("reasoning_steps", []),
            "root_cause": original_analysis.get(
                "root_cause", "Escalate to human operator for manual investigation"
            ),
            "short_term_fix": original_analysis.get(
                "short_term_fix", "Escalate to human operator for manual investigation"
            ),
        }

    # ═══════════════════════════════════════════════════════════════════
    # METHOD 6 — apply_human_override
    #
    # Purpose:
    #   Called when classify_human_feedback() returned intent =
    #   "correct_rca_fix". The human has told the AI that the enterprise
    #   data it collected is correct, but the conclusion drawn from that
    #   data (the root cause and/or the suggested fix) is wrong — and the
    #   human has supplied the correct version themselves.
    #
    #   This method does NOT call any tools and does NOT call any LLM
    #   at all. It is the simplest of the new methods by design: the
    #   human has already done the reasoning work, so the AI's job here
    #   is just to accept the correction and merge it cleanly into the
    #   existing analysis dict, ready to be finalised.
    #
    # Why no LLM call here?
    #   The whole point of separating this from
    #   re_investigate_with_feedback() is to avoid the AI re-reasoning
    #   about something the human has already settled. If we ran this
    #   through an LLM "just to be safe", there is a real risk the LLM
    #   second-guesses the human's correction or rephrases it in a way
    #   that loses precision. A human SME's direct correction should be
    #   taken at face value, not re-interpreted by the same AI that got
    #   it wrong in the first place. This keeps the method fast, free
    #   (no API cost), and fully predictable.
    #
    # Design decisions:
    #   - Either corrected_root_cause or corrected_fix (or both) may be
    #     provided — classify_human_feedback() only fills in the field(s)
    #     the human actually corrected, leaving the other as None. This
    #     method must handle a human correcting only one of the two
    #     without accidentally erasing the other.
    #   - The "findings" (the enterprise data itself) and
    #     "operational_issue" are left UNCHANGED from the original
    #     analysis, because the human explicitly said the data is correct.
    #     Only root_cause and short_term_fix are touched.
    #   - We also record that this analysis was human-overridden, by
    #     adding a "human_override" flag to the returned dict. This is
    #     a forward-looking design choice: when we later build the
    #     feedback loop that writes to the vector database and reviews
    #     YAML promotion candidates, knowing WHICH incidents had a human
    #     override (and what the override was) is exactly the signal
    #     that loop needs. We are not building that loop yet, but we are
    #     making sure this method does not throw away the information
    #     that loop will eventually need.
    # ═══════════════════════════════════════════════════════════════════

    def apply_human_override(
        self,
        original_analysis:    dict,
        corrected_root_cause:  str | None,
        corrected_fix:         str | None,
    ) -> dict:
        """
        Accept a human's correction to the root cause and/or fix, without
        re-running any investigation or calling any LLM.

        Called after classify_human_feedback() returns
        intent == "correct_rca_fix". corrected_root_cause and
        corrected_fix should come from that result's
        "extracted_root_cause" and "extracted_fix" fields respectively.

        Args:
            original_analysis:
                The dict previously returned by investigate_and_analyse()
                (or re_investigate_with_feedback()). Its "findings" and
                "operational_issue" are carried forward unchanged, since
                the human confirmed the data itself is correct.

            corrected_root_cause:
                The human's corrected root cause text, or None if the
                human only corrected the fix and left the root cause
                as the AI originally stated it.

            corrected_fix:
                The human's corrected short-term fix text, or None if
                the human only corrected the root cause and left the
                fix as the AI originally suggested.

        Returns:
            A dict with the same five keys as investigate_and_analyse(),
            PLUS one extra key:
            {
                "operational_issue": str,   # unchanged from original
                "findings":          list,  # unchanged from original
                "reasoning_steps":   list,  # unchanged from original
                "root_cause":        str,   # human's correction, if given
                "short_term_fix":    str,   # human's correction, if given
                "human_override":    bool,  # always True for this method's output
            }
            This can still be passed directly into present_to_human() —
            the extra "human_override" key is simply ignored by
            present_to_human() since it only reads the five keys it knows about.
        """
        # Start by copying every field from the original analysis.
        # dict(original_analysis) creates a new dict with the same
        # key-value pairs — this means we do NOT modify the caller's
        # original_analysis dict by accident. Modifying a dict that was
        # passed in as an argument, instead of working on a copy, is a
        # common source of confusing bugs in Python — the caller would
        # see their own variable change unexpectedly. Always copying
        # first avoids this entirely.
        updated_analysis = dict(original_analysis)

        # Only overwrite root_cause if the human actually provided one.
        # If corrected_root_cause is None, we leave the original
        # root_cause untouched — this handles the case where the human
        # only corrected the fix, not the root cause.
        if corrected_root_cause is not None:
            updated_analysis["root_cause"] = corrected_root_cause

        # Same logic for the fix — only overwrite if a correction was given.
        if corrected_fix is not None:
            updated_analysis["short_term_fix"] = corrected_fix

        # Record that this analysis was human-overridden. This flag is
        # not used by present_to_human() today, but it is exactly the
        # kind of structured signal the future feedback loop (vector DB
        # embedding, YAML promotion review) will need in order to know
        # which incidents are good candidates for SME review — without
        # this flag, that future code would have no way to distinguish
        # "AI got it right first try" from "AI needed a human correction"
        # just by looking at the final analysis dict alone.
        updated_analysis["human_override"] = True

        return updated_analysis

    # ═══════════════════════════════════════════════════════════════════
    # METHOD 7 — finalize_to_servicenow
    #
    # Purpose:
    #   The last step in the MVP loop. Once a human has approved an
    #   analysis (either the AI's original analysis, or a version that
    #   was re-investigated or corrected along the way), this method
    #   writes the final outcome back into ServiceNow as a work note —
    #   the internal, customer-invisible record of what happened and why.
    #
    # Why this lives on SupervisorAgent, but the ServiceNow client does not:
    #   SupervisorAgent has deliberately never imported or known about
    #   ServiceNowClient anywhere else in this file — every other method
    #   only deals with YAML knowledge, tools, and LLMs. We are keeping
    #   that separation here too: this method does not create or store a
    #   ServiceNowClient on self. Instead, the caller passes in an
    #   already-constructed ServiceNowClient as a parameter, exactly the
    #   same pattern already used for "tools" in investigate_and_analyse().
    #   This means SupervisorAgent stays fully testable without ever
    #   needing real ServiceNow credentials — only this one method needs
    #   a real client, and only when it is actually called.
    #
    # How it works:
    #   - Takes the final analysis dict, the decision_type (how this
    #     analysis was reached — approved as-is, re-investigated, or
    #     human-overridden), and who approved it.
    #   - Formats all of this into one clear, readable work note string.
    #   - Calls servicenow_client.update_work_notes() to write it.
    #   - Returns nothing (None) — this method's entire purpose is the
    #     side effect of writing to ServiceNow, there is no decision
    #     left to make or data left to compute after that.
    #
    # Design decisions:
    #   - decision_type is one of "approved", "re_investigated", or
    #     "human_override" — this matches directly to the four intents
    #     classify_human_feedback() can return (minus "unclear", since
    #     an unclear response never reaches finalization — the agent
    #     asks for clarification instead and nothing is finalized yet).
    #   - We include the decision_type and human_agent_id directly in
    #     the work note text itself, not just in code logic, because
    #     work notes are what an auditor or another support engineer
    #     will actually read later. A work note that just says "Resolved"
    #     tells a future reader nothing about HOW it was resolved or who
    #     was involved — including this detail in the note itself is
    #     what makes the audit trail real and useful, not just a checkbox.
    # ═══════════════════════════════════════════════════════════════════

    def finalize_to_servicenow(
        self,
        servicenow_client,
        sys_id:         str,
        final_analysis: dict,
        decision_type:  str,
        human_agent_id: str,
    ) -> None:
        """
        Write the finalized analysis to a ServiceNow incident's work
        notes — the internal, customer-invisible record of the AI
        investigation and the human decision that resolved it.

        This is the final method in the MVP pipeline. It is called once
        a human has approved an analysis, whether that analysis is the
        AI's original investigation, a re-investigated version, or a
        version corrected by direct human override.

        Args:
            servicenow_client:
                An already-constructed ServiceNowClient instance (see
                integrations/servicenow_client.py). SupervisorAgent does
                not create or store this itself — it is passed in here,
                the same way "tools" is passed into investigate_and_analyse().
                This keeps SupervisorAgent fully decoupled from any one
                specific ServiceNow connection or set of credentials.

            sys_id:
                ServiceNow's internal unique ID for this incident record
                (not the human-readable incident number like INC0010002).
                This is the same sys_id used by
                ServiceNowClient.update_work_notes().

            final_analysis:
                The analysis dict to record — this can be the dict
                returned by investigate_and_analyse(),
                re_investigate_with_feedback(), or apply_human_override().
                All three return the same shape, so this method does not
                need to know or care which one produced it.

            decision_type:
                One of "approved", "re_investigated", or "human_override".
                Describes HOW this final analysis was reached — this is
                what makes the work note an audit trail, not just a
                status update. Any other value is still written (this
                method does not reject unexpected values), but the three
                listed are the ones the rest of this MVP actually produces.

            human_agent_id:
                Identifies which human agent made the final decision —
                e.g. a Slack username, email address, or ServiceNow user
                ID, depending on what identity information is available
                to the calling code at the point this method is invoked.

        Returns:
            None. This method's entire job is the side effect of writing
            to ServiceNow — there is nothing further to compute or decide
            after that, so there is nothing meaningful to return.

        Raises:
            requests.exceptions.HTTPError:
                If the underlying ServiceNow API call fails (e.g.
                invalid sys_id, permission denied, instance unreachable).
                This method does not catch or hide that error — the
                caller (e.g. the Slack event handler we will build next)
                is responsible for deciding what to do if writing to
                ServiceNow fails, such as retrying or alerting a human.
        """

        # ── Step 1: Build a clear, human-readable work note ──────────
        # We format every field explicitly with a label, rather than
        # just dumping the raw analysis dict as JSON into the work
        # note. A future support engineer or auditor reading this in
        # ServiceNow's UI should not need to know Python or JSON to
        # understand what happened — plain labelled text is the right
        # format for a field that real humans will actually read.
        note_lines = [
            "=== AI Operations Assistant — Final Analysis ===",
            f"Decision type: {decision_type}",
            f"Approved by: {human_agent_id}",
            "",
            f"Operational issue: {final_analysis.get('operational_issue', 'N/A')}",
            "",
            "Findings:",
        ]

        # .get() with a default empty list means this never crashes
        # even if "findings" happens to be missing from final_analysis —
        # the same defensive pattern used throughout present_to_human().
        for finding in final_analysis.get("findings", []):
            note_lines.append(f"  - {finding}")

        note_lines.extend([
            "",
            f"Root cause: {final_analysis.get('root_cause', 'N/A')}",
            "",
            f"Fix applied: {final_analysis.get('short_term_fix', 'N/A')}",
        ])

        # If this analysis was a human override, make that fact
        # explicit and prominent in the note — this is exactly the
        # kind of incident a future evaluation agent (the one we
        # discussed but parked for later) would want to easily find
        # when reviewing which incidents needed a human correction.
        if final_analysis.get("human_override"):
            note_lines.append("")
            note_lines.append(
                "NOTE: Root cause and/or fix were corrected directly by "
                "the human agent — not the AI's original conclusion."
            )

        # "\n".join() combines all the lines into one multi-line string,
        # the same technique used to build the Decision Card in
        # present_to_human().
        work_note_text = "\n".join(note_lines)

        # ── Step 2: Write it to ServiceNow ────────────────────────────
        # We do not wrap this in a try/except here. If writing to
        # ServiceNow fails, the caller needs to know immediately — for
        # example, a Slack event handler might want to tell the human
        # "I couldn't save this to ServiceNow, please try again" rather
        # than silently pretending the finalization succeeded when it
        # did not. Swallowing the error here would hide a real failure
        # from the part of the system best positioned to react to it.
        servicenow_client.update_work_notes(
            sys_id=sys_id,
            note_text=work_note_text,
        )

        print(f"  ✅ Finalized to ServiceNow (sys_id={sys_id}, "
              f"decision_type={decision_type}, by={human_agent_id})")
