"""
supervisor_agent.py
===================
Business Operations Support AI Agent 

This agent investigates customer complaints by:
  1. Selecting relevant knowledge files (YAML) based on the complaint
  2. Invoking enterprise diagnostic tools to gather data
  3. Reasoning over the data to produce a structured analysis
  4. Formatting the analysis into a human-readable Decision Card

The agent follows a "Human as Collaborator" model — it investigates and
surfaces findings, but a human always makes the final decision.

Methods (in call order):
    1. select_knowledge_base()    — find relevant YAML knowledge files
    2. investigate_and_analyse()  — run tools, reason, return structured dict
    3. present_to_human()         — format dict into a readable Decision Card

Usage:
    agent  = SupervisorAgent()
    kb     = agent.select_knowledge_base(user_query)
    result = agent.investigate_and_analyse(user_query, kb["selected_yamls"], tools)
    card   = agent.present_to_human(user_query, result)
    print(card)
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
