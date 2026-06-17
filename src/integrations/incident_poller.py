"""
incident_poller.py
====================
This is the MAIN ENTRY POINT for the MVP. Running this file starts a
loop that:

  1. Polls ServiceNow every N seconds, looking for new incidents
  2. For each new incident found, runs the full AI investigation pipeline
  3. Posts the resulting Decision Card to a single, pre-configured
     Slack channel
  4. Marks the incident as "picked up" in ServiceNow so it is not
     processed again on the next poll

This file does NOT handle the human's reply (approve / request more
data / correct RCA). That happens in a separate part of the system —
a Slack Events listener — because reading a Slack message that a human
just typed requires Slack to actively notify our code (an event), which
is a different mechanism from sending a message (which is all our
SlackClient currently does via a webhook). For the MVP, this file
covers steps 1-3 of the pipeline: detect → investigate → present.

Why polling, and why this is explicitly an MVP simplification:
    The production-grade way to detect new ServiceNow incidents is to
    configure a ServiceNow Business Rule that calls our system via a
    webhook the INSTANT an incident is created — this means zero delay
    and zero wasted API calls checking "is there anything new?" when
    there often isn't. Polling, by contrast, means we ask ServiceNow
    "anything new?" on a fixed schedule regardless of whether anything
    has actually happened, which wastes API calls during quiet periods
    and introduces up to POLL_INTERVAL_SECONDS of delay before a new
    incident is even noticed. We are using polling here because it
    requires zero configuration on the ServiceNow side beyond API
    credentials — no Business Rules, no webhook setup, no admin access
    to ServiceNow's configuration screens. This makes the MVP runnable
    by anyone with basic ServiceNow API credentials. The event-driven
    approach is the clear, well-understood next step for a production
    deployment, and is intentionally deferred.

Run from project root:
    python -m integrations.incident_poller
"""

import os
import sys
import time

import dotenv

# Add the parent directory to the module search path so that
# "from agents.supervisor_agent import ..." resolves correctly,
# exactly the same pattern used in the test files throughout this project.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.supervisor_agent import SupervisorAgent
from integrations.servicenow_client import ServiceNowClient
from integrations.slack_client import SlackClient
from tools.inventory_tools import get_warehouse_availability, get_allocation_queue_details

dotenv.load_dotenv()


# ════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════
#
# All values below are read from environment variables (.env file),
# never hardcoded — this follows the same security pattern already
# established in servicenow_client.py and test_servicenow_to_slack.py.
# If any required variable is missing, we fail immediately with a
# clear error message rather than starting a poll loop that would
# only fail later, possibly silently, deep inside a request.
# ════════════════════════════════════════════════════════════════════

try:
    SERVICENOW_INSTANCE_URL = os.environ["SERVICENOW_INSTANCE_URL"]
    SERVICENOW_USERNAME     = os.environ["SERVICENOW_USERNAME"]
    SERVICENOW_PASSWORD     = os.environ["SERVICENOW_PASSWORD"]
    SLACK_WEBHOOK_URL       = os.environ["SLACK_WEBHOOK_URL"]
except KeyError as e:
    print(f"Missing required environment variable: {e}")
    print("Set SERVICENOW_INSTANCE_URL, SERVICENOW_USERNAME, "
          "SERVICENOW_PASSWORD, SLACK_WEBHOOK_URL in your .env file.")
    raise SystemExit(1)

# How often to check ServiceNow for new incidents, in seconds.
# os.environ.get() with a default means this is optional — if not
# set in .env, we fall back to checking every 300 seconds (5 minutes).
# int(...) converts the value from a string (all env vars are strings)
# to a whole number, since time.sleep() needs a number, not text.
#
# Why 5 minutes as the default:
#   This is a balance between responsiveness and API courtesy. Most
#   ServiceNow instances rate-limit API calls, and polling too
#   frequently (e.g. every few seconds) burns through that quota for
#   no real benefit — incidents are not created so rapidly that a few
#   minutes of delay meaningfully harms the user experience for an
#   MVP demo. Five minutes is a commonly used default for polling
#   integrations in ITSM tooling generally. This is fully configurable
#   via POLL_INTERVAL_SECONDS in .env if a shorter or longer interval
#   is needed for a specific demo or environment.
POLL_INTERVAL_SECONDS = int(os.environ.get("POLL_INTERVAL_SECONDS", "300"))

# The marker text we write to work notes to mark an incident as
# "already picked up by the AI agent". This must exactly match the
# pickup_marker string used inside get_new_incidents() in
# servicenow_client.py — if these two ever go out of sync, the agent
# would either re-process the same incident forever, or never
# recognise its own marker and skip incidents incorrectly.
PICKUP_MARKER = "AI Agent: investigation started"

# All tools the reasoning LLM is allowed to call during investigation.
# This is the same TOOLS list pattern used throughout the test files —
# defined once, passed into investigate_and_analyse() as a parameter.
TOOLS = [
    get_warehouse_availability,
    get_allocation_queue_details,
]


def _build_user_query(incident: dict) -> str:
    """
    Convert a raw ServiceNow incident record into the plain-text
    "customer complaint" string that select_knowledge_base() and
    investigate_and_analyse() expect as their first input.

    ServiceNow incidents have several text fields (short_description,
    description) — we combine them so the AI agent has as much context
    as possible, similar to how a human support agent would read both
    the short summary and the full description before starting work.

    Args:
        incident: a single incident dict, as returned by
                  ServiceNowClient.get_new_incidents()

    Returns:
        A single combined string suitable as the "user_query" argument
        to select_knowledge_base().
    """
    short_description = incident.get("short_description", "") or ""
    description        = incident.get("description", "") or ""

    # If both fields have content, combine them with a clear separator.
    # If only one has content, just use that one — no need for an
    # empty trailing separator.
    if description and description != short_description:
        return f"{short_description}\n\n{description}"
    return short_description


def process_incident(
    agent:        SupervisorAgent,
    snow_client:  ServiceNowClient,
    slack_client: SlackClient,
    incident:     dict,
) -> None:
    """
    Run the full investigation pipeline for ONE incident and post the
    result to Slack. This function is intentionally kept separate from
    the polling loop itself (run_poll_loop(), below) so that the logic
    for "what do we do with one incident" is easy to read, test, and
    reason about independently of "how often do we check for incidents".

    Args:
        agent:        the SupervisorAgent instance doing the investigation
        snow_client:  ServiceNowClient instance, used to mark the
                      incident as picked up once processing begins
        slack_client: SlackClient instance, used to post the Decision Card
        incident:     the raw incident dict from ServiceNow
    """
    incident_number = incident.get("number", "UNKNOWN")
    sys_id          = incident.get("sys_id")

    print(f"\n{'=' * 70}")
    print(f"Processing incident: {incident_number}")
    print(f"{'=' * 70}")

    # ── Step 1: Mark as picked up FIRST, before investigation starts ──
    # We deliberately do this BEFORE running the (potentially slow)
    # investigation, not after. If we marked it as picked up only at
    # the very end, and the process crashed midway through investigation
    # (e.g. an API timeout), the next poll cycle would see this same
    # incident as "New" again and start a second, duplicate investigation
    # for the same incident. Marking it as picked up immediately means
    # at worst we lose one incident's analysis to a crash — we never
    # process the same incident twice.
    try:
        snow_client.update_work_notes(sys_id, PICKUP_MARKER)
    except Exception as e:
        # If we can't even write the pickup marker, something is wrong
        # with our ServiceNow connection — better to stop processing
        # this incident now than to proceed and risk processing it
        # again on the next poll.
        print(f"  [ERROR] Could not mark incident as picked up: {e}")
        return

    # ── Step 2: Build the user query from the incident's text fields ──
    user_query = _build_user_query(incident)

    if not user_query.strip():
        # An incident with no description at all gives the AI agent
        # nothing to investigate. Rather than sending a nonsensical
        # empty query into the pipeline, we log this and skip —
        # a human will still see this incident through normal
        # ServiceNow channels; we are just not generating an AI
        # analysis for it.
        print(f"  [WARN] Incident {incident_number} has no description — skipping AI analysis")
        return

    print(f"  Query: {user_query[:100]}{'...' if len(user_query) > 100 else ''}")

    # ── Step 3: Run the investigation pipeline ─────────────────────────
    # This is exactly the same three-method sequence used throughout
    # the test files — select_knowledge_base() then investigate_and_analyse()
    # then present_to_human(). We are simply calling it here against a
    # real, live ServiceNow incident instead of a hardcoded test string.
    try:
        kb_result = agent.select_knowledge_base(user_query)

        if kb_result["selected_count"] == 0:
            # No relevant YAML found for this incident type. We still
            # want a human to know the AI looked at it and could not
            # find a matching playbook, rather than silently doing nothing.
            print(f"  [INFO] No matching knowledge base found for this incident")
            slack_client.send_message(
                f"🤔 *AI Agent reviewed {incident_number}* but found no "
                f"matching knowledge base entry for this type of issue.\n"
                f"Reasoning: {kb_result['overall_reasoning']}\n"
                f"This incident may need manual triage."
            )
            return

        analysis = agent.investigate_and_analyse(
            user_query=user_query,
            selected_yamls=kb_result["selected_yamls"],
            tools=TOOLS,
        )

        decision_card = agent.present_to_human(
            user_query=user_query,
            analysis=analysis,
        )

    except Exception as e:
        # If anything in the pipeline fails (LLM API error, tool error,
        # malformed YAML, etc.), we do not want the entire poll loop to
        # crash and stop processing every other incident. We log the
        # error and notify Slack so a human knows this incident needs
        # manual attention, then move on.
        print(f"  [ERROR] Investigation pipeline failed: {e}")
        slack_client.send_message(
            f"⚠️ *AI Agent error while processing {incident_number}*\n"
            f"The investigation pipeline failed: {e}\n"
            f"This incident needs manual review."
        )
        return

    # ── Step 4: Post the Decision Card to Slack ────────────────────────
    # We prepend the incident number clearly at the top of the Slack
    # message (separate from the Decision Card itself) so that when a
    # human later replies in the channel, anyone reading the thread can
    # immediately see which ServiceNow incident this conversation is
    # about — this becomes more important as more incidents flow
    # through the same single MVP channel.
    slack_message = (
        f"🔔 *New incident analysis ready: {incident_number}*\n"
        f"```\n{decision_card}\n```"
    )

    try:
        slack_client.send_message(slack_message)
        print(f"  ✅ Decision Card posted to Slack for {incident_number}")
    except Exception as e:
        print(f"  [ERROR] Could not post to Slack: {e}")


def run_poll_loop() -> None:
    """
    The main loop. Runs forever (until manually stopped with Ctrl+C),
    checking ServiceNow for new incidents every POLL_INTERVAL_SECONDS
    and processing any it finds.

    This function deliberately contains very little logic of its own —
    it just repeatedly calls get_new_incidents() and, for each one
    found, calls process_incident(). All the actual investigation logic
    lives in process_incident() and in SupervisorAgent itself. Keeping
    this loop simple means the polling mechanism and the investigation
    logic can be understood, tested, and changed independently of each
    other.
    """
    print("=" * 70)
    print("AI Operations Assistant — ServiceNow Poller")
    print("=" * 70)
    print(f"Polling {SERVICENOW_INSTANCE_URL} every {POLL_INTERVAL_SECONDS} seconds")
    print("Press Ctrl+C to stop.\n")

    # Create one shared agent, ServiceNow client, and Slack client for
    # the entire lifetime of this poll loop, rather than creating new
    # ones on every cycle. Creating a new SupervisorAgent every cycle
    # would mean reconnecting to OpenAI's API client objects
    # repeatedly for no benefit — these objects are designed to be
    # reused across many calls.
    agent = SupervisorAgent()

    snow_client = ServiceNowClient(
        instance_url=SERVICENOW_INSTANCE_URL,
        username=SERVICENOW_USERNAME,
        password=SERVICENOW_PASSWORD,
    )

    slack_client = SlackClient(webhook_url=SLACK_WEBHOOK_URL)

    # This loop runs forever. "while True" is intentional here — this
    # program is meant to run continuously in the background (e.g. as
    # a long-running process, a Docker container, or a scheduled
    # service), not to run once and exit. The only way out of this
    # loop in normal operation is the user pressing Ctrl+C, which
    # Python turns into a KeyboardInterrupt exception that we catch
    # below, outside this function.
    while True:
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Checking for new incidents...")

        try:
            new_incidents = snow_client.get_new_incidents()
        except Exception as e:
            # If ServiceNow is temporarily unreachable, we do not want
            # to crash the entire poller — we log the error and simply
            # try again on the next cycle. A temporary network issue
            # should not require manually restarting this process.
            print(f"  [ERROR] Could not fetch new incidents: {e}")
            new_incidents = []

        if not new_incidents:
            print("  No new incidents found.")
        else:
            print(f"  Found {len(new_incidents)} new incident(s).")

            for incident in new_incidents:
                # Each incident is processed independently. If one
                # incident's processing fails for any reason,
                # process_incident() already handles that internally
                # and returns — it will not raise an exception that
                # stops us from processing the remaining incidents in
                # this same batch.
                process_incident(agent, snow_client, slack_client, incident)

        print(f"  Sleeping for {POLL_INTERVAL_SECONDS} seconds...")
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    try:
        run_poll_loop()
    except KeyboardInterrupt:
        # Ctrl+C raises KeyboardInterrupt. Catching it here means the
        # program exits cleanly with a friendly message instead of
        # printing a long, alarming Python traceback to someone who
        # simply wanted to stop the poller normally.
        print("\n\nPoller stopped by user. Goodbye.")
