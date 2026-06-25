"""
incident_poller.py
====================
Polls ServiceNow for new incidents, runs the AI investigation pipeline
for each one, creates a dedicated Slack channel per incident, and posts
the Decision Card there.

This file handles the OUTBOUND side of the MVP loop:
    detect → investigate → create channel → post Decision Card

The INBOUND side (receiving human replies and acting on them) is
handled by slack_listener.py — these two files work together as
two independent processes: the poller detects and posts, the listener
receives and routes.

The two processes share an IncidentStore instance that maps Slack
channel_ids to incident context — this is how the listener knows
which incident a human reply belongs to. Since both processes run
in the same Python process space (the poller is imported by the
listener), they share the same in-memory store object automatically.

Run from project root:
    python -m integrations.incident_poller   (poller only — for testing)
    uvicorn slack_listener:app               (full system — starts both)
"""

import os
import sys
import time

import dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.supervisor_agent import SupervisorAgent
from integrations.servicenow_client import ServiceNowClient
from integrations.slack_client import SlackClient
from integrations.incident_store import IncidentStore
from tools.inventory_tools import (
    get_warehouse_availability,
    get_allocation_queue_details,
)

dotenv.load_dotenv()

# ════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════

try:
    SERVICENOW_INSTANCE_URL = os.environ["SERVICENOW_INSTANCE_URL"]
    SERVICENOW_USERNAME     = os.environ["SERVICENOW_USERNAME"]
    SERVICENOW_PASSWORD     = os.environ["SERVICENOW_PASSWORD"]
    SLACK_BOT_TOKEN         = os.environ["SLACK_BOT_TOKEN"]
except KeyError as e:
    print(f"Missing required environment variable: {e}")
    print("Set SERVICENOW_INSTANCE_URL, SERVICENOW_USERNAME, "
          "SERVICENOW_PASSWORD, SLACK_BOT_TOKEN in your .env file.")
    raise SystemExit(1)

# How often to check ServiceNow for new incidents (seconds).
# Default: 300 (5 minutes). Override via POLL_INTERVAL_SECONDS in .env.
POLL_INTERVAL_SECONDS = int(os.environ.get("POLL_INTERVAL_SECONDS", "300"))

# The ServiceNow assignment group the AI agent monitors.
# Only incidents assigned to this group AND in "New" state are picked up.
# Set via SERVICENOW_ASSIGNMENT_GROUP in .env.
# If not set, ALL New incidents are processed (not recommended for production).
ASSIGNMENT_GROUP = os.environ.get("SERVICENOW_ASSIGNMENT_GROUP", "")

# Slack user ID of the default human agent to invite to every
# new incident channel. This is the MVP Option A approach —
# one fixed agent receives all incidents.
#
# Find your Slack user ID:
#   Slack → click your profile picture → Profile
#   → three-dot menu (•••) → Copy member ID
#   It starts with "U", e.g. "U0B65SC463C"
#
# Production upgrade (Phase 2):
#   This will be replaced by UserResolver.get_slack_id() which
#   looks up the assigned agent dynamically based on workload.
DEFAULT_AGENT_SLACK_ID = os.environ.get("SLACK_DEFAULT_AGENT_USER_ID", "")

# Marker written to ServiceNow work notes to flag an incident as
# already picked up, preventing it from being processed twice.
PICKUP_MARKER = "AI Agent: investigation started"

# Maximum number of re-investigation rounds before mandatory escalation.
MAX_REINVESTIGATION_ROUNDS = 2

TOOLS = [
    get_warehouse_availability,
    get_allocation_queue_details,
]


def _build_user_query(incident: dict) -> str:
    """Combine ServiceNow incident fields into one plain-text query string."""
    short_description = incident.get("short_description", "") or ""
    description        = incident.get("description", "") or ""
    if description and description != short_description:
        return f"{short_description}\n\n{description}"
    return short_description


def process_incident(
    agent:        SupervisorAgent,
    snow_client:  ServiceNowClient,
    slack_client: SlackClient,
    store:        IncidentStore,
    incident:     dict,
) -> None:
    """
    Process one incident end to end:
        1. Mark as picked up in ServiceNow
        2. Run the investigation pipeline
        3. Create a dedicated Slack channel
        4. Post the Decision Card to that channel
        5. Save all context to the store so the listener can route replies
    """
    incident_number = incident.get("number", "UNKNOWN")
    sys_id          = incident.get("sys_id")

    print(f"\n{'=' * 70}")
    print(f"Processing incident: {incident_number}")
    print(f"{'=' * 70}")

    # ── Step 1: Mark as picked up and set state to In Progress ─────
    # Done BEFORE investigation starts — if the process crashes mid-
    # investigation, the incident won't be re-processed on the next poll.
    # Setting state to "In Progress" immediately signals to anyone
    # looking at the ServiceNow queue that this incident is being
    # actively worked by the AI agent, not sitting unattended in New.
    try:
        snow_client.update_work_notes(sys_id, PICKUP_MARKER)
        snow_client.update_state(
            sys_id,
            ServiceNowClient.STATE_IN_PROGRESS
        )
    except Exception as e:
        print(f"  [ERROR] Could not mark incident as picked up: {e}")
        return

    # ── Step 2: Build query and run investigation ─────────────────────
    user_query = _build_user_query(incident)
    if not user_query.strip():
        print(f"  [WARN] Incident {incident_number} has no description — skipping")
        return

    print(f"  Query: {user_query[:100]}{'...' if len(user_query) > 100 else ''}")

    try:
        kb_result = agent.select_knowledge_base(user_query)

        if kb_result["selected_count"] == 0:
            print(f"  [INFO] No matching knowledge base found")
            # Still create a channel and notify — a human should know
            # the AI couldn't classify this incident
            channel_id = slack_client.create_incident_channel(incident_number)
            slack_client.post_to_channel(
                channel_id,
                f"🤔 *AI Agent reviewed {incident_number}* but found no "
                f"matching knowledge base for this issue type.\n"
                f"Reasoning: {kb_result['overall_reasoning']}\n"
                f"This incident needs manual triage."
            )
            return

        analysis = agent.investigate_and_analyse(
            user_query=user_query,
            selected_yamls=kb_result["selected_yamls"],
            tools=TOOLS,
        )

        card = agent.present_to_human(
            user_query=user_query,
            analysis=analysis,
        )

    except Exception as e:
        print(f"  [ERROR] Investigation pipeline failed: {e}")
        try:
            channel_id = slack_client.create_incident_channel(incident_number)
            slack_client.post_to_channel(
                channel_id,
                f"⚠️ *AI Agent error on {incident_number}*\n"
                f"Investigation failed: {e}\nNeeds manual review."
            )
        except Exception:
            pass
        return

    # ── Step 3: Create dedicated channel and post Decision Card ───────
    try:
        channel_id = slack_client.create_incident_channel(incident_number)
        slack_client.post_to_channel(
            channel_id,
            f"🔔 *New incident: {incident_number}*\n```\n{card}\n```"
        )
        print(f"  ✅ Decision Card posted to #{incident_number.lower()}")

        # Invite the default human agent to the channel so they
        # are automatically added without needing to manually browse
        # and join. In production this will be the assigned agent's
        # Slack ID, looked up via UserResolver. For MVP it is a
        # single fixed user from .env.
        if DEFAULT_AGENT_SLACK_ID:
            slack_client.invite_user_to_channel(
                channel_id=channel_id,
                user_id=DEFAULT_AGENT_SLACK_ID,
            )
        else:
            print("  [WARN] SLACK_DEFAULT_AGENT_USER_ID not set — "
                  "human agent must join the channel manually")
    except Exception as e:
        print(f"  [ERROR] Could not create channel or post card: {e}")
        return

    # ── Step 4: Save context to store ─────────────────────────────────
    # This is the critical step that connects the poller to the listener.
    # The listener will look up channel_id in this store when a human
    # replies — without this entry, the listener can't route the reply.
    store.save_incident(
        channel_id=channel_id,
        incident_number=incident_number,
        sys_id=sys_id,
        user_query=user_query,
        analysis=analysis,
        selected_yamls=kb_result["selected_yamls"],
    )


def run_poll_loop(store: IncidentStore) -> None:
    """
    Poll ServiceNow continuously and process any new incidents found.

    Parameters
    ----------
    store : IncidentStore
        The shared store instance — passed in rather than created here
        so that the same store object is used by both the poller and
        the listener (they run in the same process when started via
        slack_listener.py).
    """
    print("=" * 70)
    print("AI Operations Assistant — ServiceNow Poller")
    print("=" * 70)
    print(f"Polling every {POLL_INTERVAL_SECONDS} seconds")
    if ASSIGNMENT_GROUP:
        print(f"Assignment group filter: {ASSIGNMENT_GROUP}")
    else:
        print("WARNING: No assignment group set — ALL New incidents will be picked up")
    print("Press Ctrl+C to stop.\n")

    agent = SupervisorAgent()

    snow_client = ServiceNowClient(
        instance_url=SERVICENOW_INSTANCE_URL,
        username=SERVICENOW_USERNAME,
        password=SERVICENOW_PASSWORD,
    )

    slack_client = SlackClient(bot_token=SLACK_BOT_TOKEN)

    while True:
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
              f"Checking for new incidents...")
        try:
            new_incidents = snow_client.get_new_incidents(
                assignment_group=ASSIGNMENT_GROUP
            )
        except Exception as e:
            print(f"  [ERROR] Could not fetch incidents: {e}")
            new_incidents = []

        if not new_incidents:
            print("  No new incidents found.")
        else:
            print(f"  Found {len(new_incidents)} new incident(s).")
            for incident in new_incidents:
                process_incident(
                    agent, snow_client, slack_client, store, incident
                )

        print(f"  Sleeping for {POLL_INTERVAL_SECONDS} seconds...")
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    # When run directly (for testing the poller in isolation),
    # create a fresh store just for this process.
    store = IncidentStore()
    try:
        run_poll_loop(store)
    except KeyboardInterrupt:
        print("\n\nPoller stopped. Goodbye.")
