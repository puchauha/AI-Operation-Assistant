"""
test_first_analysis_to_slack.py
==================================
A minimal, standalone script that does ONLY the first three steps of
the pipeline:

    1. Fetch one incident from ServiceNow
    2. Run the investigation pipeline
    3. Post the resulting Decision Card to Slack

This deliberately stops there — no human reply loop, no
classification, no finalization. The purpose is to isolate and verify
ONE thing in complete isolation: does the first Decision Card render
correctly in the real Slack channel?

This is useful right now specifically because we have NOT yet built
a way to receive a reply back from Slack (that is the next piece of
work, the "Slack listener" / Events API receiver). Everything this
script does is already fully real and working — there is nothing
simulated here, unlike run_mvp_demo.py which simulates the human's
reply via a terminal prompt.

Run from project root:
    python -m run_first_analysis_test
"""

import os
import sys

import dotenv

sys.path.insert(0, os.path.dirname(__file__))

from agents.supervisor_agent import SupervisorAgent
from integrations.servicenow_client import ServiceNowClient
from integrations.slack_client import SlackClient
from tools.inventory_tools import get_warehouse_availability, get_allocation_queue_details

dotenv.load_dotenv()


try:
    SERVICENOW_INSTANCE_URL = os.environ["SERVICENOW_INSTANCE_URL"]
    SERVICENOW_USERNAME     = os.environ["SERVICENOW_USERNAME"]
    SERVICENOW_PASSWORD     = os.environ["SERVICENOW_PASSWORD"]
    SLACK_WEBHOOK_URL       = os.environ["SLACK_WEBHOOK_URL"]
except KeyError as e:
    print(f"Missing required environment variable: {e}")
    raise SystemExit(1)

DEMO_INCIDENT_NUMBER = os.environ.get("SERVICENOW_TEST_INCIDENT", "INC0010002")

TOOLS = [
    get_warehouse_availability,
    get_allocation_queue_details,
]


def main():
    print("=" * 70)
    print("First Analysis → Slack — isolated test")
    print("=" * 70)
    print(f"Target incident: {DEMO_INCIDENT_NUMBER}\n")

    agent       = SupervisorAgent()
    snow_client = ServiceNowClient(
        instance_url=SERVICENOW_INSTANCE_URL,
        username=SERVICENOW_USERNAME,
        password=SERVICENOW_PASSWORD,
    )
    slack_client = SlackClient(webhook_url=SLACK_WEBHOOK_URL)

    # ── Step 1: Fetch the incident ────────────────────────────────
    print("[1/3] Fetching incident from ServiceNow...")
    incident = snow_client.get_incident_by_number(DEMO_INCIDENT_NUMBER)

    if not incident:
        print(f"  [ERROR] No incident found with number {DEMO_INCIDENT_NUMBER}")
        return

    short_description = incident.get("short_description", "") or ""
    description        = incident.get("description", "") or ""
    user_query = (
        f"{short_description}\n\n{description}"
        if description and description != short_description
        else short_description
    )

    if not user_query.strip():
        print("  [ERROR] This incident has no description text to investigate.")
        return

    print(f"  Query: {user_query}\n")

    # ── Step 2: Run the investigation ───────────────────────────────
    print("[2/3] Running investigation pipeline...")
    kb_result = agent.select_knowledge_base(user_query)

    if kb_result["selected_count"] == 0:
        print(f"  [INFO] No matching knowledge base found. "
              f"Reasoning: {kb_result['overall_reasoning']}")
        return

    analysis = agent.investigate_and_analyse(
        user_query=user_query,
        selected_yamls=kb_result["selected_yamls"],
        tools=TOOLS,
    )

    card = agent.present_to_human(user_query=user_query, analysis=analysis)
    print("\nDecision Card (as generated):")
    print(card)

    # ── Step 3: Post to Slack and stop ──────────────────────────────
    print("\n[3/3] Posting to Slack...")
    slack_client.send_message(
        f"🔔 *Isolated test — incident {DEMO_INCIDENT_NUMBER}*\n"
        f"```\n{card}\n```"
    )
    print("\n✅ Posted. Check your Slack channel now and confirm:")
    print("   - Box-drawing characters (╔ ═ ║) render legibly")
    print("   - Emoji (✅ ✏️ 🔺 ❓ 📋 🔍 📊 🧠 🎯 🛠) render as real emoji")
    print("   - Long lines are not awkwardly cut off or wrapped")
    print("\nThis script stops here intentionally — no human reply is")
    print("requested or processed. That capability does not exist yet.")


if __name__ == "__main__":
    main()
