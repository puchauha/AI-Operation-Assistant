"""
run_mvp_demo.py
=================
End-to-end demonstration of the complete MVP loop, with the human's
Slack reply SIMULATED via a typed input rather than a real Slack
Events listener.

Why simulate the human reply for now:
    Receiving a real reply from Slack requires a running web server
    that Slack can call (a webhook receiver), a registered Slack App
    with Event Subscriptions configured, and — for local development —
    a tool like ngrok to expose that server to the public internet.
    That is real infrastructure with its own setup and its own failure
    modes (signature verification, request retries, etc.), and is
    intentionally being treated as a SEPARATE next step, not bundled
    into this first MVP pass.

    This script proves out everything else first: ServiceNow detection,
    investigation, Decision Card formatting, posting to Slack, intent
    classification, and all three decision-handling paths
    (approve / request more data / correct RCA & fix) plus the
    finalization back to ServiceNow. The ONLY thing being simulated is
    how the human's reply text reaches our code — everything downstream
    of that text is the real, fully working pipeline.

What this script actually does:
    1. Fetches ONE real incident from ServiceNow (the same way
       incident_poller.py would, but just once, not in a loop)
    2. Runs the real investigation pipeline against it
    3. Posts the real Decision Card to your real Slack channel
    4. Asks YOU, in the terminal, to type what the human would have
       replied in Slack
    5. Runs that reply through the REAL classify_human_feedback()
    6. Routes to the REAL handler for whichever intent was detected
    7. Writes the REAL final outcome back to ServiceNow

Run from project root:
    python -m run_mvp_demo
"""

import os
import sys

import dotenv

sys.path.insert(0, os.path.dirname(__file__))

from agents.supervisor_agent import SupervisorAgent
from integrations.servicenow_client import ServiceNowClient
from integrations.slack_client import SlackClient
from tools.inventory_tools import (
    get_warehouse_availability,
    get_allocation_queue_details,
)
from tools.enterprise_tools import (
    get_shipment_status, get_proof_of_delivery,
    get_rma_status, get_credit_note_status,
    get_account_credit_status, get_payment_allocation_status,
    get_stock_levels, get_replenishment_orders,
    get_edi_transaction_status, get_partner_gateway_status,
    get_contract_price, get_applied_price_on_order,
    get_customs_clearance_status, get_trade_document_status,
    get_invoice_status, get_three_way_match_result,
    get_warranty_record, get_claim_status,
)

dotenv.load_dotenv()


# ════════════════════════════════════════════════════════════════════
# CONFIGURATION — same pattern as incident_poller.py
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

# Which incident to run this demo against. Unlike incident_poller.py
# (which finds NEW incidents automatically), this demo script targets
# one specific, known incident number, since the point here is to walk
# through one complete conversation, not to process a whole batch.
DEMO_INCIDENT_NUMBER = os.environ.get("SERVICENOW_TEST_INCIDENT", "INC0010002")

TOOLS = [
    get_warehouse_availability, get_allocation_queue_details,
    get_stock_levels, get_replenishment_orders,
    get_shipment_status, get_proof_of_delivery,
    get_rma_status, get_credit_note_status,
    get_account_credit_status, get_payment_allocation_status,
    get_edi_transaction_status, get_partner_gateway_status,
    get_contract_price, get_applied_price_on_order,
    get_customs_clearance_status, get_trade_document_status,
    get_invoice_status, get_three_way_match_result,
    get_warranty_record, get_claim_status,
]

# Maximum number of re-investigation rounds before we stop looping and
# force escalation. This is the cap we agreed on during MVP design —
# without it, a human could in theory keep asking for more data forever
# and the demo (or the real system) would never reach a decision.
# No cap on re-investigation rounds — open-ended collaboration
# is the core design of the Human as Companion model.
# The terminal demo loop runs until the human approves or
# corrects the RCA, however many rounds that takes.


def _ask_human_for_reply(round_number: int) -> str:
    """
    Stand-in for "wait for a real Slack reply". Prints a prompt to the
    terminal and waits for the person running this demo to type a
    response, exactly as if they were the human agent replying in Slack.

    This is the ONLY simulated part of this entire script — everything
    the typed text triggers afterwards is the real pipeline.
    """
    print("\n" + "─" * 70)
    print(f"  [SIMULATED SLACK REPLY — round {round_number}]")
    print("  Type the human agent's reply as if typing in Slack.")
    print("  Try things like:")
    print("    - \"Looks good, approved\"")
    print("    - \"Can you also check the allocation queue position?\"")
    print("    - \"Data is correct but root cause is wrong, it's "
          "actually a payment hold. Fix: release the hold.\"")
    print("─" * 70)
    return input("  Human reply > ").strip()


def run_demo() -> None:
    """
    Walk through one complete incident from detection to finalization,
    pausing at the human-reply step to ask for typed input instead of
    waiting for a real Slack event.
    """
    print("=" * 70)
    print("AI Operations Assistant — MVP End-to-End Demo")
    print("=" * 70)
    print(f"Target incident: {DEMO_INCIDENT_NUMBER}")
    print("(The human-reply step is simulated via terminal input for now —")
    print(" everything else in this run is the real pipeline.)\n")

    agent        = SupervisorAgent()
    snow_client  = ServiceNowClient(
        instance_url=SERVICENOW_INSTANCE_URL,
        username=SERVICENOW_USERNAME,
        password=SERVICENOW_PASSWORD,
    )
    slack_client = SlackClient(webhook_url=SLACK_WEBHOOK_URL)

    # ── Step 1: Fetch the incident from ServiceNow ───────────────────
    print(f"[1/6] Fetching incident {DEMO_INCIDENT_NUMBER} from ServiceNow...")
    incident = snow_client.get_incident_by_number(DEMO_INCIDENT_NUMBER)

    if not incident:
        print(f"  [ERROR] No incident found with number {DEMO_INCIDENT_NUMBER}")
        print("  Set SERVICENOW_TEST_INCIDENT in .env to a real incident "
              "number in your ServiceNow instance, then try again.")
        return

    sys_id = incident["sys_id"]

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

    # ── Step 2: Run the real investigation pipeline ───────────────────
    print("[2/6] Running investigation pipeline...")
    kb_result = agent.select_knowledge_base(user_query)

    if kb_result["selected_count"] == 0:
        print(f"  [INFO] No matching knowledge base found. "
              f"Reasoning: {kb_result['overall_reasoning']}")
        return

    print(f"  Selected knowledge base files: "
          f"{[s['filename'] for s in kb_result['selected_yamls']]}")

    analysis = agent.investigate_and_analyse(
        user_query=user_query,
        selected_yamls=kb_result["selected_yamls"],
        tools=TOOLS,
    )

    # ── Step 3: Present to human — post the real Decision Card ───────
    print("\n[3/6] Posting Decision Card to Slack...")
    card = agent.present_to_human(user_query=user_query, analysis=analysis)
    print(card)

    slack_client.send_message(
        f"🔔 *MVP Demo — incident {DEMO_INCIDENT_NUMBER}*\n```\n{card}\n```"
    )
    print("  ✅ Posted to Slack")

    # ── Step 4: Loop — simulate human reply, classify, route ─────────
    # This loop mirrors the Human as Companion decision flow:
    # approve finalizes immediately, request_more_data loops back
    # (unlimited rounds — open-ended collaboration), correct_rca_fix
    # applies the override and finalizes, unclear asks for clarification.
    reinvestigation_rounds = 0

    while True:
        print(f"\n[4/6] Waiting for human reply...")
        human_message = _ask_human_for_reply(reinvestigation_rounds + 1)

        if not human_message:
            print("  [INFO] Empty reply — skipping classification, asking again.")
            continue

        print("\n[5/6] Classifying human feedback...")
        feedback = agent.classify_human_feedback(
            human_message=human_message,
            original_analysis=analysis,
        )
        print(f"  Intent: {feedback['intent']} "
              f"(confidence: {feedback['confidence']})")
        print(f"  Reasoning: {feedback['reasoning']}")

        # ── Path A: Approve ──────────────────────────────────────────
        if feedback["intent"] == "approve":
            print("\n[6/6] Finalizing to ServiceNow (approved)...")
            agent.finalize_to_servicenow(
                servicenow_client=snow_client,
                sys_id=sys_id,
                final_analysis=analysis,
                decision_type="approved",
                human_agent_id="demo-user",
            )
            print("\n✅ Demo complete — incident finalized as APPROVED.")
            return

        # ── Path B: Request more data ──────────────────────────────
        elif feedback["intent"] == "request_more_data":
            reinvestigation_rounds += 1

            # No escalation cap — keep collaborating until the human
            # approves or provides a direct correction.
            pass

            print(f"\n  Re-investigating (round {reinvestigation_rounds})...")
            analysis = agent.re_investigate_with_feedback(
                original_analysis=analysis,
                data_request=feedback["extracted_data_request"],
                selected_yamls=kb_result["selected_yamls"],
                tools=TOOLS,
            )

            card = agent.present_to_human(user_query=user_query, analysis=analysis)
            print(card)
            slack_client.send_message(
                f"🔁 *Updated analysis for {DEMO_INCIDENT_NUMBER}*\n```\n{card}\n```"
            )
            print("  ✅ Updated Decision Card posted to Slack")
            # Loop back to ask for another reply against the updated analysis

        # ── Path C: Correct RCA / fix ────────────────────────────────
        elif feedback["intent"] == "correct_rca_fix":
            print("\n  Applying human override (no re-investigation)...")
            analysis = agent.apply_human_override(
                original_analysis=analysis,
                corrected_root_cause=feedback["extracted_root_cause"],
                corrected_fix=feedback["extracted_fix"],
            )

            print("\n[6/6] Finalizing to ServiceNow (human override)...")
            agent.finalize_to_servicenow(
                servicenow_client=snow_client,
                sys_id=sys_id,
                final_analysis=analysis,
                decision_type="human_override",
                human_agent_id="demo-user",
            )
            print("\n✅ Demo complete — incident finalized with HUMAN OVERRIDE.")
            return

        # ── Path D: Unclear ───────────────────────────────────────────
        else:  # feedback["intent"] == "unclear"
            print("\n  [INFO] Could not confidently classify that reply.")
            clarification = (
                "🤔 I'm not sure I understood that. Could you clarify — "
                "are you approving this analysis, asking for more "
                "information, or correcting the root cause / fix?"
            )
            slack_client.send_message(clarification)
            print(f"  Sent clarification request to Slack: \"{clarification}\"")
            # Loop back and ask for another reply — no other action taken,
            # exactly as agreed: the agent never guesses on an unclear reply.


if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n\nDemo stopped by user. Goodbye.")
