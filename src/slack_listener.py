"""
slack_listener.py
====================
FastAPI web server that receives events FROM Slack and routes human
replies through the full SupervisorAgent decision pipeline.

This is the INBOUND side of the MVP loop. It replaces the terminal
input() simulation from run_mvp_demo.py with a real Slack Events
webhook receiver.

HOW IT CONNECTS TO THE POLLER:
    This file imports and starts incident_poller.run_poll_loop() in a
    background thread when the FastAPI app starts up. Both the poller
    and the listener share a single IncidentStore instance — the poller
    writes to it when a new incident is processed, and the listener
    reads from it when a human reply arrives. This single shared store
    is what connects "which channel_id belongs to which incident."

HOW TO RUN:
    uvicorn slack_listener:app --reload --port 8000

    This starts both the FastAPI listener AND the background poller
    in one process. You do NOT need to run incident_poller.py separately.

SLACK SIGNATURE VERIFICATION:
    Production Slack apps should verify incoming requests using HMAC-
    SHA256 with the Signing Secret from your Slack App settings. This
    version omits that step for MVP simplicity — it is the clear next
    security step before any real production deployment.
"""

import os
import sys
import threading

import dotenv
from fastapi import FastAPI, Request

sys.path.insert(0, os.path.dirname(__file__))

dotenv.load_dotenv()

from agents.supervisor_agent import SupervisorAgent
from integrations.servicenow_client import ServiceNowClient
from integrations.slack_client import SlackClient
from integrations.incident_store import IncidentStore
from integrations.incident_poller import run_poll_loop
from tools.inventory_tools import (
    get_warehouse_availability,
    get_allocation_queue_details,
)
from tools.enterprise_tools import (
    get_shipment_status,
    get_proof_of_delivery,
    get_rma_status,
    get_credit_note_status,
    get_account_credit_status,
    get_payment_allocation_status,
    get_stock_levels,
    get_replenishment_orders,
    get_edi_transaction_status,
    get_partner_gateway_status,
    get_contract_price,
    get_applied_price_on_order,
    get_customs_clearance_status,
    get_trade_document_status,
    get_invoice_status,
    get_three_way_match_result,
    get_warranty_record,
    get_claim_status,
)

# ════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════

try:
    SERVICENOW_INSTANCE_URL = os.environ["SERVICENOW_INSTANCE_URL"]
    SERVICENOW_USERNAME     = os.environ["SERVICENOW_USERNAME"]
    SERVICENOW_PASSWORD     = os.environ["SERVICENOW_PASSWORD"]
    SLACK_BOT_TOKEN         = os.environ["SLACK_BOT_TOKEN"]
except KeyError as e:
    print(f"[ERROR] Missing environment variable: {e}")
    raise SystemExit(1)


TOOLS = [
    # Inventory and order fulfilment
    get_warehouse_availability,
    get_allocation_queue_details,
    get_stock_levels,
    get_replenishment_orders,
    # Shipment and delivery
    get_shipment_status,
    get_proof_of_delivery,
    # Returns and RMA
    get_rma_status,
    get_credit_note_status,
    # Finance and credit
    get_account_credit_status,
    get_payment_allocation_status,
    # Supplier and EDI compliance
    get_edi_transaction_status,
    get_partner_gateway_status,
    # Pricing
    get_contract_price,
    get_applied_price_on_order,
    # Customs and trade
    get_customs_clearance_status,
    get_trade_document_status,
    # Invoice
    get_invoice_status,
    get_three_way_match_result,
    # Warranty
    get_warranty_record,
    get_claim_status,
]

# ════════════════════════════════════════════════════════════════════
# SHARED INSTANCES
# One of each — shared between the poller thread and the listener.
# ════════════════════════════════════════════════════════════════════

store        = IncidentStore()
agent        = SupervisorAgent()
snow_client  = ServiceNowClient(
    instance_url=SERVICENOW_INSTANCE_URL,
    username=SERVICENOW_USERNAME,
    password=SERVICENOW_PASSWORD,
)
slack_client = SlackClient(bot_token=SLACK_BOT_TOKEN)

# Deduplication set — tracks Slack event_ids we have already processed.
# Slack retries event delivery if it does not receive a fast enough response
# (our LLM calls can take several seconds). Without this, the same human
# message triggers multiple re-investigation rounds.
# This is an in-memory set — it resets on server restart, which is fine
# since Slack only retries within a short window (a few minutes).
seen_event_ids: set = set()

# ════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ════════════════════════════════════════════════════════════════════

app = FastAPI()


@app.on_event("startup")
def start_poller():
    """
    Start the ServiceNow polling loop in a background thread when
    the FastAPI server starts. Running uvicorn slack_listener:app
    starts BOTH the listener AND the poller in one command.
    daemon=True means the thread stops automatically when the
    main process stops.
    """
    print("\n[Startup] Starting ServiceNow poller in background thread...")
    poller_thread = threading.Thread(
        target=run_poll_loop,
        args=(store,),
        daemon=True,
        name="ServiceNow-Poller",
    )
    poller_thread.start()
    print("[Startup] Poller thread started. Listener ready.\n")


@app.get("/")
def health_check():
    """Health check endpoint."""
    return {"status": "AI Operations Assistant Slack listener is running"}


@app.post("/slack/events")
async def handle_slack_event(request: Request):
    """
    Receive all Slack events — handles URL verification and message events.
    Routes human replies through classify_human_feedback() and to the
    correct handler (approve / re-investigate / override / unclear).
    """
    payload = await request.json()

    # ── URL verification handshake ────────────────────────────────
    if payload.get("type") == "url_verification":
        challenge = payload.get("challenge")
        print(f"\n[VERIFICATION] Responding with challenge: {challenge}\n")
        return {"challenge": challenge}

    # ── Extract the inner event ───────────────────────────────────
    event = payload.get("event", {})

    if event.get("type") != "message":
        return {"status": "ignored - not a message event"}

    channel_id   = event.get("channel")
    message_text = event.get("text", "").strip()
    bot_id       = event.get("bot_id")
    subtype      = event.get("subtype")

    # Deduplicate — skip events we have already processed.
    # Slack includes a unique event_id at the top level of the payload.
    event_id = payload.get("event_id")
    if event_id:
        if event_id in seen_event_ids:
            print(f"  [INFO] Duplicate event {event_id} — ignoring")
            return {"status": "ignored - duplicate event"}
        seen_event_ids.add(event_id)

    # Skip bot messages to prevent infinite loops — our own Decision
    # Card posts come back as message events; we must not classify them.
    if bot_id or subtype == "bot_message":
        return {"status": "ignored - bot message"}

    # Skip Slack system messages — these are fired automatically when
    # users join or leave a channel and are not human replies.
    # Without this filter, every auto-invite triggers a classification
    # attempt which always returns "unclear" and sends a confusing
    # clarification message back into the channel.
    if subtype in ("channel_join", "channel_leave", "channel_archive",
                   "channel_unarchive", "bot_add", "bot_remove"):
        return {"status": "ignored - system message"}

    print(f"\n[Message received] channel={channel_id} text='{message_text}'")

    # ── Look up the incident this channel belongs to ──────────────
    incident_context = store.get_incident(channel_id)

    if not incident_context:
        print(f"  [INFO] No active incident for channel {channel_id} — ignoring")
        # Stay silent for unknown channels — these are likely other
        # workspace channels unrelated to this system entirely.
        # We only respond in channels we created for incidents.
        return {"status": "ignored - unknown channel"}

    if not store.is_active(channel_id):
        status = incident_context["status"]
        print(f"  [INFO] Incident already {status} — responding with status message")
        # Don't stay silent — tell the human the incident is closed
        # so they know why no analysis is happening.
        slack_client.post_to_channel(
            channel_id,
            f"ℹ️ This incident has already been *{status}*. "
            f"No further analysis will be performed in this channel.\n"
            f"If you need to raise a new investigation, please create "
            f"a new incident in ServiceNow assigned to the AI Ops Agent group."
        )
        return {"status": "ignored - incident already closed"}

    incident_number = incident_context["incident_number"]
    sys_id          = incident_context["sys_id"]
    user_query      = incident_context["user_query"]
    analysis        = incident_context["analysis"]
    selected_yamls  = incident_context["selected_yamls"]

    print(f"  Processing reply for incident {incident_number}")

    # ── Classify the human's reply ────────────────────────────────
    feedback = agent.classify_human_feedback(
        human_message=message_text,
        original_analysis=analysis,
    )
    intent     = feedback["intent"]
    confidence = feedback["confidence"]

    print(f"  Intent: {intent} (confidence: {confidence})")
    print(f"  Reasoning: {feedback['reasoning']}")

    # ════════════════════════════════════════════════════════════
    # ROUTING — four paths matching the agreed MVP design
    # ════════════════════════════════════════════════════════════

    # ── Path A: Approve ───────────────────────────────────────────
    if intent == "approve":
        try:
            agent.finalize_to_servicenow(
                servicenow_client=snow_client,
                sys_id=sys_id,
                final_analysis=analysis,
                decision_type="approved",
                human_agent_id=event.get("user", "unknown"),
            )
            store.mark_finalized(channel_id)
            slack_client.post_to_channel(
                channel_id,
                f"✅ *Incident {incident_number} finalized.*\n"
                f"Analysis approved and written to ServiceNow work notes. "
                f"This channel can now be archived."
            )
            print(f"  Incident {incident_number} approved and finalized")
        except Exception as e:
            print(f"  [ERROR] Finalization failed: {e}")
            slack_client.post_to_channel(
                channel_id,
                f"⚠️ Approval received but could not write to ServiceNow: {e}\n"
                f"Please update work notes manually."
            )

    # ── Path B: Request more data ─────────────────────────────────
    elif intent == "request_more_data":
        # No cap on re-investigation rounds — the "Human as Companion"
        # model explicitly supports open-ended, multi-round collaboration.
        # The human and AI investigate together for as long as needed,
        # potentially across days (once persistent store is in place).
        # An escalation cap would contradict the core thesis of this system.
        rounds = store.increment_reinvestigation_rounds(channel_id)
        print(f"  Re-investigation round {rounds}")

        try:
            slack_client.post_to_channel(channel_id, "🔄 Re-investigating...")
            new_analysis = agent.re_investigate_with_feedback(
                original_analysis=analysis,
                data_request=feedback.get("extracted_data_request", message_text),
                selected_yamls=selected_yamls,
                tools=TOOLS,
                user_query=user_query,
            )
            store.update_analysis(channel_id, new_analysis)
            new_card = agent.present_to_human(
                user_query=user_query,
                analysis=new_analysis,
            )
            slack_client.post_to_channel(
                channel_id,
                f"🔁 *Updated analysis (round {rounds}):*\n```\n{new_card}\n```"
            )
        except Exception as e:
            print(f"  [ERROR] Re-investigation failed: {e}")
            slack_client.post_to_channel(channel_id, f"⚠️ Re-investigation failed: {e}")

    # ── Path C: Correct RCA / fix ─────────────────────────────────
    elif intent == "correct_rca_fix":
        try:
            corrected_analysis = agent.apply_human_override(
                original_analysis=analysis,
                corrected_root_cause=feedback.get("extracted_root_cause"),
                corrected_fix=feedback.get("extracted_fix"),
            )
            store.update_analysis(channel_id, corrected_analysis)
            agent.finalize_to_servicenow(
                servicenow_client=snow_client,
                sys_id=sys_id,
                final_analysis=corrected_analysis,
                decision_type="human_override",
                human_agent_id=event.get("user", "unknown"),
            )
            store.mark_finalized(channel_id)
            slack_client.post_to_channel(
                channel_id,
                f"✅ *Incident {incident_number} finalized with your correction.*\n"
                f"Root cause and fix updated. Written to ServiceNow work notes."
            )
        except Exception as e:
            print(f"  [ERROR] Override failed: {e}")
            slack_client.post_to_channel(channel_id, f"⚠️ Override failed: {e}")

    # ── Path D: Unclear ───────────────────────────────────────────
    else:
        slack_client.post_to_channel(
            channel_id,
            "🤔 I received your message but wasn't sure how to interpret it "
            "in the context of this incident analysis.\n\n"
            "Feel free to respond naturally — share your thoughts on the "
            "analysis, ask for any additional information you need, or let "
            "me know if you think something in the findings is incorrect.\n\n"
            "If your message was unrelated to this incident, please ignore this reply."
        )

    return {"status": "processed"}
