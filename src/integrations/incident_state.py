"""
incident_state.py
==================
LangGraph state design for Phase 2.

THIS FILE IS NOT USED YET. It is a design artifact — the agreed
IncidentState schema that Phase 2 will implement when migrating
from the current in-memory IncidentStore to LangGraph + Postgres.

WHY THIS EXISTS NOW:
    Designing the state schema before implementation prevents the
    most common LangGraph migration mistake: discovering mid-migration
    that you need a field that wasn't anticipated, which forces
    retrofitting the checkpointer schema and re-testing everything.
    Having this agreed upfront means Phase 2 is a clean implementation
    of a known design, not a design-and-implement combined effort.

HOW THIS REPLACES THE CURRENT ARCHITECTURE:
    Current MVP (in-memory):
        IncidentStore._store[channel_id] = {
            "incident_number", "sys_id", "user_query",
            "analysis", "selected_yamls",
            "reinvestigation_rounds", "status"
        }

    Phase 2 (LangGraph + Postgres):
        IncidentState holds ALL of the above plus:
        - full message history (messages) for true multi-turn LLM context
        - lifecycle fields (servicenow_state, decision_type)
        - audit/metrics fields (picked_up_at, finalized_at,
          human_agent_id, first_time_right)
        - human collaboration fields (last_human_message, last_intent,
          last_feedback)

    The migration is ADDITIVE — every field currently in IncidentStore
    maps directly to a field here. No existing data or logic is lost;
    the new fields are additions that enable Phase 2 features.

HOW LANGGRAPH USES THIS:
    In LangGraph, this TypedDict becomes the graph's state schema:

        from langgraph.graph import StateGraph
        graph = StateGraph(IncidentState)

    Each node (investigate, present, classify, finalize) receives the
    full state dict and returns a partial dict of fields it changed.
    LangGraph merges the partial update back into the full state.

    The checkpointer persists the full state to Postgres after every
    node execution — this is what enables multi-day conversations.
    If the server restarts mid-investigation, LangGraph simply reloads
    the last checkpoint and continues from where it left off.

LANGGRAPH NODE MAPPING:
    Current method              → Future LangGraph node
    ─────────────────────────────────────────────────────
    select_knowledge_base()     → SelectKBNode
    investigate_and_analyse()   → InvestigateNode
    present_to_human()          → PresentNode
    classify_human_feedback()   → ClassifyNode
    re_investigate_with_feedback() → ReInvestigateNode
    apply_human_override()      → OverrideNode
    finalize_to_servicenow()    → FinalizeNode

    The conditional routing in slack_listener.py becomes LangGraph's
    conditional edges — after ClassifyNode, the graph branches to
    FinalizeNode (approve), ReInvestigateNode (request_more_data),
    OverrideNode (correct_rca_fix), or back to PresentNode (unclear).

    Human-in-the-loop pause point:
    LangGraph's interrupt() mechanism replaces the current
    slack_listener.py approach entirely — the graph pauses at
    PresentNode, waits for the human's Slack reply, then resumes
    at ClassifyNode with the human's message injected into state.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, TypedDict

from langchain_core.messages import BaseMessage


class IncidentState(TypedDict):
    """
    Complete state for one incident's full lifecycle — from first
    detection by the poller through to ServiceNow finalization.

    This is the single source of truth that all LangGraph nodes read
    from and write to. Every field that is currently passed as a
    parameter between methods, or stored in IncidentStore, lives here.

    LangGraph persists this entire dict to Postgres via its checkpointer
    after every node execution, enabling:
        - Multi-day conversations (survives server restarts)
        - Full audit trail of every state transition
        - Easy debugging (inspect any checkpoint to see exact state)
        - Future evaluation queries (first_time_right, rounds, timing)
    """

    # ── Identity ─────────────────────────────────────────────────────
    # Immutable once set — written at pickup, never changed.
    # These three fields together uniquely identify the incident and
    # its dedicated Slack channel for the entire conversation.

    incident_number: str
    """
    Human-readable ServiceNow incident number.
    Example: "INC0010006"
    Used in Slack messages and log output to identify the incident.
    """

    sys_id: str
    """
    ServiceNow's internal unique ID for the incident record.
    Example: "5e93516393e10f5443643aaefaba10f6"
    Required by update_work_notes() and update_state() — the
    human-readable incident number alone is not enough for PATCH calls.
    """

    slack_channel_id: str
    """
    The Slack channel created specifically for this incident.
    Example: "C0BCKL11QQK"
    All Decision Card posts and agent replies go to this channel.
    The listener uses this to look up which incident a human reply
    belongs to — this is the join key between Slack and ServiceNow.
    """

    # ── Original complaint ────────────────────────────────────────────
    # Set once at pickup. Carried into every re-investigation call
    # so the LLM never loses track of the original order number,
    # customer name, or other specifics from the first message.
    # This was the root cause of the "wrong order number" bug
    # (12345 instead of ORD-84721) that was patched in the MVP —
    # in LangGraph, it simply cannot happen because user_query is
    # always in state.

    user_query: str
    """
    Full complaint text extracted from the ServiceNow incident.
    Combines short_description and description fields.
    Example: "My order ORD-84721 was supposed to arrive 3 days ago
              and still hasn't been dispatched"
    Passed to every investigation call — never lost between rounds.
    """

    selected_yamls: list[dict]
    """
    YAML knowledge files chosen by select_knowledge_base().
    Each item: {"filename": str, "reason": str}
    Reused on every re-investigation call — no need to re-select
    the knowledge base on round 2, 3, etc. since the incident type
    has not changed, only the depth of investigation.
    """

    # ── Investigation state ───────────────────────────────────────────
    # Updated after each investigation or re-investigation round.
    # The messages field is the key addition over the current MVP —
    # it stores the full LangChain message history, enabling true
    # multi-turn LLM context rather than reconstructing context from
    # scratch on every re-investigation.

    analysis: dict
    """
    The current (most recent) analysis dict.
    Shape: {
        "operational_issue": str,
        "findings":          list[str],
        "reasoning_steps":   list[str],
        "root_cause":        str,
        "short_term_fix":    str,
        "human_override":    bool  (only present if apply_human_override() was called)
    }
    Updated by InvestigateNode and ReInvestigateNode.
    Read by PresentNode to format the Decision Card.
    """

    messages: list[BaseMessage]
    """
    Full LangChain message history for this incident's investigation.
    Includes: SystemMessages (prompts), HumanMessages (queries),
    AIMessages (LLM responses), ToolMessages (tool call results).

    This is the most significant addition over the current MVP.
    Currently, re_investigate_with_feedback() reconstructs context
    from scratch by re-passing the original analysis as JSON.
    With LangGraph, the full message history is simply continued —
    the LLM sees the entire investigation conversation, not just
    a snapshot, which means genuinely better multi-round reasoning.

    In LangGraph, use operator.add as the reducer for this field:
        messages: Annotated[list[BaseMessage], operator.add]
    This tells LangGraph to APPEND new messages rather than replace
    the whole list on each node update.
    """

    reinvestigation_rounds: int
    """
    Counter of how many re-investigation rounds have occurred.
    No cap — the Human as Companion model supports open-ended
    collaboration for as long as the human needs.
    Used by the future IncidentAnalysisEvaluator to measure
    collaboration depth (how many rounds before approval).
    """

    human_override: bool
    """
    True if apply_human_override() was called during this incident.
    Set by OverrideNode, never unset.
    The future evaluation agent uses this to identify incidents
    where the AI's initial conclusion needed a human correction —
    exactly the signal needed for YAML improvement decisions.
    """

    # ── Human collaboration ───────────────────────────────────────────
    # Written by ClassifyNode after each human message.
    # Read by the conditional edge router to decide which node to
    # go to next. Stored in state (not just passed as parameters)
    # so every node has full visibility into the human's last intent.

    last_human_message: str
    """
    The exact free text the human most recently typed in Slack.
    Stored here so it is available to all subsequent nodes without
    needing to be re-passed as a parameter.
    """

    last_intent: Literal[
        "approve", "request_more_data", "correct_rca_fix", "unclear", ""
    ]
    """
    The intent classified from last_human_message.
    Empty string "" means the human has not yet replied
    (initial state, before the first Decision Card is reviewed).
    LangGraph's conditional edge reads this to route the graph:
        "approve"           → FinalizeNode
        "request_more_data" → ReInvestigateNode
        "correct_rca_fix"   → OverrideNode → FinalizeNode
        "unclear"           → PresentNode (re-ask for clarification)
        ""                  → interrupt() — wait for human input
    """

    last_feedback: dict | None
    """
    Full result dict from classify_human_feedback().
    Shape: {
        "intent":                   str,
        "confidence":               str,
        "extracted_data_request":   str | None,
        "extracted_root_cause":     str | None,
        "extracted_fix":            str | None,
        "reasoning":                str
    }
    Stored so downstream nodes (ReInvestigateNode, OverrideNode) can
    access the extracted fields (data_request, root_cause, fix) directly
    from state rather than needing to re-parse the human's message.
    """

    # ── Lifecycle ─────────────────────────────────────────────────────
    # The definitive record of where this incident is in its journey.
    # Replaces IncidentStore.status and the scattered ServiceNow state
    # update calls that currently happen in multiple places.

    status: Literal[
        "new",           # poller picked it up, investigation not yet started
        "in_progress",   # investigation running or Decision Card posted
        "awaiting_human", # Decision Card posted, waiting for human reply
        "finalized",     # finalize_to_servicenow() completed successfully
        "escalated",     # reserved for future use
    ]
    """
    High-level lifecycle status of the incident within the AI system.
    LangGraph's conditional edge also reads this — if status is
    "finalized", any further messages in the channel are ignored.
    Replaces IncidentStore.is_active() and mark_finalized().
    """

    servicenow_state: str
    """
    The current ServiceNow state code for this incident.
    Mirrors what was last written to ServiceNow via update_state().
    Stored here so the system always knows the ServiceNow state
    without making an extra API call to check.
    Values: "1" (New), "2" (In Progress), "6" (Resolved)
    Using ServiceNowClient.STATE_* constants is recommended.
    """

    decision_type: str | None
    """
    How the final analysis was reached.
    "approved"       — human accepted the AI's analysis as-is
    "human_override" — human corrected the root cause or fix
    None             — incident not yet finalized
    Written at finalization, stored for audit trail and metrics.
    """

    # ── Audit and metrics ─────────────────────────────────────────────
    # Written throughout the lifecycle. These fields are not used by
    # any current agent logic — they exist specifically for the future
    # IncidentAnalysisEvaluator (Phase 3) that will compute metrics
    # like first-time-right rate, mean time to resolution, and
    # collaboration depth per agent and per incident type.

    picked_up_at: datetime
    """
    When the poller first claimed this incident.
    Set once, never changed. Combined with finalized_at to compute
    mean time to resolution (MTTR) — a key ITSM metric.
    """

    finalized_at: datetime | None
    """
    When finalize_to_servicenow() completed successfully.
    None until the incident is finalized.
    """

    human_agent_id: str | None
    """
    Slack user ID of whoever made the final decision (approved or
    corrected). Written at finalization from the Slack event's
    user field. Used for per-agent accuracy metrics in Phase 3.
    None until the incident is finalized.
    """

    first_time_right: bool | None
    """
    True  — human approved with 0 re-investigation rounds
            (the AI got it right on the first attempt)
    False — at least one re-investigation round was needed, OR
            the human had to override the root cause / fix
    None  — incident not yet finalized

    This is the primary accuracy metric for the AI system.
    A rising first-time-right rate indicates the YAML knowledge base
    is improving. A falling rate signals YAML maintenance is needed.
    Set automatically at finalization:
        first_time_right = (reinvestigation_rounds == 0
                            and not human_override)
    """


# ════════════════════════════════════════════════════════════════════
# Factory function — creates the initial state when an incident
# is first picked up by the poller. All optional/nullable fields
# start at their empty/None values.
# ════════════════════════════════════════════════════════════════════

def create_initial_state(
    incident_number: str,
    sys_id:          str,
    slack_channel_id: str,
    user_query:      str,
    selected_yamls:  list[dict],
    picked_up_at:    datetime | None = None,
) -> IncidentState:
    """
    Create the initial IncidentState when the poller first picks up
    an incident. All fields that are not yet known (analysis, messages,
    human collaboration fields, finalization fields) start at safe
    empty/None defaults.

    In Phase 2, this dict is handed to LangGraph's graph.invoke()
    as the starting state. The graph's checkpointer immediately
    persists it to Postgres, so even if the server crashes before
    the first LLM call completes, the incident is not lost.

    Args:
        incident_number:  ServiceNow incident number (e.g. "INC0010006")
        sys_id:           ServiceNow internal sys_id for API calls
        slack_channel_id: ID of the dedicated Slack channel created
                          for this incident
        user_query:       Combined complaint text from the incident
        selected_yamls:   Output of select_knowledge_base()
        picked_up_at:     When the poller picked this up (defaults to now)

    Returns:
        A fully populated IncidentState dict with safe defaults for
        all fields not yet known at pickup time.
    """
    return IncidentState(
        # Identity
        incident_number=incident_number,
        sys_id=sys_id,
        slack_channel_id=slack_channel_id,

        # Original complaint
        user_query=user_query,
        selected_yamls=selected_yamls,

        # Investigation — empty until InvestigateNode runs
        analysis={},
        messages=[],
        reinvestigation_rounds=0,
        human_override=False,

        # Human collaboration — empty until human replies
        last_human_message="",
        last_intent="",
        last_feedback=None,

        # Lifecycle
        status="new",
        servicenow_state="2",  # already set to In Progress at pickup
        decision_type=None,

        # Audit and metrics
        picked_up_at=picked_up_at or datetime.utcnow(),
        finalized_at=None,
        human_agent_id=None,
        first_time_right=None,
    )
