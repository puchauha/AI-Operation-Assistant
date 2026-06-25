"""
incident_store.py
==================
A simple in-memory store that maps each Slack channel_id to the full
context of the incident being investigated in that channel.

WHY THIS EXISTS:
    When a human types a reply in a Slack channel, the only information
    Slack sends us is the message text and the channel_id it came from.
    Slack does NOT tell us "this reply is about INC0010002" — we have
    to figure that out ourselves by looking up which incident was
    being discussed in that channel.

    This store solves that problem by maintaining a dict where:
        key   = Slack channel_id (e.g. "C0B69NEWCHAN")
        value = everything we need to continue the investigation
                (incident number, sys_id, current analysis, etc.)

    When the slack_listener receives a message event, it extracts the
    channel_id from the payload, looks it up here, and immediately
    has all the context it needs to route the reply correctly.

SCOPE — MVP DELIBERATELY KEEPS THIS SIMPLE:
    This is an in-memory Python dict. It lives only as long as the
    FastAPI process (slack_listener.py) is running. If the server
    restarts, all stored context is lost.

    This is an accepted MVP trade-off. The natural upgrade path
    (already on the roadmap as item 6 — "persistent store for
    multi-day collaboration") is to replace this with a real
    database (Postgres, Redis, or similar) that survives process
    restarts and supports long-running, multi-day investigations.
    The interface of this class is designed to make that upgrade
    straightforward — callers only use save_incident(),
    get_incident(), and update_analysis(), so swapping the
    in-memory dict for a database call only changes these three
    methods, not any of the code that calls them.
"""


class IncidentStore:
    """
    In-memory mapping of Slack channel_id → incident context.

    Each stored entry has this shape:
    {
        "incident_number":       str,   e.g. "INC0010002"
        "sys_id":                str,   ServiceNow internal ID for finalization
        "user_query":            str,   original complaint text
        "analysis":              dict,  latest analysis dict from the agent
        "selected_yamls":        list,  YAML files selected for this incident
        "reinvestigation_rounds": int,  how many re-investigation rounds done
        "status":                str,   "awaiting_human" | "finalized" | "escalated"
    }
    """

    def __init__(self):
        # The actual storage — a plain Python dict.
        # key: channel_id (str)
        # value: incident context dict (described above)
        self._store = {}

    def save_incident(
        self,
        channel_id:      str,
        incident_number: str,
        sys_id:          str,
        user_query:      str,
        analysis:        dict,
        selected_yamls:  list,
    ) -> None:
        """
        Store a new incident's context, keyed by its Slack channel_id.

        Called by incident_poller.py immediately after the Decision
        Card has been posted to the incident's dedicated channel —
        at that point, we know the channel_id and have all the
        investigation results to store.

        Parameters
        ----------
        channel_id : str
            The Slack channel this incident is being discussed in.
            This is the lookup key for all future operations on
            this incident.

        incident_number : str
            Human-readable ServiceNow incident number, e.g. "INC0010002".
            Stored for display and for finalize_to_servicenow().

        sys_id : str
            ServiceNow's internal unique ID for the incident record.
            Needed by finalize_to_servicenow() to write the work note
            to the correct incident — the human-readable number alone
            is not enough for the ServiceNow PATCH API.

        user_query : str
            The original complaint text extracted from the incident.
            Needed by present_to_human() when generating an updated
            Decision Card after re-investigation.

        analysis : dict
            The current analysis dict, as returned by
            investigate_and_analyse() or re_investigate_with_feedback().
            Updated in place via update_analysis() as the investigation
            progresses through re-investigation rounds.

        selected_yamls : list
            The list of {"filename": ..., "reason": ...} dicts returned
            by select_knowledge_base(). Needed by
            re_investigate_with_feedback() if the human asks for more data.
        """
        self._store[channel_id] = {
            "incident_number":        incident_number,
            "sys_id":                 sys_id,
            "user_query":             user_query,
            "analysis":               analysis,
            "selected_yamls":         selected_yamls,
            "reinvestigation_rounds": 0,
            "status":                 "awaiting_human",
        }
        print(f"  [Store] Saved incident {incident_number} → channel {channel_id}")

    def get_incident(self, channel_id: str) -> dict | None:
        """
        Look up the incident context for a given Slack channel.

        Called by slack_listener.py when a message event arrives —
        the channel_id from the event payload is the lookup key.

        Parameters
        ----------
        channel_id : str
            The Slack channel_id from the incoming message event.

        Returns
        -------
        dict
            The full incident context dict if found.
        None
            If no incident is associated with this channel — which
            is normal and expected, since not every Slack channel
            belongs to an active incident (e.g. the #general channel,
            or a channel from a previous session). The caller should
            handle None gracefully by ignoring the event.
        """
        return self._store.get(channel_id)

    def update_analysis(self, channel_id: str, new_analysis: dict) -> None:
        """
        Replace the stored analysis dict with a new version.

        Called after re_investigate_with_feedback() or
        apply_human_override() produces an updated analysis — we
        need the store to hold the LATEST version so that if the
        human asks for yet another round of investigation, we pass
        in the most recent state, not the original first-pass result.

        Parameters
        ----------
        channel_id : str
            The channel whose stored analysis should be updated.

        new_analysis : dict
            The updated analysis dict to store.
        """
        if channel_id in self._store:
            self._store[channel_id]["analysis"] = new_analysis

    def increment_reinvestigation_rounds(self, channel_id: str) -> int:
        """
        Increment and return the re-investigation round counter.

        Called each time the agent starts a new re-investigation
        for this incident. The return value is the NEW round count
        after incrementing — the caller uses this to check whether
        how many re-investigation rounds have occurred so far.

        Parameters
        ----------
        channel_id : str
            The channel whose counter should be incremented.

        Returns
        -------
        int
            The updated round count after incrementing.
        """
        if channel_id in self._store:
            self._store[channel_id]["reinvestigation_rounds"] += 1
            return self._store[channel_id]["reinvestigation_rounds"]
        return 0

    def mark_finalized(self, channel_id: str) -> None:
        """
        Mark an incident as finalized so duplicate replies are ignored.

        Once finalize_to_servicenow() has been called for an incident,
        any further messages in that channel should be ignored — the
        incident is done. This flag is how the listener knows to skip
        those messages rather than trying to re-classify them.
        """
        if channel_id in self._store:
            self._store[channel_id]["status"] = "finalized"

    def mark_escalated(self, channel_id: str) -> None:
        """
        Mark an incident as escalated — re-investigation cap was hit.
        Same purpose as mark_finalized() but distinguishes the
        escalation case for future metrics/evaluation work.
        """
        if channel_id in self._store:
            self._store[channel_id]["status"] = "escalated"

    def is_active(self, channel_id: str) -> bool:
        """
        Return True if this channel has an incident that is still
        awaiting a human decision — i.e. not yet finalized or escalated.
        Used by the listener to quickly skip channels that are done.
        """
        entry = self._store.get(channel_id)
        if not entry:
            return False
        return entry["status"] == "awaiting_human"
