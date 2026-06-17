"""
tests/test_finalize_to_servicenow.py
======================================
Unit tests for SupervisorAgent.finalize_to_servicenow()

This method's entire job is to format a work note and call
servicenow_client.update_work_notes(). We do NOT want these tests to
depend on a real ServiceNow instance — that would make the tests slow,
require real credentials, and risk writing test data into a real
ServiceNow incident every time the test suite runs.

Instead, we use a small "fake" ServiceNow client defined right here in
this test file. It has the exact same update_work_notes() method
signature as the real ServiceNowClient, but instead of making a real
HTTP request, it just records what it was called with, in memory. This
lets us verify finalize_to_servicenow() is building the right work
note text and calling the client correctly, without any network call
or API key at all.

This pattern is called a "test double" or "fake" — it stands in for a
real dependency during testing. It is different from a "mock" (which
usually comes from a mocking library like unittest.mock) in that we
are writing this fake by hand as a plain Python class, which keeps the
test easy to read for anyone not yet familiar with mocking libraries.

Run from project root:
    python -m tests.test_finalize_to_servicenow -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.supervisor_agent import SupervisorAgent


class FakeServiceNowClient:
    """
    A fake stand-in for the real ServiceNowClient, used only in tests.

    Instead of sending a real HTTP PATCH request to a ServiceNow
    instance, update_work_notes() here just remembers what it was
    called with — sys_id and note_text — so the test can check
    afterwards that finalize_to_servicenow() called it correctly.
    """

    def __init__(self):
        # These start as None, meaning "update_work_notes has not been
        # called yet". After a call, they hold whatever values were
        # passed in, so the test can inspect them.
        self.last_sys_id    = None
        self.last_note_text = None
        self.call_count     = 0

    def update_work_notes(self, sys_id: str, note_text: str):
        """
        Fake version of the real method. Records its arguments instead
        of making a real API call, then returns True — exactly what the
        real ServiceNowClient.update_work_notes() returns on success.
        """
        self.last_sys_id    = sys_id
        self.last_note_text = note_text
        self.call_count    += 1
        return True


# A fixed, fake analysis dict — same shape as investigate_and_analyse()
# would return. Reused across multiple tests below.
SAMPLE_ANALYSIS = {
    "operational_issue": "Order stuck in allocation queue due to warehouse allocation backlog.",
    "findings": [
        "Warehouse stock available: 18 units.",
        "Pending allocation requests: 34 for the order.",
    ],
    "reasoning_steps": [
        "Checked warehouse stock — 18 units available.",
    ],
    "root_cause": "Warehouse allocation backlog preventing order from being processed.",
    "short_term_fix": "Prioritise the order in the allocation queue.",
}


class TestFinalizeToServiceNow(unittest.TestCase):
    """
    Unit tests for finalize_to_servicenow().

    No OpenAI API key or real ServiceNow connection is required — this
    method makes no LLM calls, and we use FakeServiceNowClient instead
    of a real one.
    """

    def setUp(self):
        """Called automatically before every test method."""
        self.agent = SupervisorAgent()
        # A fresh fake client for every test, so call_count and other
        # recorded values never leak between tests.
        self.fake_snow = FakeServiceNowClient()

    def test_calls_update_work_notes_exactly_once(self):
        """
        finalize_to_servicenow() must call update_work_notes() exactly
        once per call — never zero times (meaning it silently did
        nothing) and never more than once (meaning it is duplicating
        the work note unnecessarily).
        """
        self.agent.finalize_to_servicenow(
            servicenow_client=self.fake_snow,
            sys_id="abc123",
            final_analysis=SAMPLE_ANALYSIS,
            decision_type="approved",
            human_agent_id="jane.doe@example.com",
        )
        self.assertEqual(self.fake_snow.call_count, 1)

    def test_passes_correct_sys_id(self):
        """
        The exact sys_id passed into finalize_to_servicenow() must be
        the exact sys_id passed through to update_work_notes() — this
        is what ensures the work note lands on the correct incident.
        """
        self.agent.finalize_to_servicenow(
            servicenow_client=self.fake_snow,
            sys_id="xyz789",
            final_analysis=SAMPLE_ANALYSIS,
            decision_type="approved",
            human_agent_id="jane.doe@example.com",
        )
        self.assertEqual(self.fake_snow.last_sys_id, "xyz789")

    def test_work_note_contains_decision_type_and_approver(self):
        """
        The work note text must clearly state HOW the decision was
        reached (decision_type) and WHO approved it (human_agent_id).
        This is what makes the work note a genuine audit trail rather
        than just a vague "resolved" status update.
        """
        self.agent.finalize_to_servicenow(
            servicenow_client=self.fake_snow,
            sys_id="abc123",
            final_analysis=SAMPLE_ANALYSIS,
            decision_type="approved",
            human_agent_id="jane.doe@example.com",
        )
        note = self.fake_snow.last_note_text
        self.assertIn("approved", note)
        self.assertIn("jane.doe@example.com", note)

    def test_work_note_contains_root_cause_and_fix(self):
        """
        The work note must include the actual root cause and fix text
        from the analysis — without these, the work note would record
        THAT something was decided but not WHAT was decided, which
        defeats the purpose of an audit trail.
        """
        self.agent.finalize_to_servicenow(
            servicenow_client=self.fake_snow,
            sys_id="abc123",
            final_analysis=SAMPLE_ANALYSIS,
            decision_type="approved",
            human_agent_id="jane.doe@example.com",
        )
        note = self.fake_snow.last_note_text
        self.assertIn(SAMPLE_ANALYSIS["root_cause"], note)
        self.assertIn(SAMPLE_ANALYSIS["short_term_fix"], note)

    def test_human_override_flag_adds_explicit_note(self):
        """
        When final_analysis contains human_override=True (as returned
        by apply_human_override()), the work note must explicitly call
        this out, so a future reader (or a future evaluation agent)
        can easily identify incidents where the AI's conclusion needed
        a human correction.
        """
        overridden_analysis = dict(SAMPLE_ANALYSIS)
        overridden_analysis["human_override"] = True

        self.agent.finalize_to_servicenow(
            servicenow_client=self.fake_snow,
            sys_id="abc123",
            final_analysis=overridden_analysis,
            decision_type="human_override",
            human_agent_id="jane.doe@example.com",
        )
        note = self.fake_snow.last_note_text
        self.assertIn("corrected directly by the human agent", note)

    def test_no_override_note_when_flag_absent(self):
        """
        The opposite of the previous test — when human_override is NOT
        present in final_analysis (the normal case, e.g. a straight
        approval of the AI's first analysis), the work note must NOT
        contain the override callout text. Otherwise every work note
        would misleadingly suggest a human correction happened even
        when the AI got it right on the first attempt.
        """
        self.agent.finalize_to_servicenow(
            servicenow_client=self.fake_snow,
            sys_id="abc123",
            final_analysis=SAMPLE_ANALYSIS,  # no human_override key at all
            decision_type="approved",
            human_agent_id="jane.doe@example.com",
        )
        note = self.fake_snow.last_note_text
        self.assertNotIn("corrected directly by the human agent", note)

    def test_returns_none(self):
        """
        finalize_to_servicenow() has nothing meaningful to return —
        its entire job is the side effect of writing to ServiceNow.
        This test confirms it returns None, as documented, rather than
        accidentally returning some other value a caller might
        mistakenly start relying on.
        """
        result = self.agent.finalize_to_servicenow(
            servicenow_client=self.fake_snow,
            sys_id="abc123",
            final_analysis=SAMPLE_ANALYSIS,
            decision_type="approved",
            human_agent_id="jane.doe@example.com",
        )
        self.assertIsNone(result)

    def test_missing_findings_does_not_crash(self):
        """
        If final_analysis is missing the "findings" key entirely (which
        should not normally happen, but defensive code should handle it
        gracefully rather than crashing), finalize_to_servicenow() must
        still complete without raising an exception.
        """
        incomplete_analysis = {
            "operational_issue": "Some issue",
            "root_cause": "Some cause",
            "short_term_fix": "Some fix",
            # "findings" deliberately omitted
        }
        # This should not raise — if it does, the test fails automatically
        self.agent.finalize_to_servicenow(
            servicenow_client=self.fake_snow,
            sys_id="abc123",
            final_analysis=incomplete_analysis,
            decision_type="approved",
            human_agent_id="jane.doe@example.com",
        )
        self.assertEqual(self.fake_snow.call_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
