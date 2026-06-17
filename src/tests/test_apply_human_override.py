"""
tests/test_apply_human_override.py
====================================
Unit tests for SupervisorAgent.apply_human_override()

Unlike most of the other test files in this project, these tests do
NOT need an OpenAI API key and are NOT skipped when one is missing.
This is because apply_human_override() makes no LLM calls at all —
it is pure Python dict manipulation. This makes these tests extremely
fast and free to run, and they are a good example of why we deliberately
designed this method to avoid an LLM call in the first place.

Run from project root:
    python -m tests.test_apply_human_override -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.supervisor_agent import SupervisorAgent


# A fixed, fake "original_analysis" dict — shaped exactly like what
# investigate_and_analyse() would return. We reuse this same dict
# across multiple tests so we are always overriding from the same
# known starting point.
ORIGINAL_ANALYSIS = {
    "operational_issue": "Order stuck in allocation queue due to warehouse allocation backlog.",
    "findings": [
        "Warehouse stock available: 18 units.",
        "Pending allocation requests: 34 for the order.",
    ],
    "reasoning_steps": [
        "Checked warehouse stock — 18 units available.",
        "Checked allocation queue — order at position 14.",
    ],
    "root_cause": "Warehouse allocation backlog preventing order from being processed.",
    "short_term_fix": "Prioritise the order in the allocation queue.",
}


class TestApplyHumanOverride(unittest.TestCase):
    """
    Unit tests for apply_human_override().

    No OpenAI API key is required — this method makes no LLM calls.
    We create a fresh agent in setUp() purely so the test has access
    to the method; the agent's LLM attributes are never actually used
    by apply_human_override().
    """

    def setUp(self):
        """Called automatically before every test method."""
        self.agent = SupervisorAgent()

    def test_both_root_cause_and_fix_corrected(self):
        """
        When the human corrects BOTH root cause and fix, both fields
        in the returned dict must reflect the human's correction exactly.
        """
        result = self.agent.apply_human_override(
            original_analysis=ORIGINAL_ANALYSIS,
            corrected_root_cause="Payment hold from finance, not a warehouse backlog.",
            corrected_fix="Release the payment hold.",
        )
        self.assertEqual(
            result["root_cause"],
            "Payment hold from finance, not a warehouse backlog."
        )
        self.assertEqual(result["short_term_fix"], "Release the payment hold.")

    def test_only_root_cause_corrected_fix_untouched(self):
        """
        When the human only corrects the root cause and leaves the fix
        as None, the original fix must be preserved exactly as it was —
        not erased, not replaced with None, not replaced with an empty string.
        """
        result = self.agent.apply_human_override(
            original_analysis=ORIGINAL_ANALYSIS,
            corrected_root_cause="Payment hold from finance.",
            corrected_fix=None,
        )
        self.assertEqual(result["root_cause"], "Payment hold from finance.")
        # The fix should be untouched — exactly the original value
        self.assertEqual(
            result["short_term_fix"],
            ORIGINAL_ANALYSIS["short_term_fix"]
        )

    def test_only_fix_corrected_root_cause_untouched(self):
        """
        The mirror image of the previous test — when only the fix is
        corrected, the original root_cause must be preserved unchanged.
        """
        result = self.agent.apply_human_override(
            original_analysis=ORIGINAL_ANALYSIS,
            corrected_root_cause=None,
            corrected_fix="Escalate to the finance team directly.",
        )
        self.assertEqual(
            result["root_cause"],
            ORIGINAL_ANALYSIS["root_cause"]
        )
        self.assertEqual(
            result["short_term_fix"],
            "Escalate to the finance team directly."
        )

    def test_findings_and_operational_issue_always_unchanged(self):
        """
        apply_human_override() must NEVER touch "findings" or
        "operational_issue" — the human has confirmed the data is
        correct, only the conclusion drawn from it was wrong. If these
        fields changed, it would mean the method is doing more than its
        documented job.
        """
        result = self.agent.apply_human_override(
            original_analysis=ORIGINAL_ANALYSIS,
            corrected_root_cause="Something else entirely.",
            corrected_fix="A different fix.",
        )
        self.assertEqual(result["findings"], ORIGINAL_ANALYSIS["findings"])
        self.assertEqual(
            result["operational_issue"],
            ORIGINAL_ANALYSIS["operational_issue"]
        )

    def test_human_override_flag_is_set(self):
        """
        The returned dict must include human_override=True. This flag
        is what a future evaluation/feedback-loop agent will rely on to
        identify which incidents needed a human correction — without
        it, that future code would have no reliable way to tell an
        AI-only resolution apart from a human-corrected one just by
        reading the final analysis dict.
        """
        result = self.agent.apply_human_override(
            original_analysis=ORIGINAL_ANALYSIS,
            corrected_root_cause="New root cause.",
            corrected_fix=None,
        )
        self.assertIn("human_override", result)
        self.assertTrue(result["human_override"])

    def test_original_analysis_dict_is_not_mutated(self):
        """
        Calling apply_human_override() must NOT modify the
        original_analysis dict that was passed in. If it did, any code
        elsewhere still holding a reference to the original dict would
        see its values silently change — a classic and confusing Python
        bug. We guard against this by checking the original dict's
        root_cause is still exactly what it was before the call.
        """
        # Take a snapshot of the original root_cause before calling the method
        original_root_cause_before_call = ORIGINAL_ANALYSIS["root_cause"]

        self.agent.apply_human_override(
            original_analysis=ORIGINAL_ANALYSIS,
            corrected_root_cause="A completely different root cause.",
            corrected_fix="A completely different fix.",
        )

        # The original dict must be unchanged after the call
        self.assertEqual(
            ORIGINAL_ANALYSIS["root_cause"],
            original_root_cause_before_call
        )

    def test_result_contains_all_required_keys(self):
        """
        The returned dict must contain every key present_to_human()
        expects, plus the human_override flag. Missing a key here would
        cause a silent "N/A" to appear in the Decision Card.
        """
        result = self.agent.apply_human_override(
            original_analysis=ORIGINAL_ANALYSIS,
            corrected_root_cause="X",
            corrected_fix="Y",
        )
        required_keys = [
            "operational_issue", "findings", "reasoning_steps",
            "root_cause", "short_term_fix", "human_override"
        ]
        for key in required_keys:
            self.assertIn(key, result, f"Missing key: {key}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
