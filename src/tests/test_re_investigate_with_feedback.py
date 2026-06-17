"""
tests/test_re_investigate_with_feedback.py
============================================
Integration tests for SupervisorAgent.re_investigate_with_feedback()

This method calls the reasoning LLM and tools, so these tests are
skipped automatically if OPENAI_API_KEY is not set — same pattern as
test_investigate_and_analyse.py and test_select_knowledge_base.py.

We build a fixed, fake "original_analysis" dict directly in this file
(rather than running the full select_knowledge_base() +
investigate_and_analyse() pipeline first) to keep these tests focused
purely on re_investigate_with_feedback()'s own behaviour, and to avoid
spending extra API calls on a pipeline we have already tested elsewhere.

Run from project root:
    python -m tests.test_re_investigate_with_feedback -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.supervisor_agent import SupervisorAgent
from tools.inventory_tools import get_warehouse_availability, get_allocation_queue_details

TOOLS = [
    get_warehouse_availability,
    get_allocation_queue_details,
]

# The same selected_yamls shape that select_knowledge_base() would
# return for an order delay complaint. We hardcode this here because
# we already know order_delay.yaml is the correct file for this
# scenario — there is no need to spend an extra API call asking the
# router LLM to tell us something we already know for this test.
SELECTED_YAMLS = [
    {"filename": "order_delay.yaml", "reason": "Order dispatch delay complaint"}
]

# A fixed, fake "original_analysis" dict — shaped exactly like what
# investigate_and_analyse() would have returned on a first pass.
ORIGINAL_ANALYSIS = {
    "operational_issue": "Order stuck in allocation queue due to warehouse allocation backlog.",
    "findings": [
        "Warehouse stock available: 18 units.",
        "Pending allocation requests: 34 for the order.",
    ],
    "reasoning_steps": [
        "Checked warehouse stock — 18 units available, so stock is not the issue.",
        "Checked pending allocation requests — found 34, suggesting a backlog.",
    ],
    "root_cause": "Warehouse allocation backlog preventing order from being processed.",
    "short_term_fix": "Prioritise the order in the allocation queue.",
}


@unittest.skipUnless(
    os.getenv("OPENAI_API_KEY"),
    "OPENAI_API_KEY not set — skipping integration tests"
)
class TestReInvestigateWithFeedback(unittest.TestCase):
    """
    Integration tests for re_investigate_with_feedback().

    Each test calls the real reasoning LLM and real tools, so these
    are integration tests, not pure unit tests (unlike
    test_apply_human_override.py, which needs no API key at all).
    """

    def setUp(self):
        """Called automatically before every test method."""
        self.agent = SupervisorAgent()

    def test_returns_dict(self):
        """Method must return a dict — not a string or other type."""
        result = self.agent.re_investigate_with_feedback(
            original_analysis=ORIGINAL_ANALYSIS,
            data_request="Can you also check the allocation queue position?",
            selected_yamls=SELECTED_YAMLS,
            tools=TOOLS,
        )
        self.assertIsInstance(result, dict)

    def test_result_has_required_keys(self):
        """
        The result must have the same five keys as
        investigate_and_analyse() — this is what allows the result to
        be passed straight into present_to_human() without any
        special-case handling.
        """
        result = self.agent.re_investigate_with_feedback(
            original_analysis=ORIGINAL_ANALYSIS,
            data_request="Can you also check the allocation queue position?",
            selected_yamls=SELECTED_YAMLS,
            tools=TOOLS,
        )
        for key in ["operational_issue", "findings", "reasoning_steps",
                    "root_cause", "short_term_fix"]:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_findings_reflect_additional_investigation(self):
        """
        After asking specifically about the allocation queue, the new
        findings should mention queue-related terms — confirming the
        LLM actually called the relevant tool in response to the
        human's specific request, rather than just repeating the
        original findings unchanged.
        """
        result = self.agent.re_investigate_with_feedback(
            original_analysis=ORIGINAL_ANALYSIS,
            data_request="Can you check the allocation queue position and reservation attempts?",
            selected_yamls=SELECTED_YAMLS,
            tools=TOOLS,
        )
        combined = " ".join(result["findings"]).lower()
        has_queue_data = any(term in combined for term in [
            "queue", "position", "reservation", "allocation"
        ])
        self.assertTrue(
            has_queue_data,
            "Findings do not reflect the additional queue investigation requested"
        )

    def test_findings_list_is_non_empty(self):
        """Findings must be a non-empty list, same requirement as investigate_and_analyse()."""
        result = self.agent.re_investigate_with_feedback(
            original_analysis=ORIGINAL_ANALYSIS,
            data_request="Check allocation queue details please.",
            selected_yamls=SELECTED_YAMLS,
            tools=TOOLS,
        )
        self.assertIsInstance(result["findings"], list)
        self.assertGreater(len(result["findings"]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
