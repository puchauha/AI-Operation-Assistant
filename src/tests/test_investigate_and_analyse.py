"""
tests/test_investigate_and_analyse.py
=======================================
Integration tests for SupervisorAgent.investigate_and_analyse()

Tests the full two-method pipeline:
    1. select_knowledge_base()   — identifies relevant YAMLs
    2. investigate_and_analyse() — invokes tools, returns structured dict

The pipeline result is computed once in setUp() and reused across all
tests in this class — avoiding redundant API calls per test.

Run from project root:
    python -m tests.test_investigate_and_analyse -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.supervisor_agent import SupervisorAgent
from tools.inventory_tools import get_warehouse_availability, get_allocation_queue_details

# All tools available to the reasoning LLM for this investigation.
# New tools can be added here as the tool library grows.
TOOLS = [
    get_warehouse_availability,
    get_allocation_queue_details,
]


@unittest.skipUnless(
    os.getenv("OPENAI_API_KEY"),
    "OPENAI_API_KEY not set — skipping integration tests"
)
class TestInvestigateAndAnalyse(unittest.TestCase):
    """
    Integration tests for investigate_and_analyse().

    setUp() runs the full pipeline once and stores the result.
    All test methods then assert against that single result —
    no redundant API calls, faster test run.
    """

    # Class-level variable to cache the pipeline result.
    # None means it hasn't been computed yet.
    # Prefixed with _ to indicate it is internal to this test class.
    _pipeline_result = None

    def setUp(self):
        """
        Run the full two-method pipeline once before the first test.
        Cache the result so subsequent tests reuse it without API calls.

        setUpClass() would be more appropriate for true class-level caching,
        but setUp() with a class variable works correctly and is simpler
        for developers unfamiliar with setUpClass().
        """
        # Only run the pipeline if it hasn't been run yet for this class
        if TestInvestigateAndAnalyse._pipeline_result is None:
            agent = SupervisorAgent()
            query = "My order ORD-84721 was supposed to arrive 3 days ago and still hasn't been dispatched."

            # Step 1: select relevant YAML files
            kb_result = agent.select_knowledge_base(query)
            self.assertGreater(
                kb_result["selected_count"], 0,
                "No YAMLs selected — cannot proceed with investigation"
            )

            # Step 2: investigate using tools and return structured dict
            TestInvestigateAndAnalyse._pipeline_result = agent.investigate_and_analyse(
                user_query=query,
                selected_yamls=kb_result["selected_yamls"],
                tools=TOOLS,
            )

        # Each test accesses the cached result via self.result
        self.result = TestInvestigateAndAnalyse._pipeline_result

    def test_returns_dict(self):
        """Method must return a dict — not a string or other type."""
        self.assertIsInstance(self.result, dict)

    def test_dict_has_required_keys(self):
        """
        All five keys must be present.
        present_to_human() depends on all of them — a missing key
        would cause a silent N/A in the Decision Card.
        """
        for key in ["operational_issue", "findings", "reasoning_steps",
                    "root_cause", "short_term_fix"]:
            self.assertIn(key, self.result, f"Missing key: {key}")

    def test_findings_is_non_empty_list(self):
        """Findings must be a list with at least one item."""
        self.assertIsInstance(self.result["findings"], list)
        self.assertGreater(len(self.result["findings"]), 0)

    def test_reasoning_steps_is_non_empty_list(self):
        """Reasoning steps must be a list with at least one item."""
        self.assertIsInstance(self.result["reasoning_steps"], list)
        self.assertGreater(len(self.result["reasoning_steps"]), 0)

    def test_findings_reflect_tool_results(self):
        """
        Tool results must appear in findings — LLM must not ignore them.
        We check for domain terms that should appear when warehouse and
        allocation tools have been called.
        """
        combined = " ".join(self.result["findings"]).lower()
        has_tool_data = any(term in combined for term in [
            "warehouse", "allocation", "stock", "queue", "pending", "reservation"
        ])
        self.assertTrue(has_tool_data, "Findings do not reflect tool results")


if __name__ == "__main__":
    unittest.main(verbosity=2)
