"""
tests/test_select_knowledge_base.py
=====================================
Integration tests for SupervisorAgent.select_knowledge_base()

Environment setup:
    - OPENAI_API_KEY must be set in the .env file at the project root
    - load_dotenv() is called inside supervisor_agent.py — not needed here
    - All tests are skipped automatically if the API key is not set

Run from project root:
    python -m tests.test_select_knowledge_base -v
"""

import os
import sys
import unittest

# Add the src/ directory to Python's module search path.
# This allows `from agents.supervisor_agent import ...` to resolve correctly
# regardless of which directory the test is run from.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.supervisor_agent import SupervisorAgent


@unittest.skipUnless(
    os.getenv("OPENAI_API_KEY"),
    "OPENAI_API_KEY not set — skipping integration tests"
)
class TestSelectKnowledgeBase(unittest.TestCase):
    """
    Integration tests for select_knowledge_base().

    Each test exercises a different type of user query to verify that
    the router LLM selects the correct YAML file(s) in each case.
    """

    def setUp(self):
        """
        Called automatically before each test method.
        Creates a fresh agent instance so tests are fully independent.
        """
        self.agent = SupervisorAgent()

    def test_order_delay_selects_correct_yaml(self):
        """A clear order delay complaint must select order_delay.yaml."""
        result = self.agent.select_knowledge_base(
            "My order was supposed to arrive 3 days ago and still hasn't been dispatched."
        )
        filenames = [s["filename"] for s in result["selected_yamls"]]
        self.assertIn("order_delay.yaml", filenames)
        self.assertNotIn("warranty_coverage.yaml", filenames)

    def test_invoice_query_selects_correct_yaml(self):
        """A billing dispute complaint must select invoice_discrepancy.yaml."""
        result = self.agent.select_knowledge_base(
            "The invoice we received from the partner is higher than the purchase order amount."
        )
        filenames = [s["filename"] for s in result["selected_yamls"]]
        self.assertIn("invoice_discrepancy.yaml", filenames)

    def test_multi_domain_query_selects_multiple_yamls(self):
        """
        A complaint spanning two domains must result in both YAMLs being selected.
        The LLM must not artificially restrict itself to a single file.
        """
        result = self.agent.select_knowledge_base(
            "My order is delayed and on top of that the invoice amount looks completely wrong."
        )
        filenames = [s["filename"] for s in result["selected_yamls"]]
        self.assertIn("order_delay.yaml",         filenames)
        self.assertIn("invoice_discrepancy.yaml",  filenames)

    def test_warranty_query_selects_correct_yaml(self):
        """An after-sales warranty complaint must select warranty_coverage.yaml."""
        result = self.agent.select_knowledge_base(
            "I raised a warranty claim last week and it was rejected without any explanation."
        )
        filenames = [s["filename"] for s in result["selected_yamls"]]
        self.assertIn("warranty_coverage.yaml", filenames)

    def test_each_selection_has_filename_and_reason(self):
        """
        Every item in selected_yamls must have both filename and reason.
        The reason field feeds the Decision Card audit trail.
        """
        result = self.agent.select_knowledge_base("Order not dispatched.")
        for item in result["selected_yamls"]:
            self.assertIn("filename", item)
            self.assertIn("reason",   item)
            self.assertTrue(item["reason"])

    def test_vague_query_returns_valid_structure(self):
        """
        A vague query must return a valid structure without crashing.
        An empty selection with an explanation is an acceptable outcome.
        """
        result = self.agent.select_knowledge_base("Something is wrong.")
        self.assertIn("selected_yamls",    result)
        self.assertIn("overall_reasoning", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
