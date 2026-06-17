"""
tests/test_present_to_human.py
================================
Integration tests for SupervisorAgent.present_to_human()

Tests the complete three-method pipeline:
    1. select_knowledge_base()   — identifies relevant YAMLs
    2. investigate_and_analyse() — invokes tools, returns structured dict
    3. present_to_human()        — formats dict into Decision Card string

The full pipeline is run once in setUp() and the Decision Card string
is cached across all tests — avoiding redundant API calls.

Run from project root:
    python -m tests.test_present_to_human -v
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


@unittest.skipUnless(
    os.getenv("OPENAI_API_KEY"),
    "OPENAI_API_KEY not set — skipping integration tests"
)
class TestPresentToHuman(unittest.TestCase):
    """
    Integration tests for present_to_human().

    The full three-method pipeline is run once. The resulting Decision Card
    string is cached at the class level and reused by all test methods.
    """

    # Cache the Decision Card string across all tests in this class
    _decision_card = None
    _query         = "My order ORD-84721 was supposed to arrive 3 days ago and still hasn't been dispatched."

    def setUp(self):
        """Run the full pipeline once and cache the Decision Card."""
        if TestPresentToHuman._decision_card is None:
            agent = SupervisorAgent()

            # Step 1: select relevant YAML files
            kb_result = agent.select_knowledge_base(self._query)
            self.assertGreater(kb_result["selected_count"], 0, "No YAMLs selected")

            # Step 2: investigate and return structured dict
            analysis = agent.investigate_and_analyse(
                user_query=self._query,
                selected_yamls=kb_result["selected_yamls"],
                tools=TOOLS,
            )

            # Step 3: format into Decision Card string
            TestPresentToHuman._decision_card = agent.present_to_human(
                user_query=self._query,
                analysis=analysis,
            )

        # Each test accesses the cached card via self.card
        self.card = TestPresentToHuman._decision_card

    def test_returns_non_empty_string(self):
        """present_to_human() must return a non-empty string."""
        self.assertIsInstance(self.card, str)
        self.assertGreater(len(self.card.strip()), 0)

    def test_contains_header(self):
        """The card must open with the AI Operations Assistant header."""
        self.assertIn("AI OPERATIONS ASSISTANT", self.card)
        self.assertIn("DECISION CARD",            self.card)

    def test_contains_customer_complaint(self):
        """
        The customer complaint section must contain the original user query
        verbatim. The human reviewer must see the exact complaint they are
        reviewing — no paraphrasing or truncation.
        """
        # Check the exact query string appears unchanged in the card.
        # present_to_human() displays the complaint without wrapping
        # precisely so this assertion can be made reliably.
        self.assertIn(f'"{self._query}"', self.card)

    def test_contains_all_sections(self):
        """
        All five analysis sections plus the status section must be present.
        A missing section means the human reviewer has incomplete information.
        """
        required_sections = [
            "MOST PROBABLE OPERATIONAL ISSUE",
            "ENTERPRISE FINDINGS",
            "AI REASONING STEPS",
            "PROBABLE ROOT CAUSE",
            "SUGGESTED SHORT-TERM FIX",
            "AWAITING HUMAN REVIEW",
        ]
        for section in required_sections:
            self.assertIn(section, self.card, f"Missing section: {section}")

    def test_contains_action_buttons(self):
        """
        All four human action options must be present.
        These are the interface between the AI investigation and human decision.
        """
        for action in ["APPROVE", "OVERRIDE", "ESCALATE", "MORE DATA"]:
            self.assertIn(action, self.card, f"Missing action: {action}")

    def test_full_card_prints_cleanly(self):
        """
        Smoke test — print the full Decision Card for visual inspection.
        If no exception is raised, formatting completed successfully.
        """
        print("\n" + self.card)
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
