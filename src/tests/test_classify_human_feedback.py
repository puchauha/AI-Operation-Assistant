"""
tests/test_classify_human_feedback.py
=======================================
Integration tests for SupervisorAgent.classify_human_feedback()

This tests the fourth method in the pipeline. Unlike the earlier tests,
this one does NOT need to run the full investigation pipeline first —
classify_human_feedback() only needs an "original_analysis" dict to
compare the human's reply against. We can construct a fake (but
realistic-looking) analysis dict directly in the test, rather than
spending API calls re-running select_knowledge_base() and
investigate_and_analyse() every time.

This keeps these tests fast and cheap — only ONE LLM call per test
(the classification itself), instead of three or more.

Run from project root:
    python -m tests.test_classify_human_feedback -v
"""

import os
import sys
import unittest

# Add the src/ directory to Python's module search path so that
# "from agents.supervisor_agent import ..." resolves correctly no
# matter which folder this test is launched from.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.supervisor_agent import SupervisorAgent


# ════════════════════════════════════════════════════════════════════
# A fake "original_analysis" dict, shaped exactly like what
# investigate_and_analyse() would have returned for a real incident.
# We use this same fixed dict across multiple tests below so that the
# human's reply is always being judged against the same AI starting
# point — this makes test results easy to compare and reason about.
# ════════════════════════════════════════════════════════════════════
SAMPLE_ANALYSIS = {
    "operational_issue": "Order stuck in allocation queue due to warehouse allocation backlog.",
    "findings": [
        "Warehouse stock available: 18 units.",
        "Pending allocation requests: 34 for the order.",
        "Allocation status: Pending with 3 reservation attempts.",
    ],
    "reasoning_steps": [
        "Checked warehouse stock — 18 units available, so stock is not the issue.",
        "Checked allocation queue — order is at position 14 with 34 requests ahead.",
        "Three failed reservation attempts confirm a backlog, not a stock issue.",
    ],
    "root_cause": "Warehouse allocation backlog preventing order from being processed.",
    "short_term_fix": "Prioritise the order in the allocation queue and notify the customer.",
}


@unittest.skipUnless(
    os.getenv("OPENAI_API_KEY"),
    "OPENAI_API_KEY not set — skipping integration tests"
)
class TestClassifyHumanFeedback(unittest.TestCase):
    """
    Integration tests for classify_human_feedback().

    Each test sends a different style of human reply and checks that
    the classifier correctly identifies which of the four intents it is.
    """

    def setUp(self):
        """
        Called automatically before every test method.
        We create a fresh SupervisorAgent for each test so that no
        state leaks between tests — every test starts from a clean slate.
        """
        self.agent = SupervisorAgent()

    # ── Path 1: Approve ─────────────────────────────────────────────

    def test_clear_approval_classified_as_approve(self):
        """
        A clear, unambiguous approval message must be classified as
        "approve" with high confidence. This is the simplest, most
        common case and should never be misclassified.
        """
        result = self.agent.classify_human_feedback(
            human_message="Looks good, please go ahead and update the ticket.",
            original_analysis=SAMPLE_ANALYSIS,
        )
        self.assertEqual(result["intent"], "approve")

    def test_short_approval_classified_as_approve(self):
        """
        Real humans often reply very briefly. A short, casual approval
        like "yep, go for it" must still be classified correctly — the
        classifier should not require formal language to recognise approval.
        """
        result = self.agent.classify_human_feedback(
            human_message="Yep, go for it.",
            original_analysis=SAMPLE_ANALYSIS,
        )
        self.assertEqual(result["intent"], "approve")

    # ── Path 2: Request more data ───────────────────────────────────

    def test_request_for_more_data_classified_correctly(self):
        """
        When the human asks for additional information without disputing
        the existing findings, this must be classified as
        "request_more_data" — NOT as a correction, since the human is
        not saying anything is wrong, just that they want more.
        """
        result = self.agent.classify_human_feedback(
            human_message="Can you also check if there's an invoice issue on this order?",
            original_analysis=SAMPLE_ANALYSIS,
        )
        self.assertEqual(result["intent"], "request_more_data")

    def test_request_for_more_data_extracts_what_was_asked(self):
        """
        When intent is "request_more_data", the method should extract
        WHAT the human is asking for into extracted_data_request, so
        the caller does not need to re-parse the free text again later
        to figure out what additional investigation is needed.
        """
        result = self.agent.classify_human_feedback(
            human_message="Please also pull the carrier tracking details for this shipment.",
            original_analysis=SAMPLE_ANALYSIS,
        )
        self.assertEqual(result["intent"], "request_more_data")
        # extracted_data_request must not be empty/None when this is the intent
        self.assertIsNotNone(result["extracted_data_request"])
        self.assertTrue(len(result["extracted_data_request"]) > 0)

    # ── Path 3: Correct RCA / fix ────────────────────────────────────

    def test_rca_correction_classified_correctly(self):
        """
        When the human says the data is fine but the root cause is
        wrong, and provides their own correct root cause, this must be
        classified as "correct_rca_fix" — not as a rejection requiring
        re-investigation, since the human already knows the right answer.
        """
        result = self.agent.classify_human_feedback(
            human_message=(
                "The data you collected is correct, but this isn't actually "
                "a warehouse backlog — the order has a payment hold from "
                "finance. The fix should be to release the payment hold, "
                "not prioritise the allocation queue."
            ),
            original_analysis=SAMPLE_ANALYSIS,
        )
        self.assertEqual(result["intent"], "correct_rca_fix")

    def test_rca_correction_extracts_root_cause_and_fix(self):
        """
        When intent is "correct_rca_fix", both extracted_root_cause and
        extracted_fix should be populated with the human's correction —
        this structured extraction is what makes apply_human_override()
        (a future method) possible without re-parsing free text again.
        """
        result = self.agent.classify_human_feedback(
            human_message=(
                "Data is right, but the real root cause is a payment hold "
                "from finance, not a warehouse backlog. Fix: release the "
                "payment hold."
            ),
            original_analysis=SAMPLE_ANALYSIS,
        )
        self.assertEqual(result["intent"], "correct_rca_fix")
        self.assertIsNotNone(result["extracted_root_cause"])
        self.assertIsNotNone(result["extracted_fix"])

    # ── Path 4: Unclear ──────────────────────────────────────────────

    def test_ambiguous_message_classified_as_unclear(self):
        """
        A genuinely ambiguous or off-topic message must be classified as
        "unclear" rather than the classifier guessing one of the other
        three intents. Guessing incorrectly here is more dangerous than
        admitting uncertainty, because a wrong guess could trigger the
        wrong downstream action (e.g. closing a ticket that should have
        been re-investigated).
        """
        result = self.agent.classify_human_feedback(
            human_message="Hmm, not sure about this one.",
            original_analysis=SAMPLE_ANALYSIS,
        )
        self.assertEqual(result["intent"], "unclear")

    # ── Structural / safety checks ───────────────────────────────────

    def test_result_always_has_all_required_keys(self):
        """
        Regardless of which intent is detected, the result dict must
        always contain all six keys. Downstream code (e.g. a Slack event
        handler) should be able to safely access any of these fields
        without first checking if the key exists.
        """
        result = self.agent.classify_human_feedback(
            human_message="Approved.",
            original_analysis=SAMPLE_ANALYSIS,
        )
        required_keys = [
            "intent", "confidence", "extracted_data_request",
            "extracted_root_cause", "extracted_fix", "reasoning"
        ]
        for key in required_keys:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_intent_is_always_one_of_four_valid_values(self):
        """
        The "intent" field must always be exactly one of the four
        defined values — never an unexpected fifth value, a typo, or
        a free-form string. This protects every downstream piece of
        code that does something like:
            if feedback["intent"] == "approve": ...
            elif feedback["intent"] == "request_more_data": ...
        from silently falling through every branch if the LLM ever
        returns an unexpected value.
        """
        result = self.agent.classify_human_feedback(
            human_message="This all looks correct to me.",
            original_analysis=SAMPLE_ANALYSIS,
        )
        valid_intents = {"approve", "request_more_data", "correct_rca_fix", "unclear"}
        self.assertIn(result["intent"], valid_intents)

    def test_low_confidence_is_never_paired_with_approve(self):
        """
        This tests the safety rule built into the method: if the
        classifier ever reports low confidence, the intent must be
        forced to "unclear", never left as "approve" (or any other
        intent). A low-confidence approval is exactly the dangerous
        case we want to prevent — the system should never take a
        confident action (like closing a ticket) based on an
        uncertain reading of what the human meant.

        We deliberately send a vague, low-signal message here to try
        to trigger a low-confidence classification, then check that
        if confidence does come back low, intent is "unclear".
        """
        result = self.agent.classify_human_feedback(
            human_message="ok i guess maybe",
            original_analysis=SAMPLE_ANALYSIS,
        )
        if result["confidence"] == "low":
            self.assertEqual(result["intent"], "unclear")


if __name__ == "__main__":
    unittest.main(verbosity=2)
