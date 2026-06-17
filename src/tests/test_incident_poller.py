"""
tests/test_incident_poller.py
================================
Unit tests for the helper functions in integrations/incident_poller.py

These tests focus on _build_user_query() — the one piece of pure logic
in incident_poller.py that can be tested without needing a real
ServiceNow connection, a real Slack webhook, or a real OpenAI API key.

process_incident() and run_poll_loop() are NOT unit tested here because
they require live ServiceNow and Slack credentials to exercise
meaningfully — testing them properly belongs in a manual end-to-end
smoke test against a real (dev) ServiceNow instance, not an automated
unit test that runs on every commit.

Run from project root:
    python -m tests.test_incident_poller -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from integrations.incident_poller import _build_user_query


class TestBuildUserQuery(unittest.TestCase):
    """
    Unit tests for _build_user_query().

    No API keys or external services are needed for any test in this
    class — this is pure string-handling logic.
    """

    def test_combines_short_description_and_description(self):
        """
        When both fields are present and different, both must appear
        in the combined query so the AI agent has full context.
        """
        result = _build_user_query({
            "short_description": "Order not dispatched",
            "description": "Customer says order ORD-123 was due 3 days ago.",
        })
        self.assertIn("Order not dispatched", result)
        self.assertIn("ORD-123", result)

    def test_handles_empty_description(self):
        """
        When description is an empty string, the result should be just
        the short_description, with no stray separator or whitespace
        left over from the empty field.
        """
        result = _build_user_query({
            "short_description": "Order delayed",
            "description": "",
        })
        self.assertEqual(result, "Order delayed")

    def test_avoids_duplicating_identical_fields(self):
        """
        ServiceNow sometimes populates short_description and description
        with the exact same text. In that case we must not show the
        same sentence twice in the query sent to the AI agent.
        """
        result = _build_user_query({
            "short_description": "Same text",
            "description": "Same text",
        })
        self.assertEqual(result, "Same text")

    def test_handles_missing_fields_without_crashing(self):
        """
        An incident dict with neither field present must not raise an
        exception — it should simply return an empty string, which
        process_incident() then detects and skips gracefully.
        """
        result = _build_user_query({})
        self.assertEqual(result, "")

    def test_handles_none_values_without_crashing(self):
        """
        ServiceNow can return None (not just an empty string) for an
        unset field. The "or ''" pattern inside _build_user_query()
        is specifically there to guard against this — this test
        confirms that guard actually works.
        """
        result = _build_user_query({
            "short_description": None,
            "description": None,
        })
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
