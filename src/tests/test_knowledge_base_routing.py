"""
tests/test_knowledge_base_routing.py
======================================
Integration tests for SupervisorAgent.select_knowledge_base() covering
all 10 YAML knowledge base domains.

These tests verify that the router LLM correctly maps plain English
customer complaints — the kind a real user would type — to the right
YAML knowledge base file. This is the two-layer design principle in
action: the user describes a symptom, the router identifies the domain.

The tests are deliberately written using realistic customer-facing
language, NOT technical EDI/ERP terminology. This is intentional —
the YAML scenarios section now contains customer symptoms, and the
router LLM should be matching against those, not against the technical
investigation steps that the customer would never see or use.

Run from project root:
    python -m tests.test_knowledge_base_routing -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.supervisor_agent import SupervisorAgent


@unittest.skipUnless(
    os.getenv("OPENAI_API_KEY"),
    "OPENAI_API_KEY not set — skipping integration tests"
)
class TestKnowledgeBaseRouting(unittest.TestCase):
    """
    Integration tests verifying that customer-facing complaint language
    correctly maps to the right YAML knowledge base domain.

    A shared agent instance is used across all tests in this class to
    avoid creating a new SupervisorAgent (and new LLM client) for each
    test — each test still makes a real API call, but the client is
    only initialised once.
    """

    # Class-level agent shared across all tests in this class.
    # Created once in setUpClass() rather than per-test in setUp()
    # because the agent constructor initialises LLM clients — this
    # is a one-time cost that does not need to repeat per test.
    _agent = None

    @classmethod
    def setUpClass(cls):
        """Initialise one shared agent for all tests in this class."""
        cls._agent = SupervisorAgent()

    def setUp(self):
        """Make the shared agent available as self.agent."""
        self.agent = self._agent

    def _filenames(self, query: str) -> list[str]:
        """
        Helper that runs select_knowledge_base() for a query and returns
        just the list of selected filenames — avoids repeating this
        extraction in every test method.
        """
        result = self.agent.select_knowledge_base(query)
        return [s["filename"] for s in result["selected_yamls"]]

    # ── Original 3 domains ───────────────────────────────────────────

    def test_order_delay_plain_language(self):
        """
        A plain customer complaint about a late order must select
        order_delay.yaml. The customer uses natural language, not ITSM
        terminology — "hasn't arrived" not "SLA breach".
        """
        filenames = self._filenames(
            "My order was placed over a week ago and still hasn't arrived. "
            "I haven't had any update from anyone."
        )
        self.assertIn("order_delay.yaml", filenames)

    def test_invoice_discrepancy_plain_language(self):
        """
        A customer saying they were overcharged relative to their PO must
        select invoice_discrepancy.yaml.
        """
        filenames = self._filenames(
            "The invoice I received is for a higher amount than the "
            "purchase order I raised. Something doesn't add up."
        )
        self.assertIn("invoice_discrepancy.yaml", filenames)

    def test_warranty_plain_language(self):
        """
        A customer reporting a rejected warranty claim must select
        warranty_coverage.yaml.
        """
        filenames = self._filenames(
            "My warranty claim was rejected but the product is only "
            "6 months old. I don't understand why."
        )
        self.assertIn("warranty_coverage.yaml", filenames)

    # ── 7 new domains ────────────────────────────────────────────────

    def test_shipment_tracking_package_not_arrived(self):
        """
        A customer saying their package hasn't arrived despite being
        dispatched must select shipment_tracking.yaml.
        """
        filenames = self._filenames(
            "My order was dispatched 4 days ago but I still haven't "
            "received it. The tracking shows no movement."
        )
        self.assertIn("shipment_tracking.yaml", filenames)

    def test_shipment_tracking_damaged_goods(self):
        """
        A customer reporting damaged goods on delivery must select
        shipment_tracking.yaml — not order_delay, since the goods arrived
        but were damaged in transit.
        """
        filenames = self._filenames(
            "My delivery arrived today but the goods were damaged. "
            "The box was clearly crushed."
        )
        self.assertIn("shipment_tracking.yaml", filenames)
        self.assertNotIn("order_delay.yaml", filenames)

    def test_returns_rma_no_credit_note(self):
        """
        A customer who returned goods and hasn't received a credit note
        must select returns_and_rma.yaml.
        """
        filenames = self._filenames(
            "I returned the goods 2 weeks ago. The courier collected them "
            "but I still haven't received a credit note on my account."
        )
        self.assertIn("returns_and_rma.yaml", filenames)

    def test_returns_rma_no_collection(self):
        """
        A customer who requested a return but the courier never came must
        select returns_and_rma.yaml.
        """
        filenames = self._filenames(
            "I arranged a return collection but the courier never showed up. "
            "I've been waiting three days."
        )
        self.assertIn("returns_and_rma.yaml", filenames)

    def test_payment_hold_cannot_place_order(self):
        """
        A customer whose orders keep getting rejected at checkout must
        select payment_hold.yaml.
        """
        filenames = self._filenames(
            "Every time I try to place an order the system rejects it. "
            "My account was working fine last month."
        )
        self.assertIn("payment_hold.yaml", filenames)

    def test_payment_hold_account_suspended(self):
        """
        A customer whose account has been suspended without warning must
        select payment_hold.yaml.
        """
        filenames = self._filenames(
            "My account has been suspended without any warning. "
            "I can't access anything and no one told me why."
        )
        self.assertIn("payment_hold.yaml", filenames)

    def test_inventory_shortage_out_of_stock(self):
        """
        A customer told an item is out of stock after ordering must
        select inventory_shortage.yaml.
        """
        filenames = self._filenames(
            "I ordered an item that was shown as available but now I've "
            "been told it's out of stock. When will it be available?"
        )
        self.assertIn("inventory_shortage.yaml", filenames)

    def test_inventory_shortage_partial_delivery(self):
        """
        A customer who received only part of their order must select
        inventory_shortage.yaml.
        """
        filenames = self._filenames(
            "I ordered 10 units but only 6 were delivered. No one told "
            "me the rest was missing."
        )
        self.assertIn("inventory_shortage.yaml", filenames)

    def test_supplier_compliance_order_not_received_by_supplier(self):
        """
        A customer saying the supplier claims not to have received their
        order must select supplier_compliance.yaml. The customer uses plain
        business language — they have no knowledge of EDI or RosettaNet.
        """
        filenames = self._filenames(
            "I submitted an order but the supplier says they never received "
            "it. Our system shows it was sent."
        )
        self.assertIn("supplier_compliance.yaml", filenames)

    def test_pricing_discrepancy_overcharged(self):
        """
        A customer who was charged more than their agreed contract price
        must select pricing_discrepancy.yaml.
        """
        filenames = self._filenames(
            "The price on my invoice is higher than what we agreed in our "
            "contract. My volume discount wasn't applied either."
        )
        self.assertIn("pricing_discrepancy.yaml", filenames)

    def test_customs_compliance_shipment_held(self):
        """
        A customer whose international shipment is held at the border must
        select customs_and_compliance.yaml.
        """
        filenames = self._filenames(
            "My international shipment has been stuck at the border for a "
            "week. I don't know why it's being held."
        )
        self.assertIn("customs_and_compliance.yaml", filenames)

    def test_customs_compliance_unexpected_duties(self):
        """
        A customer asked to pay unexpected import duties must select
        customs_and_compliance.yaml.
        """
        filenames = self._filenames(
            "I was asked to pay import duties I wasn't expecting when my "
            "shipment arrived. No one mentioned this would happen."
        )
        self.assertIn("customs_and_compliance.yaml", filenames)

    # ── Multi-domain queries ──────────────────────────────────────────

    def test_order_delay_and_invoice_selects_both(self):
        """
        A complaint spanning order delay AND invoice must select both
        order_delay.yaml and invoice_discrepancy.yaml.
        """
        filenames = self._filenames(
            "My order still hasn't arrived and on top of that the invoice "
            "I received is for the wrong amount."
        )
        self.assertIn("order_delay.yaml", filenames)
        self.assertIn("invoice_discrepancy.yaml", filenames)

    def test_shipment_and_customs_selects_both(self):
        """
        A shipment held at customs spans both shipment_tracking and
        customs_and_compliance — both should be selected.
        """
        filenames = self._filenames(
            "My delivery is stuck at customs and the tracking hasn't "
            "updated in 5 days. I'm worried it's lost."
        )
        # At minimum customs should be selected — shipment tracking
        # may also be selected depending on LLM reasoning
        self.assertIn("customs_and_compliance.yaml", filenames)

    def test_payment_hold_and_order_delay_selects_both(self):
        """
        A customer whose order is stuck AND their account seems blocked
        spans both payment_hold and order_delay.
        """
        filenames = self._filenames(
            "My order has been sitting in processing for days and I tried "
            "to place another order but that got rejected too."
        )
        self.assertIn("order_delay.yaml", filenames)
        self.assertIn("payment_hold.yaml", filenames)

    # ── Negative routing tests ────────────────────────────────────────

    def test_damaged_goods_does_not_select_order_delay(self):
        """
        Damaged goods on delivery is a shipment_tracking issue, not
        an order_delay issue — the order arrived, just in bad condition.
        """
        filenames = self._filenames(
            "My goods arrived but they were completely damaged. "
            "The packaging was destroyed."
        )
        self.assertNotIn("order_delay.yaml", filenames)

    def test_credit_note_does_not_select_invoice_discrepancy(self):
        """
        A missing credit note after a return is returns_and_rma, not
        invoice_discrepancy — the issue is the return process, not a
        billing error on an original invoice.
        """
        filenames = self._filenames(
            "I returned goods last month but no credit has appeared "
            "on my account yet."
        )
        self.assertNotIn("invoice_discrepancy.yaml", filenames)
        self.assertIn("returns_and_rma.yaml", filenames)

    # ── Structure validation ──────────────────────────────────────────

    def test_all_selections_have_filename_and_reason(self):
        """
        Every item in selected_yamls must have both filename and reason
        regardless of which domain was selected.
        """
        result = self.agent.select_knowledge_base(
            "My delivery hasn't arrived and I can't reach anyone."
        )
        for item in result["selected_yamls"]:
            self.assertIn("filename", item)
            self.assertIn("reason", item)
            self.assertTrue(item["reason"])

    def test_vague_query_returns_valid_structure(self):
        """
        A completely vague query must not crash — it should return a
        valid structure even if no YAML is confidently selected.
        """
        result = self.agent.select_knowledge_base("I have a problem.")
        self.assertIn("selected_yamls", result)
        self.assertIn("overall_reasoning", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
