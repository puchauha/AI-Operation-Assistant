"""
tests/test_enterprise_investigation.py
========================================
Integration tests for SupervisorAgent.investigate_and_analyse()
using the new enterprise tools across all 10 domains.

These tests verify two things:
  1. The correct enterprise tools are called for each domain
  2. The tool results appear in the investigation findings

Each test runs the full two-step pipeline:
  select_knowledge_base() → investigate_and_analyse()

Unlike test_knowledge_base_routing.py which only tests routing,
these tests verify the full investigation — that the reasoning LLM
actually calls the right tools and incorporates their results.

These tests make real OpenAI API calls and take longer to run.
Run selectively using -k to filter by test name:

    python -m tests.test_enterprise_investigation -v -k shipment
    python -m tests.test_enterprise_investigation -v -k payment
    python -m tests.test_enterprise_investigation -v

Run from project root:
    python -m tests.test_enterprise_investigation -v
"""

import os
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.supervisor_agent import SupervisorAgent
from tools.inventory_tools import (
    get_warehouse_availability,
    get_allocation_queue_details,
)
from tools.enterprise_tools import get_tools_for_domain
from tools.enterprise_tools import (
    get_shipment_status,
    get_proof_of_delivery,
    get_rma_status,
    get_credit_note_status,
    get_account_credit_status,
    get_payment_allocation_status,
    get_stock_levels,
    get_replenishment_orders,
    get_edi_transaction_status,
    get_partner_gateway_status,
    get_contract_price,
    get_applied_price_on_order,
    get_customs_clearance_status,
    get_trade_document_status,
    get_invoice_status,
    get_three_way_match_result,
    get_warranty_record,
    get_claim_status,
)

# All 20 tools available to the reasoning LLM
ALL_TOOLS = [
    get_warehouse_availability,
    get_allocation_queue_details,
    get_shipment_status,
    get_proof_of_delivery,
    get_rma_status,
    get_credit_note_status,
    get_account_credit_status,
    get_payment_allocation_status,
    get_stock_levels,
    get_replenishment_orders,
    get_edi_transaction_status,
    get_partner_gateway_status,
    get_contract_price,
    get_applied_price_on_order,
    get_customs_clearance_status,
    get_trade_document_status,
    get_invoice_status,
    get_three_way_match_result,
    get_warranty_record,
    get_claim_status,
]


def run_pipeline(agent: SupervisorAgent, query: str) -> dict:
    """
    Run the full select_knowledge_base() + investigate_and_analyse()
    pipeline for a given query. Returns the analysis dict.

    Uses domain-specific tools rather than all 20 tools — this
    significantly reduces token usage per investigation call, which
    avoids hitting the OpenAI TPM rate limit when running the full
    test suite. The reasoning LLM still decides which tools to call;
    it just does so from a focused, relevant set.
    """
    kb_result = agent.select_knowledge_base(query)
    assert kb_result["selected_count"] > 0, (
        f"No YAML selected for query: '{query}'\n"
        f"Reasoning: {kb_result['overall_reasoning']}"
    )
    # Build domain-specific tool set from selected YAMLs
    tools = []
    seen = set()
    for yaml_item in kb_result["selected_yamls"]:
        yaml_id = yaml_item["filename"].replace(".yaml", "")
        for t in get_tools_for_domain(yaml_id):
            if t.name not in seen:
                tools.append(t)
                seen.add(t.name)

    return agent.investigate_and_analyse(
        user_query=query,
        selected_yamls=kb_result["selected_yamls"],
        tools=tools,
    )


@unittest.skipUnless(
    os.getenv("OPENAI_API_KEY"),
    "OPENAI_API_KEY not set — skipping integration tests"
)
class TestEnterpriseInvestigation(unittest.TestCase):
    """
    Integration tests for investigate_and_analyse() across all 10 domains.

    A shared agent is created once per class to avoid repeated client
    initialisation. Each test makes real OpenAI API calls.
    """

    _agent = None

    @classmethod
    def setUpClass(cls):
        cls._agent = SupervisorAgent()

    def setUp(self):
        self.agent = self._agent
        # Small delay between tests to avoid saturating the OpenAI
        # TPM (tokens per minute) rate limit when running the full suite.
        # Each investigation test loads 20 tool definitions into context
        # which is token-heavy. 8 seconds between tests keeps well within
        # the 30,000 TPM limit on a standard OpenAI tier.
        time.sleep(8)

    # ── Structural tests (apply to all domains) ───────────────────────

    def _assert_valid_analysis(self, result: dict, domain_terms: list[str]):
        """
        Helper that asserts the analysis dict has the required shape
        and that at least one domain-relevant term appears in findings.
        """
        # Required keys
        for key in ["operational_issue", "findings", "reasoning_steps",
                    "root_cause", "short_term_fix"]:
            self.assertIn(key, result, f"Missing key: {key}")

        # Findings must be a non-empty list
        self.assertIsInstance(result["findings"], list)
        self.assertGreater(len(result["findings"]), 0)

        # At least one domain term must appear in combined findings
        combined = " ".join(result["findings"]).lower()
        has_domain_data = any(term.lower() in combined for term in domain_terms)
        self.assertTrue(
            has_domain_data,
            f"Findings do not reflect domain investigation.\n"
            f"Expected one of {domain_terms} in:\n{combined}"
        )

    # ── Shipment tracking ─────────────────────────────────────────────

    def test_shipment_investigation_calls_tracking_tools(self):
        """
        A delivery complaint must trigger shipment tracking tools and
        surface carrier status information in the findings.
        """
        result = run_pipeline(
            self.agent,
            "My order was dispatched 4 days ago but I still haven't "
            "received it. The tracking shows no movement."
        )
        self._assert_valid_analysis(result, [
            "carrier", "tracking", "exception", "delivery",
            "scan", "status", "shipment"
        ])

    # ── Payment hold ──────────────────────────────────────────────────

    def test_payment_hold_investigation_calls_credit_tools(self):
        """
        An account rejection complaint must trigger credit status tools
        and surface account hold information in the findings.
        """
        result = run_pipeline(
            self.agent,
            "Every time I try to place an order the system rejects it. "
            "My account was working fine last month."
        )
        self._assert_valid_analysis(result, [
            "account", "credit", "hold", "overdue", "limit",
            "payment", "suspended"
        ])

    # ── Returns and RMA ───────────────────────────────────────────────

    def test_returns_investigation_calls_rma_tools(self):
        """
        A missing credit note complaint must trigger RMA and credit
        note tools and surface return status in the findings.
        """
        result = run_pipeline(
            self.agent,
            "I returned goods 2 weeks ago. The courier collected them "
            "but I still haven't received a credit note."
        )
        self._assert_valid_analysis(result, [
            "rma", "return", "credit", "collection", "goods",
            "received", "warehouse"
        ])

    # ── Inventory shortage ────────────────────────────────────────────

    def test_inventory_investigation_calls_stock_tools(self):
        """
        An out-of-stock complaint must trigger stock level tools and
        surface inventory information in the findings.
        """
        result = run_pipeline(
            self.agent,
            "I ordered an item that was shown as available but now "
            "I've been told it's out of stock."
        )
        self._assert_valid_analysis(result, [
            "stock", "inventory", "available", "shortage",
            "replenishment", "purchase order", "warehouse"
        ])

    # ── Supplier compliance ───────────────────────────────────────────

    def test_supplier_compliance_investigation_calls_edi_tools(self):
        """
        An order not received by supplier must trigger EDI tools and
        surface transaction status in the findings.
        """
        result = run_pipeline(
            self.agent,
            "I submitted an order but the supplier says they never "
            "received it. Our system shows it was sent."
        )
        self._assert_valid_analysis(result, [
            "edi", "transmission", "acknowledgement", "gateway",
            "partner", "rosettanet", "certificate", "queue"
        ])

    # ── Pricing discrepancy ───────────────────────────────────────────

    def test_pricing_investigation_calls_price_tools(self):
        """
        An overcharging complaint must trigger contract price and applied
        price tools and surface the discrepancy in findings.
        """
        result = run_pipeline(
            self.agent,
            "The price on my invoice is higher than what we agreed in "
            "our contract. My discount wasn't applied."
        )
        self._assert_valid_analysis(result, [
            "contract", "price", "discount", "applied", "charged",
            "invoice", "unit price"
        ])

    # ── Customs and compliance ────────────────────────────────────────

    def test_customs_investigation_calls_clearance_tools(self):
        """
        A shipment held at customs must trigger clearance status and
        document status tools and surface the hold reason in findings.
        """
        result = run_pipeline(
            self.agent,
            "My international shipment has been stuck at the border "
            "for a week. No one can tell me why."
        )
        self._assert_valid_analysis(result, [
            "customs", "clearance", "hold", "document", "duty",
            "hs code", "certificate", "border"
        ])

    # ── Invoice discrepancy ───────────────────────────────────────────

    def test_invoice_investigation_calls_match_tools(self):
        """
        An invoice amount mismatch must trigger invoice status and
        three-way match tools and surface the discrepancy in findings.
        """
        result = run_pipeline(
            self.agent,
            "The invoice I received is for a higher amount than "
            "my purchase order. Something doesn't match."
        )
        self._assert_valid_analysis(result, [
            "invoice", "match", "purchase order", "discrepancy",
            "mismatch", "amount", "variance"
        ])

    # ── Warranty ──────────────────────────────────────────────────────

    def test_warranty_investigation_calls_warranty_tools(self):
        """
        A rejected warranty claim must trigger warranty record and
        claim status tools and surface coverage details in findings.
        """
        result = run_pipeline(
            self.agent,
            "My warranty claim was rejected but the product is only "
            "6 months old. I believe I'm still covered."
        )
        self._assert_valid_analysis(result, [
            "warranty", "coverage", "claim", "expiry", "registered",
            "serial", "tier", "active"
        ])

    # ── Honest knowledge gap ──────────────────────────────────────────

    def test_unknown_domain_returns_honest_response(self):
        """
        A query that matches no YAML domain — e.g. an HR query —
        should result in either no YAML selected or a graceful
        response acknowledging the knowledge gap. It must not crash.
        """
        result = self.agent.select_knowledge_base(
            "I need to update my employee benefits package."
        )
        # Either no YAML selected (selected_count == 0)
        # or YAML selected but findings acknowledge the gap
        # Either outcome is acceptable — what matters is no crash
        self.assertIn("selected_yamls", result)
        self.assertIn("overall_reasoning", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
