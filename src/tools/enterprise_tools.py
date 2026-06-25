# ============================================================
# File: enterprise_tools.py
#
# Purpose:
# Mock enterprise diagnostic tools for the AI Operations
# Assistant. Each tool represents a real enterprise system
# API call that the reasoning LLM can invoke during an
# investigation to gather specific operational data.
#
# Current state — MVP mock data:
# All tools return realistic, structured mock data that
# reflects what a real enterprise API would return. The
# data shapes and field names are designed to match real
# ERP, WMS, EDI, and carrier system responses, so that
# migrating to real API calls in Phase 3 is a drop-in
# replacement of the return values — the tool signatures,
# docstrings, and field names stay the same.
#
# Phase 3 upgrade path:
# Each tool will be replaced by a real HTTP call to an
# enterprise system (ERP, WMS, TMS, EDI gateway, customs
# broker portal) via AWS API Gateway + Lambda. The tool
# docstrings and return shapes will remain unchanged —
# the reasoning LLM's behaviour does not need to change
# when the data source becomes real.
#
# Tool organisation by domain:
#   1. Shipment tracking      — carrier and delivery status
#   2. Returns and RMA        — return authorisation and credit
#   3. Payment hold           — credit and account status
#   4. Inventory shortage     — stock levels and replenishment
#   5. Supplier compliance    — EDI and partner gateway status
#   6. Pricing discrepancy    — contract price and applied price
#   7. Customs and compliance — customs clearance and documents
#   8. Invoice discrepancy    — invoice status and 3-way match
#   9. Warranty coverage      — warranty record and claim status
#
# ============================================================

from langchain.tools import tool
from tools.inventory_tools import (
    get_warehouse_availability,
    get_allocation_queue_details,
)


# ============================================================
# 1. SHIPMENT TRACKING TOOLS
# ============================================================

@tool
def get_shipment_status(tracking_number: str):
    """
    Fetch the current carrier tracking status for a shipment.
    Provides the following information:
    - tracking_number: The carrier tracking reference for this shipment.
    - carrier_name: Name of the carrier handling the shipment.
    - current_status: Current status in the carrier system (e.g., In Transit,
      Out for Delivery, Delivered, Exception, Returned to Sender).
    - last_scan_location: The location of the most recent carrier scan event.
    - last_scan_timestamp: Date and time of the most recent scan.
    - exception_code: Carrier exception code if a delivery problem has been
      flagged (e.g., ADDRESS_NOT_FOUND, ACCESS_DENIED, REFUSED). Null if no
      exception.
    - exception_description: Human-readable description of the exception if
      one exists. Null if no exception.
    - estimated_delivery_date: Carrier's current estimated delivery date.
    - delivery_attempts: Number of delivery attempts made so far.
    """
    return {
        "tracking_number": tracking_number,
        "carrier_name": "FastFreight Express",
        "current_status": "Exception",
        "last_scan_location": "Birmingham Hub, UK",
        "last_scan_timestamp": "2026-06-23T09:14:00Z",
        "exception_code": "ADDRESS_NOT_FOUND",
        "exception_description": "Delivery address could not be located — "
                                 "unit number missing from label",
        "estimated_delivery_date": "2026-06-25",
        "delivery_attempts": 1,
    }


@tool
def get_proof_of_delivery(tracking_number: str):
    """
    Fetch proof of delivery details for a completed or attempted delivery.
    Provides the following information:
    - delivered: Boolean indicating whether delivery was completed.
    - delivery_timestamp: Date and time of successful delivery. Null if not
      yet delivered.
    - recipient_name: Name of the person who signed for the delivery. Null if
      not delivered or if no signature was required.
    - signature_obtained: Boolean indicating whether a signature was captured.
    - delivery_location: Description of where the goods were left (e.g.,
      Reception, Safe Place: Front Porch, Neighbour at No.12).
    - pod_image_available: Boolean indicating whether a photo or scanned
      signature image is available in the carrier system.
    """
    return {
        "delivered": False,
        "delivery_timestamp": None,
        "recipient_name": None,
        "signature_obtained": False,
        "delivery_location": None,
        "pod_image_available": False,
    }


# ============================================================
# 2. RETURNS AND RMA TOOLS
# ============================================================

@tool
def get_rma_status(order_number: str):
    """
    Fetch the return merchandise authorisation (RMA) status for an order.
    Provides the following information:
    - rma_number: The RMA reference number if one has been created. Null if
      no RMA exists for this order.
    - rma_status: Current status of the RMA (e.g., Pending Approval, Approved,
      Rejected, Expired, Goods Received, Closed).
    - created_date: Date the RMA was created. Null if no RMA exists.
    - expiry_date: Date the RMA authorisation expires — customer must return
      goods before this date. Null if no RMA exists.
    - return_collection_scheduled: Boolean indicating whether a carrier
      collection has been booked to pick up the returned goods.
    - collection_date: Scheduled collection date. Null if not yet booked.
    - goods_received_at_warehouse: Boolean indicating whether the returned
      goods have been physically received and logged at the returns warehouse.
    - goods_condition: Condition assessed at receipt (e.g., As New, Minor
      Damage, Significant Damage, Rejected). Null if not yet received.
    """
    return {
        "rma_number": None,
        "rma_status": None,
        "created_date": None,
        "expiry_date": None,
        "return_collection_scheduled": False,
        "collection_date": None,
        "goods_received_at_warehouse": False,
        "goods_condition": None,
    }


@tool
def get_credit_note_status(order_number: str):
    """
    Fetch the credit note status associated with a return or dispute for
    an order. Provides the following information:
    - credit_note_number: The credit note reference number. Null if no credit
      note has been raised.
    - credit_note_status: Current status (e.g., Pending, Approved, Issued,
      Applied to Account, On Hold).
    - credit_amount: The value of the credit note in the customer's invoicing
      currency. Null if not yet raised.
    - credit_currency: Currency of the credit note (e.g., GBP, EUR, USD).
    - issued_date: Date the credit note was formally issued to the customer.
      Null if not yet issued.
    - applied_to_account: Boolean indicating whether the credit has been
      applied to reduce the customer's outstanding balance.
    - original_invoice_number: The invoice number the credit relates to.
    """
    return {
        "credit_note_number": None,
        "credit_note_status": None,
        "credit_amount": None,
        "credit_currency": "GBP",
        "issued_date": None,
        "applied_to_account": False,
        "original_invoice_number": None,
    }


# ============================================================
# 3. PAYMENT HOLD TOOLS
# ============================================================

@tool
def get_account_credit_status(customer_id: str):
    """
    Fetch the credit and account status for a customer in the ERP system.
    Provides the following information:
    - account_status: Current status of the customer account (e.g., Active,
      On Hold, Suspended, Closed).
    - hold_reason: The reason code for a hold or suspension if applicable
      (e.g., CREDIT_LIMIT_EXCEEDED, OVERDUE_INVOICES, MANUAL_HOLD,
      CREDIT_INSURANCE_WITHDRAWN). Null if account is active.
    - hold_placed_by: The user or system that placed the hold. Null if active.
    - credit_limit: The customer's approved credit limit in their currency.
    - credit_used: The current amount of credit utilised (outstanding invoices
      plus open orders not yet invoiced).
    - credit_available: Remaining credit headroom before the limit is reached.
    - overdue_invoice_count: Number of invoices past their payment due date.
    - overdue_amount: Total value of overdue invoices.
    - oldest_overdue_days: Age in days of the oldest overdue invoice.
    """
    return {
        "account_status": "On Hold",
        "hold_reason": "OVERDUE_INVOICES",
        "hold_placed_by": "Credit Management System",
        "credit_limit": 50000.00,
        "credit_used": 48750.00,
        "credit_available": 1250.00,
        "overdue_invoice_count": 3,
        "overdue_amount": 12400.00,
        "oldest_overdue_days": 45,
    }


@tool
def get_payment_allocation_status(customer_id: str):
    """
    Fetch recent payment receipts and their allocation status for a customer.
    Provides the following information:
    - recent_payments: A list of the most recent payments received, each with:
        - payment_date: Date the payment was received.
        - amount: Payment amount.
        - currency: Payment currency.
        - allocated: Boolean indicating whether this payment has been matched
          to invoices.
        - unallocated_amount: Portion of the payment not yet matched to an
          invoice. Zero if fully allocated.
        - payment_reference: Customer's remittance reference if provided.
    - total_unallocated: Total unallocated cash sitting on the account.
    """
    return {
        "recent_payments": [
            {
                "payment_date": "2026-06-20",
                "amount": 15000.00,
                "currency": "GBP",
                "allocated": False,
                "unallocated_amount": 15000.00,
                "payment_reference": "BACS-REF-84721",
            }
        ],
        "total_unallocated": 15000.00,
    }


# ============================================================
# 4. INVENTORY SHORTAGE TOOLS
# ============================================================

@tool
def get_stock_levels(sku: str):
    """
    Fetch current stock levels for a specific product SKU across all warehouse
    locations. Provides the following information:
    - sku: The product SKU being checked.
    - total_available: Total units available to promise across all locations,
      after subtracting reservations for other orders.
    - locations: A list of warehouse locations with their individual stock:
        - location_name: Warehouse or DC name.
        - on_hand: Physical units in this location.
        - reserved: Units already reserved for other orders.
        - available: Units available at this location (on_hand minus reserved).
    - safety_stock_level: The minimum stock threshold configured for this SKU.
    - safety_stock_breached: Boolean — True if available stock is below the
      safety stock level.
    - product_status: Current lifecycle status of the product (e.g., Active,
      Discontinuing, Discontinued, Superseded).
    - superseded_by_sku: If the product has been superseded, the replacement
      SKU. Null if product is still active.
    """
    return {
        "sku": sku,
        "total_available": 0,
        "locations": [
            {
                "location_name": "Birmingham DC",
                "on_hand": 0,
                "reserved": 0,
                "available": 0,
            },
            {
                "location_name": "Manchester Hub",
                "on_hand": 12,
                "reserved": 12,
                "available": 0,
            },
        ],
        "safety_stock_level": 20,
        "safety_stock_breached": True,
        "product_status": "Active",
        "superseded_by_sku": None,
    }


@tool
def get_replenishment_orders(sku: str):
    """
    Fetch open purchase orders for replenishing a specific SKU from suppliers.
    Provides the following information:
    - open_purchase_orders: List of open POs for this SKU, each with:
        - po_number: Purchase order reference.
        - supplier_name: Name of the supplying partner.
        - ordered_quantity: Units ordered.
        - expected_receipt_date: Confirmed or estimated delivery date from
          the supplier.
        - po_status: Current PO status (e.g., Acknowledged, In Production,
          Shipped, Overdue).
    - total_on_order: Total units across all open purchase orders.
    - next_expected_receipt: The earliest expected receipt date across all
      open POs. Null if no open POs exist.
    """
    return {
        "open_purchase_orders": [
            {
                "po_number": "PO-2026-00412",
                "supplier_name": "Global Parts Ltd",
                "ordered_quantity": 200,
                "expected_receipt_date": "2026-07-05",
                "po_status": "Acknowledged",
            }
        ],
        "total_on_order": 200,
        "next_expected_receipt": "2026-07-05",
    }


# ============================================================
# 5. SUPPLIER COMPLIANCE / EDI TOOLS
# ============================================================

@tool
def get_edi_transaction_status(order_number: str):
    """
    Fetch EDI transaction log status for a purchase order with a trading partner.
    Provides the following information:
    - edi_850_sent: Boolean — whether an EDI 850 Purchase Order was transmitted.
    - edi_850_sent_timestamp: Timestamp of the 850 transmission. Null if not sent.
    - edi_855_received: Boolean — whether an EDI 855 Purchase Order Acknowledgement
      was received from the trading partner.
    - edi_855_received_timestamp: Timestamp of the 855 receipt. Null if not received.
    - edi_855_status: Acknowledgement status code from the partner (e.g., Accepted,
      Accepted with Changes, Rejected). Null if no 855 received.
    - rosettanet_pip_status: RosettaNet PIP 3A4 handshake status (e.g., Completed,
      Pending, Timed Out, Failed).
    - last_successful_transaction_timestamp: Timestamp of the last successfully
      completed EDI transaction with this trading partner.
    - message_queue_depth: Number of messages currently queued awaiting processing
      between our system and this trading partner.
    - pending_errors: Number of EDI transactions in an error state requiring
      manual review.
    """
    return {
        "edi_850_sent": True,
        "edi_850_sent_timestamp": "2026-06-23T08:30:00Z",
        "edi_855_received": False,
        "edi_855_received_timestamp": None,
        "edi_855_status": None,
        "rosettanet_pip_status": "Timed Out",
        "last_successful_transaction_timestamp": "2026-06-20T14:22:00Z",
        "message_queue_depth": 7,
        "pending_errors": 3,
    }


@tool
def get_partner_gateway_status(partner_id: str):
    """
    Fetch the connectivity and configuration status of a B2B trading partner
    gateway. Provides the following information:
    - partner_name: Display name of the trading partner.
    - gateway_status: Current connectivity status (e.g., Connected, Disconnected,
      Degraded, Certificate Expired).
    - last_successful_handshake: Timestamp of the last successful AS2 or SFTP
      handshake with the partner.
    - certificate_expiry_date: Expiry date of the partner's AS2 security
      certificate. Null if not applicable.
    - certificate_expired: Boolean — True if the certificate is past its
      expiry date.
    - partner_isa_id: The partner's EDI ISA identifier as currently configured
      in our system.
    - spec_version: The EDI message specification version currently configured
      for this partner.
    - test_mode_active: Boolean — True if the connection is currently configured
      to send to the partner's test environment rather than production. This is
      a common misconfiguration cause.
    """
    return {
        "partner_name": "Global Parts Ltd",
        "gateway_status": "Certificate Expired",
        "last_successful_handshake": "2026-06-20T14:22:00Z",
        "certificate_expiry_date": "2026-06-21",
        "certificate_expired": True,
        "partner_isa_id": "GLOBALPARTS01",
        "spec_version": "X12-5010",
        "test_mode_active": False,
    }


# ============================================================
# 6. PRICING DISCREPANCY TOOLS
# ============================================================

@tool
def get_contract_price(customer_id: str, sku: str):
    """
    Fetch the contracted price for a specific SKU on a specific customer's
    active price agreement. Provides the following information:
    - customer_id: The customer reference.
    - sku: The product SKU being checked.
    - contract_reference: The price agreement or contract reference number.
    - unit_price: The agreed unit price for this SKU under the active contract.
    - currency: The currency of the contracted price.
    - price_list_name: The name of the price list applied to this customer.
    - effective_from: The date from which this price agreement is active.
    - effective_to: The date on which this price agreement expires.
    - volume_discounts: A list of volume discount tiers configured for this
      customer and SKU, each with a minimum quantity and discount percentage.
    - promotional_price_active: Boolean — whether a time-limited promotional
      price is currently active for this SKU.
    - promotional_price: The promotional unit price if active. Null if not.
    """
    return {
        "customer_id": customer_id,
        "sku": sku,
        "contract_reference": "CONTRACT-2025-0142",
        "unit_price": 45.00,
        "currency": "GBP",
        "price_list_name": "Tier 2 Partner Pricing 2025",
        "effective_from": "2025-01-01",
        "effective_to": "2026-12-31",
        "volume_discounts": [
            {"min_quantity": 100, "discount_percent": 5.0},
            {"min_quantity": 500, "discount_percent": 10.0},
        ],
        "promotional_price_active": False,
        "promotional_price": None,
    }


@tool
def get_applied_price_on_order(order_number: str):
    """
    Fetch the actual price that was applied when a specific order was placed,
    for comparison against the contracted price. Provides the following:
    - order_number: The order reference.
    - price_list_applied: The name of the price list that was used when
      the order was booked.
    - unit_price_charged: The unit price that was actually charged.
    - currency: The currency used on the order.
    - discount_applied: The discount percentage applied at time of order.
      Zero if no discount was applied.
    - manual_override: Boolean — True if a manual price override was applied
      by a user rather than the system price list.
    - override_applied_by: The user who applied the manual override. Null if
      no override was applied.
    - surcharges: A list of surcharges added to this order, each with a name
      and amount. Empty list if no surcharges applied.
    """
    return {
        "order_number": order_number,
        "price_list_applied": "Standard List Price 2025",
        "unit_price_charged": 52.50,
        "currency": "GBP",
        "discount_applied": 0.0,
        "manual_override": False,
        "override_applied_by": None,
        "surcharges": [
            {"name": "Small Order Handling Fee", "amount": 15.00}
        ],
    }


# ============================================================
# 7. CUSTOMS AND COMPLIANCE TOOLS
# ============================================================

@tool
def get_customs_clearance_status(shipment_reference: str):
    """
    Fetch the customs clearance status for an international shipment from
    the customs broker portal. Provides the following information:
    - shipment_reference: The shipment or air waybill reference.
    - clearance_status: Current customs status (e.g., Cleared, Held for
      Inspection, Awaiting Documentation, Duty Assessment Pending,
      Released, Rejected).
    - hold_reason: The specific reason for a customs hold if applicable
      (e.g., HS_CODE_QUERY, MISSING_CERTIFICATE_OF_ORIGIN,
      DUTY_UNPAID, SANCTIONS_SCREENING_HIT, PROHIBITED_ITEM).
      Null if no hold.
    - import_duty_assessed: The import duty amount assessed by customs.
      Null if not yet assessed.
    - import_duty_paid: Boolean — whether the assessed duty has been paid.
    - duty_currency: Currency of the duty assessment.
    - hs_code_used: The Harmonised System commodity code declared on the
      entry.
    - entry_submitted_date: Date the customs entry was submitted.
    - broker_reference: The customs broker's own internal reference number.
    """
    return {
        "shipment_reference": shipment_reference,
        "clearance_status": "Held for Inspection",
        "hold_reason": "HS_CODE_QUERY",
        "import_duty_assessed": None,
        "import_duty_paid": False,
        "duty_currency": "GBP",
        "hs_code_used": "8471.30.00",
        "entry_submitted_date": "2026-06-22",
        "broker_reference": "BRK-2026-44821",
    }


@tool
def get_trade_document_status(shipment_reference: str):
    """
    Fetch the trade document checklist status for an international shipment.
    Provides the following information:
    - commercial_invoice_present: Boolean — whether a valid commercial invoice
      has been submitted.
    - packing_list_present: Boolean — whether a packing list has been submitted.
    - certificate_of_origin_present: Boolean — whether a certificate of origin
      has been submitted.
    - certificate_of_origin_valid: Boolean — whether the certificate of origin
      has been validated and accepted.
    - export_licence_required: Boolean — whether an export licence is required
      for this product and destination combination.
    - export_licence_present: Boolean — whether the required export licence
      has been provided. Null if not required.
    - sanctions_screening_status: Result of denied party screening (e.g.,
      Clear, Hit — Escalate Immediately, Pending Review).
    - additional_documents_required: List of any additional documents flagged
      as missing or required by the customs authority.
    """
    return {
        "commercial_invoice_present": True,
        "packing_list_present": True,
        "certificate_of_origin_present": False,
        "certificate_of_origin_valid": False,
        "export_licence_required": False,
        "export_licence_present": None,
        "sanctions_screening_status": "Clear",
        "additional_documents_required": [
            "Certificate of Origin — required for preferential duty rate claim"
        ],
    }


# ============================================================
# 8. INVOICE DISCREPANCY TOOLS
# ============================================================

@tool
def get_invoice_status(invoice_number: str):
    """
    Fetch the current status of a specific invoice in the ERP system.
    Provides the following information:
    - invoice_number: The invoice reference.
    - invoice_status: Current status (e.g., Draft, Issued, Approved, Disputed,
      On Hold, Paid, Cancelled).
    - invoice_amount: Total invoice value.
    - currency: Invoice currency.
    - invoice_date: Date the invoice was issued.
    - payment_due_date: Date by which payment is contractually due.
    - days_overdue: Number of days past the payment due date. Zero if not overdue.
    - purchase_order_number: The customer's purchase order number this invoice
      relates to.
    - goods_receipt_confirmed: Boolean — whether the corresponding goods receipt
      has been confirmed in the system.
    - dispute_reason: If the invoice is disputed, the stated reason. Null if not.
    """
    return {
        "invoice_number": invoice_number,
        "invoice_status": "Disputed",
        "invoice_amount": 4750.00,
        "currency": "GBP",
        "invoice_date": "2026-06-15",
        "payment_due_date": "2026-07-15",
        "days_overdue": 0,
        "purchase_order_number": "PO-2026-00387",
        "goods_receipt_confirmed": True,
        "dispute_reason": "Invoice amount exceeds purchase order value by £250",
    }


@tool
def get_three_way_match_result(order_number: str):
    """
    Fetch the result of the three-way match between purchase order, goods
    receipt note, and supplier invoice for a specific order. The three-way
    match is the standard accounts payable control that ensures we only pay
    for goods we ordered and actually received at the agreed price.
    Provides the following information:
    - match_status: Overall result (e.g., Matched, Mismatched — Price,
      Mismatched — Quantity, Mismatched — Both, Pending GRN,
      Pending Invoice).
    - po_total_value: Total value of the original purchase order.
    - grn_total_value: Total value of goods confirmed as received.
    - invoice_total_value: Total value of the supplier invoice.
    - mismatched_lines: A list of specific order line items where the match
      failed, each with the line number, PO quantity, received quantity,
      invoiced quantity, PO unit price, and invoiced unit price.
    - duplicate_invoice_detected: Boolean — whether a duplicate invoice
      reference has been detected for the same purchase order.
    """
    return {
        "match_status": "Mismatched — Price",
        "po_total_value": 4500.00,
        "grn_total_value": 4500.00,
        "invoice_total_value": 4750.00,
        "mismatched_lines": [
            {
                "line_number": 3,
                "sku": "SKU-00892",
                "po_quantity": 50,
                "received_quantity": 50,
                "invoiced_quantity": 50,
                "po_unit_price": 45.00,
                "invoiced_unit_price": 50.00,
                "variance": 250.00,
            }
        ],
        "duplicate_invoice_detected": False,
    }


# ============================================================
# 9. WARRANTY COVERAGE TOOLS
# ============================================================

@tool
def get_warranty_record(serial_number: str):
    """
    Fetch the warranty record for a specific product unit identified by its
    serial number. Provides the following information:
    - serial_number: The product's unique serial number.
    - product_sku: The product SKU.
    - product_description: Human-readable product name.
    - registered_to_customer_id: The customer ID the warranty is registered
      to. Null if the product is not registered.
    - purchase_date: The date the product was originally purchased.
    - warranty_start_date: The date the warranty period began.
    - warranty_expiry_date: The date the warranty expires.
    - warranty_tier: The type of warranty coverage (e.g., Manufacturer 12M,
      Distributor 24M, Extended Service Contract 36M).
    - warranty_active: Boolean — whether the warranty is currently in force.
    - previous_claim_count: Number of warranty claims previously made on
      this unit.
    - unit_modified: Boolean — whether the unit has been flagged as modified
      or tampered with, which typically voids warranty coverage.
    """
    return {
        "serial_number": serial_number,
        "product_sku": "SKU-00892",
        "product_description": "Industrial Control Module X200",
        "registered_to_customer_id": "CUST-10042",
        "purchase_date": "2024-11-15",
        "warranty_start_date": "2024-11-15",
        "warranty_expiry_date": "2026-11-14",
        "warranty_tier": "Distributor 24M",
        "warranty_active": True,
        "previous_claim_count": 0,
        "unit_modified": False,
    }


@tool
def get_claim_status(serial_number: str):
    """
    Fetch the status of an active or recent warranty claim for a specific
    product unit. Provides the following information:
    - claim_reference: The warranty claim reference number. Null if no claim
      has been raised.
    - claim_status: Current claim status (e.g., Submitted, Under Review,
      Approved, Rejected, Repair in Progress, Awaiting Parts,
      Repair Complete, Replacement Dispatched, Closed).
    - fault_description: The fault reported by the customer.
    - fault_classification: How the fault has been classified by the service
      team (e.g., Manufacturing Defect, Wear and Tear, Physical Damage,
      Software Fault, Misuse — Not Covered).
    - covered_under_warranty: Boolean — whether the fault has been determined
      to be covered under the active warranty terms.
    - rejection_reason: If the claim was rejected, the stated reason. Null
      if approved or pending.
    - repair_or_replace_decision: Whether the service team has decided to
      repair or replace the unit. Null if not yet decided.
    - sla_due_date: The date by which the repair or replacement should be
      completed under the warranty SLA.
    - sla_breached: Boolean — whether the service SLA has already been
      breached.
    """
    return {
        "claim_reference": None,
        "claim_status": None,
        "fault_description": None,
        "fault_classification": None,
        "covered_under_warranty": None,
        "rejection_reason": None,
        "repair_or_replace_decision": None,
        "sla_due_date": None,
        "sla_breached": False,
    }


# ============================================================
# TOOL REGISTRY — maps each YAML domain to relevant tools
#
# Purpose:
#   Rather than passing all 20 tools to the reasoning LLM on
#   every investigation (which is token-expensive), this registry
#   lets the caller pass only the tools relevant to the selected
#   YAML domain. The reasoning LLM still decides which tools to
#   call, but it does so from a focused, relevant set rather than
#   a full list of 20.
#
#   This directly addresses the OpenAI TPM rate limit issue seen
#   when running multiple investigation tests back-to-back — each
#   investigation uses fewer input tokens when only domain-relevant
#   tools are included.
#
# Usage:
#   from tools.enterprise_tools import get_tools_for_domain
#   tools = get_tools_for_domain("shipment_tracking")
# ============================================================

def get_tools_for_domain(yaml_id: str) -> list:
    """
    Return the list of tools relevant to a specific YAML knowledge
    base domain. The reasoning LLM will only see and call tools
    from this list during the investigation.

    Parameters
    ----------
    yaml_id : str
        The id field from the YAML metadata block.
        Example: "shipment_tracking", "payment_hold"

    Returns
    -------
    list
        List of @tool functions relevant to this domain.
        Falls back to all tools if the domain is not recognised.
    """
    registry = {
        "order_delay": [
            get_warehouse_availability,
            get_allocation_queue_details,
            get_stock_levels,
        ],
        "invoice_discrepancy": [
            get_invoice_status,
            get_three_way_match_result,
            get_payment_allocation_status,
        ],
        "warranty_coverage": [
            get_warranty_record,
            get_claim_status,
        ],
        "shipment_tracking": [
            get_shipment_status,
            get_proof_of_delivery,
        ],
        "returns_and_rma": [
            get_rma_status,
            get_credit_note_status,
        ],
        "payment_hold": [
            get_account_credit_status,
            get_payment_allocation_status,
        ],
        "inventory_shortage": [
            get_stock_levels,
            get_replenishment_orders,
            get_warehouse_availability,
            get_allocation_queue_details,
        ],
        "supplier_compliance": [
            get_edi_transaction_status,
            get_partner_gateway_status,
        ],
        "pricing_discrepancy": [
            get_contract_price,
            get_applied_price_on_order,
        ],
        "customs_and_compliance": [
            get_customs_clearance_status,
            get_trade_document_status,
        ],
    }

    tools = registry.get(yaml_id)
    if not tools:
        # Unknown domain — return all tools as fallback
        return [
            get_warehouse_availability, get_allocation_queue_details,
            get_shipment_status, get_proof_of_delivery,
            get_rma_status, get_credit_note_status,
            get_account_credit_status, get_payment_allocation_status,
            get_stock_levels, get_replenishment_orders,
            get_edi_transaction_status, get_partner_gateway_status,
            get_contract_price, get_applied_price_on_order,
            get_customs_clearance_status, get_trade_document_status,
            get_invoice_status, get_three_way_match_result,
            get_warranty_record, get_claim_status,
        ]
    return tools
