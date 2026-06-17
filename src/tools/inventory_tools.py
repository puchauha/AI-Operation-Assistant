from langchain.tools import tool


# ============================================================
# INVENTORY TOOLS
# ============================================================


@tool
def get_warehouse_availability(order_number: str):
    """
    Fetch inventory availability for the items in this order. It provides following information:
    - warehouse_stock: Current stock available in the warehouse for the items in the order. 
    - pending_allocation_requests: Number of pending allocation requests for the items in the order, which indicates how many units are currently being processed for allocation but have not yet been reserved.

    """

    return {
        "warehouse_stock": 0,
        "pending_allocation_requests": 34,
    }


@tool
def get_allocation_queue_details(order_number: str):
    """
    Fetch allocation queue details for the order. It provides following information:
    - allocation_status: Current status of the inventory allocation for the order (e.g., Pending, In Progress, Completed).
    - reservation_attempts: Number of attempts made to reserve inventory for the order. 
    - allocation_queue_position: Current position of the order in the allocation queue, if applicable.      
    - allocation_failure_reason: If there have been failed attempts to allocate inventory, this field provides the reason for the failure (e.g., Insufficient stock, Warehouse allocation backlog).

    """
    return {
        "allocation_status": "Pending",
        "reservation_attempts": 0,
        "allocation_queue_position": 1,
        "allocation_failure_reason":
            " "
    }

