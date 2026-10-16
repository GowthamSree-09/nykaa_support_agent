"""Task 6 - check_order_status with a designed escalation score.

escalation_score = W_DELAY * delayed_shipment + W_AGE * (days_since_created / 30)
  - W_DELAY = 0.6  : a delayed shipment is the stronger signal that the customer needs help
  - W_AGE   = 0.4  : the older the order, the longer the customer has waited (normalized to 0-1)
  - weights add up to 1, so the score is always inside [0, 1].
A not-delayed order can reach at most 0.4, so age alone never triggers escalation.

Threshold: the score of a DELAYED order at the dataset's MEDIAN age (17 days in our data)
    = 0.6 + 0.4 * (17 / 30) = 0.827 -> 0.83.
In words: "escalate a delayed order that has already waited longer than half of all orders".
It is only recommended for open orders (Placed / Shipped); a Delivered or Refunded order is resolved.
"""
import statistics

import dataset

W_DELAY, W_AGE, MAX_DAYS = 0.6, 0.4, 30
OPEN_STATUSES = {"Placed", "Shipped"}
MEDIAN_DAYS = statistics.median(o["days_since_created"] for o in dataset.ORDERS)
ESCALATION_THRESHOLD = round(W_DELAY + W_AGE * MEDIAN_DAYS / MAX_DAYS, 2)


def escalation_score(delayed_shipment: bool, days_since_created: int) -> float:
    recency = min(max(days_since_created, 0), MAX_DAYS) / MAX_DAYS
    return round(W_DELAY * float(delayed_shipment) + W_AGE * recency, 4)


def check_order_status(record_id: str) -> dict:
    order = dataset.get_order(record_id.strip().upper())
    if order is None:
        return {"found": False, "record_id": record_id, "error": "Order not found"}
    score = escalation_score(order["delayed_shipment"], order["days_since_created"])
    return {
        "found": True,
        "record_id": order["record_id"],
        "status": order["status"],
        "order_value_inr": order["order_value_inr"],
        "escalation_score": score,
        "recommend_escalation": score >= ESCALATION_THRESHOLD and order["status"] in OPEN_STATUSES,
    }
