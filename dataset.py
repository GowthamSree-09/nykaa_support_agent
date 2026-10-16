"""Task 1 - Seeded, deterministic order dataset for the Nykaa support agent.

All data is fabricated. Run `python dataset.py` to print and verify the distribution.
"""
import random
from collections import Counter

# ---- Design choices (also copied into README.md) --------------------------------
# Seed 42 (first try) gave 10.0% delayed and only 3 Electronics records - too marginal.
# Per the brief we changed the SEED (never individual records) and picked seed 12.
SEED = 12
NUM_ORDERS = 60
CATEGORIES = ["Apparel", "Electronics", "Home", "Footwear", "Beauty"]
CATEGORY_WEIGHTS = [0.25, 0.15, 0.15, 0.20, 0.25]
STATUSES = ["Placed", "Shipped", "Delivered", "Returned", "Refunded"]
STATUS_WEIGHTS = [0.15, 0.20, 0.40, 0.15, 0.10]
DELAY_PROBABILITY = 0.20  # target ~20%; the real percentage is measured, not forced

# Order value in INR. One sentence of reasoning: Beauty/Home items are low-ticket
# (a lipstick or a cushion cover), apparel/footwear are mid-ticket, and electronics
# (earbuds, trimmers, hair dryers) are the highest-ticket category.
VALUE_RANGE_INR = {
    "Beauty": (199, 3999),
    "Home": (299, 9999),
    "Apparel": (499, 4999),
    "Footwear": (799, 6999),
    "Electronics": (1999, 49999),
}


def generate_orders(seed: int = SEED, n: int = NUM_ORDERS) -> list[dict]:
    rng = random.Random(seed)  # own RNG object -> never touches global random state
    orders = []
    for i in range(n):
        category = rng.choices(CATEGORIES, weights=CATEGORY_WEIGHTS)[0]
        status = rng.choices(STATUSES, weights=STATUS_WEIGHTS)[0]
        low, high = VALUE_RANGE_INR[category]
        orders.append(
            {
                "record_id": f"ORD-{1001 + i}",
                "category": category,
                "status": status,
                "order_value_inr": rng.randint(low, high),
                "days_since_created": rng.randint(0, 30),
                "delayed_shipment": rng.random() < DELAY_PROBABILITY,
            }
        )
    return orders


ORDERS = generate_orders()


def summarize(orders: list[dict]) -> dict:
    return {
        "total": len(orders),
        "per_category": dict(Counter(o["category"] for o in orders)),
        "per_status": dict(Counter(o["status"] for o in orders)),
        "delayed_pct": 100 * sum(o["delayed_shipment"] for o in orders) / len(orders),
    }


def validate(orders: list[dict]) -> list[str]:
    """Return a list of problems (empty list = dataset meets every Task 1 rule)."""
    problems = []
    s = summarize(orders)
    if s["total"] < 40:
        problems.append(f"only {s['total']} orders (<40)")
    for c in CATEGORIES:
        if s["per_category"].get(c, 0) < 3:
            problems.append(f"category {c} has <3 records")
    for st in STATUSES:
        if s["per_status"].get(st, 0) < 1:
            problems.append(f"status {st} missing")
    if not 10 <= s["delayed_pct"] <= 30:
        problems.append(f"delayed % {s['delayed_pct']:.1f} outside 10-30")
    for o in orders:
        if not (isinstance(o["days_since_created"], int) and 0 <= o["days_since_created"] <= 30):
            problems.append(f"{o['record_id']}: bad days_since_created")
    if len({o["record_id"] for o in orders}) != len(orders):
        problems.append("duplicate record_id")
    return problems


def get_order(record_id: str) -> dict | None:
    return next((o for o in ORDERS if o["record_id"] == record_id), None)


if __name__ == "__main__":
    s = summarize(ORDERS)
    print(f"seed={SEED}  orders={s['total']}")
    print("Count per category:", s["per_category"])
    print("Count per status:  ", s["per_status"])
    print(f"Delayed shipments: {s['delayed_pct']:.1f}%")
    print(f"Value range INR: {min(o['order_value_inr'] for o in ORDERS)}"
          f" - {max(o['order_value_inr'] for o in ORDERS)}")
    problems = validate(ORDERS)
    print("VALIDATION:", "PASS" if not problems else problems)
    print("First 3 records:", *ORDERS[:3], sep="\n  ")
