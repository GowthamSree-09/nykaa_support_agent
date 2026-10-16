import dataset
from agents import order_tool as ot


def test_lookup_matches_dataset():
    o = dataset.ORDERS[0]
    r = ot.check_order_status(o["record_id"])
    assert r["status"] == o["status"] and r["order_value_inr"] == o["order_value_inr"]


def test_unknown_order():
    assert ot.check_order_status("ORD-0")["found"] is False


def test_score_in_range_and_not_a_boolean_or():
    scores = {ot.check_order_status(o["record_id"])["escalation_score"] for o in dataset.ORDERS}
    assert all(0 <= s <= 1 for s in scores)
    assert len(scores) > 5  # a graded score, not just 0/1
    assert ot.escalation_score(True, 0) == 0.6 and ot.escalation_score(True, 30) == 1.0
    assert ot.escalation_score(False, 30) == 0.4


def test_threshold_from_dataset_median():
    assert ot.MEDIAN_DAYS == 17 and ot.ESCALATION_THRESHOLD == 0.83


def test_escalation_cases():
    assert ot.check_order_status("ORD-1023")["recommend_escalation"] is True    # Shipped, delayed, 25 days
    assert ot.check_order_status("ORD-1009")["recommend_escalation"] is False   # delayed but Delivered
    assert ot.check_order_status("ORD-1055")["recommend_escalation"] is False   # delayed, only 10 days
