"""Shared query sets for Tasks 4 and 5 (query -> set of truly relevant parent documents)."""

IN_SCOPE = [
    ("How many days do I have to return a lipstick?", {"return_window"}),
    ("When will my cash on delivery refund reach my bank account?", {"cod_refund"}),
    ("Can I cancel my order after it has been shipped?", {"order_cancellation"}),
    ("My parcel arrived broken, how do I make a claim?", {"damaged_item_claim"}),
    ("How do I exchange my shoes for a bigger size?", {"size_exchange"}),
    ("How long is the warranty on my earphones?", {"warranty_terms"}),
    ("My parcel is very late, how long should delivery take and who do I escalate to?",
     {"delivery_sla", "escalation_matrix"}),
]

OUT_OF_SCOPE = [
    "Who won the cricket world cup?",
    "Give me a recipe for chocolate cake.",
    "Explain how quantum computers work.",
    "What is the capital of France?",
]
