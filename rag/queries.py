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

# Extra multi-topic questions used only to compare single-query vs multi-query retrieval.
MULTI_TOPIC = [
    ("Can I cancel a shipped order and how long does a refund take?", {"order_cancellation", "cod_refund"}),
    ("How long is the warranty on electronics and who do I contact if it fails?", {"warranty_terms", "escalation_matrix"}),
]

OUT_OF_SCOPE = [
    "Who won the cricket world cup?",
    "Give me a recipe for chocolate cake.",
    "Explain how quantum computers work.",
    "What is the capital of France?",
]

# Harder out-of-scope questions that share words with the domain (return, order, delivery ...).
NEAR_DOMAIN_OUT_OF_SCOPE = [
    "How do I return my car to the dealership?",
    "What is the weather in Mumbai today?",
    "Can you recommend a good movie to watch?",
    "How do I order a pizza online?",
]

# Legitimate questions that name a PRODUCT instead of a category word (used to evidence the product glossary).
PRODUCT_NOUN_QUESTIONS = [
    "How do I return my earphones?",
    "How can I return my headphones?",
    "How do I return my trimmer?",
    "How do I return my kurta?",
    "How do I return my sneakers?",
    "How do I return my cushion?",
    "Can I return my shampoo?",
    "How do I return my mascara?",
]
