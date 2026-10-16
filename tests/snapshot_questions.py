"""A broad set of refund / return / delivery / warranty / cancellation / exchange questions.

tests/test_duration_answers.py compares the CURRENT answers with the reviewed snapshot in
tests/answer_snapshot.json, so any change to answer selection shows up as an explicit diff.

NOTE: a snapshot guards against UNINTENDED changes; it does not certify that every frozen answer is right.
Known wrong answer frozen as-is: "How long does delivery take to remote areas?" (misses the 10-working-day
remote-pin-code sentence). "When will ... refund reach ...?" is also frozen with the general COD sentence.
"""
SNAPSHOT_QUESTIONS = [
    # refunds
    "How long will a refund take if I paid cash on delivery?",
    "How long does a COD refund take to reach my bank account?",
    "When will my cash on delivery refund reach my bank account?",
    "How quickly do I get store credit after a return?",
    "Do I need to give bank details for a refund?",
    "What happens if I cancel a prepaid order, how long is the refund?",
    "My payment failed but money was deducted, what happens?",
    "How long does it take to get my money back after a payment failure?",
    # returns
    "How many days do I have to return a lipstick?",
    "What is the return policy for beauty products?",
    "Return window for home products?",
    "How many days to return apparel?",
    "Return policy for electronics?",
    "How do I return my earphones?",
    "Can I get a reverse pickup for a beauty product?",
    # delivery
    "How many days will delivery take to a metro city?",
    "How long does delivery take to remote areas?",
    "How long should delivery take to other towns?",
    "When is a shipment called delayed?",
    # warranty / damage / exchange / cancellation / loyalty / escalation
    "How long is the warranty on my earphones?",
    "What warranty do electronics have?",
    "How long is the warranty on home appliances?",
    "How soon must I report a damaged item?",
    "How long do I have to exchange the size of my shoes?",
    "Can I cancel an order that has already been shipped?",
    "How long are loyalty points valid?",
    "How much of my order can I pay with loyalty points?",
    "Who handles a case that is older than 72 hours?",
    "How long does a support agent have to resolve a case?",
    "Can I ship nail polish to another country?",
    # multi-topic
    "My parcel is very late, how long should delivery take and who do I escalate to?",
    "Can I cancel a shipped order and how long does a refund take?",
]
