"""Task 2 - Fictional Nykaa-style support policies (NOT real Nykaa policy).

Each document has 2-5 sentences. No abbreviations or decimals with full stops are used,
so a simple sentence splitter stays accurate.
"""

KB_DOCUMENTS = [
    {
        "doc_id": "return_window",
        "title": "Return window by product category",
        "text": (
            "Beauty products can be returned within 7 days of delivery, and only if the seal is unbroken. "
            "Apparel and footwear can be returned within 15 days if they are unworn and the tags are attached. "
            "Electronics have a 10 day return window and must come with the original box and accessories. "
            "Home products can be returned within 14 days of delivery."
        ),
    },
    {
        "doc_id": "cod_refund",
        "title": "Cash on delivery refund timelines",
        "text": (
            "Refunds for cash on delivery orders are paid to the customer's bank account or as store credit. "
            "A bank refund takes 5 to 7 working days after the returned item passes quality checks. "
            "Store credit is added within 24 hours of the quality check. "
            "The customer must share valid bank details before the bank refund can be started."
        ),
    },
    {
        "doc_id": "delivery_sla",
        "title": "Delivery SLAs",
        "text": (
            "Metro cities receive orders within 2 to 4 working days of shipping. "
            "Other towns receive orders within 5 to 8 working days. "
            "Remote pin codes may need up to 10 working days. "
            "An order is called delayed when it has not arrived 2 days after the promised date."
        ),
    },
    {
        "doc_id": "reverse_pickup",
        "title": "Reverse-pickup eligibility",
        "text": (
            "Reverse pickup is offered for apparel, footwear, home and electronics orders when the pin code is serviceable. "
            "Beauty products are not picked up and must be dropped at a partner courier point because of hygiene rules. "
            "The pickup is attempted twice, and the customer must keep the item packed and ready. "
            "If both attempts fail, the return request is closed."
        ),
    },
    {
        "doc_id": "warranty_terms",
        "title": "Warranty terms by category",
        "text": (
            "Electronics carry a 12 month brand warranty that covers manufacturing defects only. "
            "Home appliances carry a 6 month warranty. "
            "Apparel, footwear and beauty products have no warranty, but a quality complaint can be raised within 7 days of delivery. "
            "Physical damage and water damage are not covered by any warranty."
        ),
    },
    {
        "doc_id": "order_cancellation",
        "title": "Order cancellation policy",
        "text": (
            "An order can be cancelled for free while its status is Placed. "
            "Once the order is Shipped, it cannot be cancelled, but the customer can refuse delivery or start a return after it arrives. "
            "Prepaid cancellations are refunded to the original payment method within 5 to 7 working days. "
            "Cancelled cash on delivery orders need no refund."
        ),
    },
    {
        "doc_id": "loyalty_points",
        "title": "Loyalty points redemption policy",
        "text": (
            "Each 1 loyalty point is worth 1 INR at checkout. "
            "Points can pay for at most 20 percent of the order value. "
            "Points expire 12 months after they are earned. "
            "Points earned on an order are removed if that order is returned or refunded."
        ),
    },
    {
        "doc_id": "payment_failure",
        "title": "Payment failure and retry policy",
        "text": (
            "If a payment fails but money is deducted, the amount is returned automatically within 3 to 5 working days. "
            "The customer can retry payment up to 3 times for one order within 30 minutes. "
            "After 3 failed attempts the cart is held for 24 hours and the customer should choose a different payment method. "
            "Customers should not share card numbers or one-time passwords with support agents."
        ),
    },
    {
        "doc_id": "size_exchange",
        "title": "Size exchange policy",
        "text": (
            "Size exchange is available for apparel and footwear within 15 days of delivery. "
            "The first exchange is free, and each later exchange has a 99 INR pickup fee. "
            "An exchange is only possible when the new size is in stock. "
            "If the size is not in stock, the customer can choose a refund instead."
        ),
    },
    {
        "doc_id": "damaged_item_claim",
        "title": "Damaged-item claim process",
        "text": (
            "A damaged or wrong item must be reported within 48 hours of delivery. "
            "The customer must upload clear photos of the item and the outer packaging. "
            "After the claim is approved, a free replacement or a full refund is given within 5 working days. "
            "Claims made after 48 hours go to the quality team for manual review."
        ),
    },
    {
        "doc_id": "international_shipping",
        "title": "International shipping restrictions",
        "text": (
            "International shipping is available only to selected countries, and the list is shown at checkout. "
            "Aerosols, perfumes with alcohol, and nail polish cannot be shipped abroad because of airline safety rules. "
            "Customs duties and taxes are paid by the customer at delivery. "
            "International orders cannot be returned for change of mind."
        ),
    },
    {
        "doc_id": "escalation_matrix",
        "title": "Customer-support escalation matrix",
        "text": (
            "Level 1 is the support agent, who must resolve a case within 24 hours. "
            "Level 2 is the team lead, who takes cases that are older than 24 hours or involve an order value above 10000 INR. "
            "Level 3 is the support manager, who handles cases older than 72 hours, repeated delivery delays, and legal threats. "
            "Delayed shipments with a high escalation score should go straight to Level 2."
        ),
    },
]
