"""Relevance and question-understanding helpers shared by retrieval, routing, memory and the reviewer.

* Category gate: if the customer asks about one product category (beauty, home, footwear ...), a policy
  sentence that talks ONLY about other categories is not relevant.
* Product glossary: the embedding model does not link product nouns to our category words
  ("earphones" scores 0.33 against the return policy, "electronics" scores 0.56), so retrieval uses the
  category word instead of the product noun. Glossary words must NOT appear in the knowledge base
  (a test enforces this), so exact matches such as "nail polish" are never disturbed.
* Sub-queries: a question like "how long does delivery take and who do I escalate to" is split into
  parts so each topic is retrieved separately (multi-query retrieval).
* References: "it / that / this" with nothing to point at is ambiguous; it is resolved from earlier
  words of the same question or from the session, otherwise the bot asks for clarification.
* Topic hints / answer type: "contact / escalate / complain" questions belong to the escalation matrix,
  and a "who ..." question must be answered by a sentence that names a role.
"""
import re

from kb.documents import KB_DOCUMENTS
from llm.mock_llm import tokens

# ---- categories and the product glossary -------------------------------------------------------------
CATEGORY_PHRASE = {"Beauty": "beauty products", "Apparel": "apparel", "Footwear": "footwear",
                   "Electronics": "electronics", "Home": "home products"}

CATEGORY_NAMES = {
    "Beauty": {"beauty", "perfume", "nail", "polish"},
    "Apparel": {"apparel", "top"},
    "Footwear": {"footwear", "heel", "boot"},
    "Electronics": {"electronic", "dryer", "appliance"},
    "Home": {"home", "kitchen", "furniture", "appliance"},
}

# Product nouns (singular). None of these may occur in the knowledge base text.
PRODUCT_NOUNS = {
    "Beauty": {"lipstick", "mascara", "shampoo", "conditioner", "moisturiser", "moisturizer", "lotion", "serum",
               "foundation", "kajal", "eyeliner", "sunscreen", "makeup", "skincare", "cosmetic", "facewash",
               "concealer", "cream"},
    "Apparel": {"kurta", "kurti", "saree", "shirt", "tshirt", "jeans", "dress", "jacket", "hoodie", "skirt",
                "trouser", "legging", "clothes", "clothing", "lehenga"},
    "Footwear": {"shoe", "sandal", "sneaker", "slipper", "loafer", "flipflop"},
    "Electronics": {"earphone", "earbud", "headphone", "trimmer", "charger", "laptop", "speaker", "smartwatch",
                    "gadget", "powerbank", "hairdryer", "straightener"},
    "Home": {"cushion", "curtain", "bedsheet", "lamp", "mug", "vase", "towel", "pillow", "mattress",
             "cookware", "kitchenware", "decor"},
}

CATEGORY_TERMS = {cat: CATEGORY_NAMES[cat] | PRODUCT_NOUNS[cat] for cat in CATEGORY_NAMES}
_NOUN_TO_CATEGORY = {noun: cat for cat, nouns in PRODUCT_NOUNS.items() for noun in nouns}
_WORD_RE = re.compile(r"[A-Za-z]+")


def _norm(word: str) -> str:
    return word[:-1] if word.endswith("s") and len(word) > 3 else word


def categories_in(text: str) -> set[str]:
    words = {_norm(w) for w in re.findall(r"[a-z]+", text.lower())}
    return {cat for cat, terms in CATEGORY_TERMS.items() if words & terms}


def category_compatible(question: str, sentence: str) -> bool:
    """True unless the sentence is only about categories the customer did not ask about."""
    q_cats, s_cats = categories_in(question), categories_in(sentence)
    return not q_cats or not s_cats or bool(q_cats & s_cats)


def normalize_products(text: str) -> str:
    """Replace product nouns by their category phrase for retrieval ("my earphones" -> "my electronics")."""
    def swap(m: re.Match) -> str:
        cat = _NOUN_TO_CATEGORY.get(_norm(m.group(0).lower()))
        return CATEGORY_PHRASE[cat] if cat else m.group(0)
    return _WORD_RE.sub(swap, text)


# Customers say "remote areas / villages"; the delivery policy says "remote pin codes". The embedding model
# does not link them (measured: the remote sentence is not the top hit), so for retrieval only the phrase
# is rewritten to the policy's own wording.
_REMOTE_RE = re.compile(r"\bremote\s+(?:areas?|locations?|places?|regions?|villages?|towns?|pincodes?)\b"
                        r"|\b(?:villages?|rural\s+areas?)\b", re.I)


_COD_RE = re.compile(r"\bCOD\b")  # the abbreviation is unknown to the embedding model (measured: matches payment-failure text)


def normalize_phrases(text: str) -> str:
    """Rewrite customer wording to the policy's own wording, for retrieval only."""
    return _COD_RE.sub("cash on delivery", _REMOTE_RE.sub("remote pin codes", text))


def find_product_phrase(text: str) -> str | None:
    """First category word or product noun in the text, as typed (e.g. 'earphones', 'electronics')."""
    for w in _WORD_RE.findall(text):
        n = _norm(w.lower())
        if n in _NOUN_TO_CATEGORY or any(n in names for names in CATEGORY_NAMES.values()):
            return w.lower()
    return None


# ---- references and anchors ----------------------------------------------------------------------------
# A real reference: it/they/them, "that/this" ending the question ("Can I cancel that?") or right after an
# auxiliary verb ("Is that covered?"). A relative "that" ("a case that is older than 72 hours") is NOT one.
DANGLING_RE = re.compile(
    r"\b(?:it|they|them)\b|\b(?:that|this)\s*[?.!]*\s*$"
    r"|^\s*(?:is|are|was|can|could|does|do|will|would)\s+(?:that|this)\b", re.I)
_TITLE_STOP = {"customer", "support", "policy", "process", "terms", "product", "category", "matrix", "window"}
TOPIC_PREFIXES = {w[:5] for d in KB_DOCUMENTS for w in re.findall(r"[a-z]+", d["title"].lower())
                  if len(w) >= 5 and w not in _TITLE_STOP}


def has_topic(text: str) -> bool:
    """True if the text mentions a policy topic from the knowledge base titles (return, refund, cancel ...)."""
    return any(len(w) >= 5 and w[:5] in TOPIC_PREFIXES for w in re.findall(r"[a-z]+", text.lower()))


def has_anchor(text: str) -> bool:
    """A product/category word or a policy topic word: something the question is actually about."""
    return bool(categories_in(text)) or has_topic(text)


def resolve_reference(text: str, antecedent: str | None) -> str:
    """Replace the first dangling pronoun by 'my <antecedent>' when there is something to point at."""
    if antecedent and DANGLING_RE.search(text) and not categories_in(text):
        return DANGLING_RE.sub(f"my {antecedent}", text, count=1)
    return text


# ---- topic hints and answer type ------------------------------------------------------------------------
TOPIC_HINTS = {"escalation_matrix": re.compile(
    r"\b(?:contact|escalat\w*|complain\w*|manager|supervisor|speak to|talk to)\b", re.I)}
WHO_RE = re.compile(r"\b(?:who|whom)\b", re.I)
ROLE_TERMS = {"agent", "lead", "manager", "team", "support", "level", "human", "supervisor"}


def topic_hint(text: str) -> str | None:
    return next((doc for doc, rx in TOPIC_HINTS.items() if rx.search(text)), None)


def is_who_question(text: str) -> bool:
    return bool(WHO_RE.search(text))


def names_a_role(sentence: str) -> bool:
    return bool(ROLE_TERMS & set(re.findall(r"[a-z]+", sentence.lower())))


# A "how long / how many days / how soon" question asks for a DURATION, so a sentence that states one
# ("5 to 7 working days", "24 hours", "12 months") answers it better than a general sentence that only
# shares topic words ("Refunds ... are paid to the customer's bank account"). Used as a score bonus, not
# a filter, so a duration question with no duration sentence in the policy still gets its best match.
DURATION_QUESTION_RE = re.compile(
    r"\bhow\s+(?:long|soon|quickly|many\s+(?:working\s+|business\s+)?(?:days?|hours?|weeks?|months?))\b", re.I)
DURATION_RE = re.compile(
    r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve|twenty[- ]four)(?:\s*(?:to|-)\s*\d+)?"
    r"\s+(?:working\s+|business\s+)?(?:minutes?|hours?|days?|weeks?|months?)\b", re.I)


# "When will my refund reach my bank account?" also asks for a duration. Only a "when will ..." followed by an
# arrival verb counts: "When can I cancel?" is a rule question, not a duration, so it is deliberately excluded.
WHEN_ARRIVES_RE = re.compile(
    r"\bwhen\s+(?:will|would|should)\b.*\b(?:reach|arrive|credited|received|refunded|delivered|processed)\b", re.I)


def is_duration_question(text: str) -> bool:
    return bool(DURATION_QUESTION_RE.search(text) or WHEN_ARRIVES_RE.search(text))


def has_duration(sentence: str) -> bool:
    return bool(DURATION_RE.search(sentence))


# ---- ambiguous appliances --------------------------------------------------------------------------------
# The policy sets different terms for Electronics and for Home appliances but never says which one a hair dryer,
# kettle or microwave is. Answering with one category would be a guess, so for these the answer states both
# written rules and asks the customer to confirm the category.
AMBIGUOUS_APPLIANCE_RE = re.compile(
    r"\b(?:hair\s*dryer|mixer|grinder|blender|microwave|toaster|kettle|juicer|air\s*fryer|oven|"
    r"vacuum\s+cleaner|water\s+heater|room\s+heater|heater|cooler|fan)s?\b", re.I)
_EXPLICIT_CATEGORY_RE = re.compile(r"\b(?:electronics?|home|appliances?)\b", re.I)
CATEGORY_DIFFERS_TOPICS = {  # knowledge-base document whose rule differs by category -> question wording
    "warranty_terms": re.compile(r"\b(?:warranty|warranties|guarantee)\b", re.I),
    "return_window": re.compile(r"\b(?:return\w*|exchange\w*)\b", re.I),
}


def ambiguous_appliance_doc(text: str) -> str | None:
    """Doc id to answer from if the question names an appliance with no stated category, else None."""
    if not AMBIGUOUS_APPLIANCE_RE.search(text) or _EXPLICIT_CATEGORY_RE.search(text):
        return None
    return next((doc for doc, rx in CATEGORY_DIFFERS_TOPICS.items() if rx.search(text)), None)


# ---- order words -----------------------------------------------------------------------------------------
# Words that only talk about an order ("What is the status of order ...") and carry no policy topic.
ORDER_WORDS = tokens("order status track tracking shipment package parcel delivered update")


def is_order_only(text: str) -> bool:
    """True if the text has no content words besides order words (nothing to look up in the policy KB)."""
    return not (tokens(text) - ORDER_WORDS)


# ---- sub-queries -----------------------------------------------------------------------------------------
_SPLIT_RE = re.compile(r"\b(?:and|also|plus)\b|[,;?]", re.I)
_QUESTION_STARTERS = {"how", "what", "when", "where", "who", "which", "why", "can", "could", "do", "does", "did",
                      "will", "would", "should", "is", "are", "am", "tell", "show", "explain", "give", "get",
                      "list", "find", "check", "please"}


def split_subqueries(question: str) -> list[str]:
    """Split a multi-topic question into parts, one per question.

    Statement fragments ("My parcel is very late") are not questions: they are context for the whole
    question and are not searched on their own when the question splits into several parts.
    If the question does not split into 2+ parts, the original question is returned.
    """
    fragments = [p.strip() for p in _SPLIT_RE.split(question) if len(p.split()) >= 2]
    parts = [f for f in fragments if f.split()[0].lower() in _QUESTION_STARTERS]
    return parts if len(parts) > 1 else [question]
