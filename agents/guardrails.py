"""Task 10 - guardrails.

Input side : mask_pii (phone + card last-4) and detect_injection.
Output side: groundedness_check - every sentence of an answer must be supported by the retrieved
             context (or by the order record); otherwise we refuse instead of answering.
Names and addresses are free text and are intentionally NOT masked (out of scope per the brief).
"""
import re

from llm.mock_llm import tokens

# Indian mobile numbers: optional +91 / 0, then 10 digits starting 6-9 (spaces or dashes allowed inside).
PHONE_RE = re.compile(r"(?<![\d-])(?:\+?91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")
# Card last 4: "card ending 4242", "ending in 4242", "last 4 digits 4242", "****4242", "XXXX-4242",
# plus a full 13-16 digit card number (only its last 4 are kept out of the model's sight too).
CARD_RES = [
    re.compile(r"(?i)\b((?:card|visa|mastercard|debit|credit)[^.\n]{0,25}?(?:ending(?: in| with)?|last\s*4(?:\s*digits)?(?: are| is)?|\bno\.?)\s*)(\d{4})\b"),
    re.compile(r"(?i)\b(ending(?: in| with)?\s+)(\d{4})\b"),
    re.compile(r"([*xX]{2,}[\s-]*)(\d{4})\b"),
]
FULL_CARD_RE = re.compile(r"\b(?:\d[ -]?){12,15}\d\b")

INJECTION_PATTERNS = [
    r"ignore (?:all |any |the )?(?:previous|prior|above|earlier) (?:instructions|rules|prompts?)",
    r"disregard (?:all |any |the )?(?:previous|prior|above|your) (?:instructions|rules|prompts?)",
    r"(?:reveal|show|print|repeat|leak) (?:me )?(?:your|the) (?:system )?(?:prompt|instructions|rules)",
    r"you are now (?:an?|the) ",
    r"\bact as (?:an? )?(?:unrestricted|jailbroken|dan|developer)",
    r"developer mode|jailbreak|do anything now",
    r"override (?:your|the|all) (?:safety|rules|instructions|guardrails)",
    r"forget (?:everything|all|your) (?:you|instructions|rules)",
]
INJECTION_RES = [re.compile(p, re.I) for p in INJECTION_PATTERNS]


def mask_pii(text: str) -> tuple[str, list[str]]:
    """Return (masked_text, list_of_field_types_found)."""
    found = []
    # Cards first so a 16-digit card number is not mistaken for a phone number.
    def full_card(m):
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 16:
            found.append("card")
            return "[CARD_MASKED]"
        return m.group(0)
    text = FULL_CARD_RE.sub(full_card, text)
    for rx in CARD_RES:
        def sub(m):
            found.append("card")
            return f"{m.group(1)}[CARD_LAST4_MASKED]"
        text = rx.sub(sub, text)

    def phone(m):
        found.append("phone")
        return "[PHONE_MASKED]"
    text = PHONE_RE.sub(phone, text)
    return text, sorted(set(found))


def detect_injection(text: str) -> list[str]:
    return [rx.pattern for rx in INJECTION_RES if rx.search(text)]


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


# ---- groundedness -------------------------------------------------------------------------------
# A sentence is "supported" only if ONE context sentence (a) covers at least `min_support` of its content
# words AND (b) agrees with it on numbers, (c) on negation, (d) on contradictory word pairs, and
# (e) contains every risky absolute the answer uses ("guaranteed", "tomorrow" ...).
# Limitations: this is deterministic text matching, not understanding. It misses subtle contradictions
# that use words outside the lists below, and it can reject a correct answer that merges two context
# sentences into one. Real deployments would add an LLM/NLI judge on top.
NEGATION_RE = re.compile(r"\b(?:not|no|never|cannot|nor|without|none)\b|n't\b", re.I)
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
NUMBER_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
                "eight": "8", "nine": "9", "ten": "10", "twelve": "12", "fifteen": "15", "thirty": "30"}
ANTONYMS = [("broken", "unbroken"), ("worn", "unworn"), ("allowed", "prohibited"), ("before", "after"),
            ("increase", "decrease"), ("refundable", "nonrefundable"), ("opened", "sealed"),
            ("approve", "reject"), ("accept", "refuse"), ("free", "paid")]
RISKY_TERMS = {"guarantee", "guaranteed", "always", "never", "instant", "instantly", "immediately", "tomorrow",
               "today", "lifetime", "unlimited", "free", "bonus", "insurance", "cashback", "refundable"}
MIN_SUPPORT = 0.6


def _numbers(text: str) -> set[str]:
    words = [NUMBER_WORDS.get(w, w) for w in re.findall(r"[a-z0-9.]+", text.lower())]
    return set(NUMBER_RE.findall(" ".join(words)))


def _plain_words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def sentence_problem(sentence: str, support_sentences: list[str], min_support: float = MIN_SUPPORT) -> str | None:
    """None if supported, otherwise a short reason."""
    words = tokens(sentence)
    if not words:
        return None
    reasons = []
    for t in support_sentences:
        coverage = len(words & tokens(t)) / len(words)
        if coverage < min_support:
            continue
        if not _numbers(sentence) <= _numbers(t):
            reasons.append("number not in context"); continue
        if bool(NEGATION_RE.search(sentence)) != bool(NEGATION_RE.search(t)):
            reasons.append("negation differs from context"); continue
        sw, tw = _plain_words(sentence), _plain_words(t)
        if any((a in sw and b in tw and a not in tw) or (b in sw and a in tw and b not in tw) for a, b in ANTONYMS):
            reasons.append("contradicts context"); continue
        if (RISKY_TERMS & sw) - tw:
            reasons.append("unsupported absolute claim"); continue
        return None
    return reasons[0] if reasons else "not found in context"


def groundedness_problems(answer: str, support_texts: list[str]) -> list[tuple[str, str]]:
    support_sentences = [s for t in support_texts for s in _sentences(t)]
    return [(s, r) for s in _sentences(answer) if (r := sentence_problem(s, support_sentences))]


def groundedness_check(answer: str, support_texts: list[str], min_support: float = MIN_SUPPORT) -> tuple[bool, list[str]]:
    """Returns (all_supported, unsupported_sentences)."""
    problems = groundedness_problems(answer, support_texts)
    return (not problems, [s for s, _ in problems])
