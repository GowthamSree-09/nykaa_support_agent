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


def groundedness_check(answer: str, support_texts: list[str], min_support: float = 0.6) -> tuple[bool, list[str]]:
    """Each answer sentence must have >= min_support of its content words present in the support text.
    Returns (all_supported, unsupported_sentences)."""
    support = set().union(*(tokens(t) for t in support_texts)) if support_texts else set()
    unsupported = []
    for s in _sentences(answer):
        words = tokens(s)
        if words and len(words & support) / len(words) < min_support:
            unsupported.append(s)
    return (not unsupported, unsupported)
