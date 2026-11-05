"""Fresh, previously unseen probe questions (round 4). Written BEFORE the round-4 fixes and first run
against the unfixed code; results of that first run are in transcripts/unseen_probe_initial.txt.

Each case: (question, regex the answer MUST match, regex the answer MUST NOT match or None).
Rule-based and deterministic; it checks facts in the written knowledge base, nothing more.
Run: python -m eval.unseen_probe
"""
import re

import llm  # noqa: F401  (telemetry off before crewai import)

from agents.service import core_pipeline

CASES = [
    ("Do you ship to remote villages and how many days will that take?", r"10 working days", r"returned within"),
    ("What is the delivery time for a remote pin code?", r"10 working days", None),
    ("How many days does a metro city delivery take?", r"2 to 4 working days", r"5 to 8"),
    ("When will the money for my COD return be credited to my bank?", r"5 to 7 working days", None),
    ("How long is the warranty on my kettle?", r"(?i)electronics|home appliance", None),
    ("Is there a warranty on a microwave oven?", r"(?i)6 month|12 month", None),
    ("What is the warranty on lipstick?", r"(?i)no warranty", r"12 month"),
    ("Can I exchange sandals for a bigger size?", r"(?i)size|exchange", r"warranty"),
    ("Can I cancel an order that is already shipped?", r"(?i)cancel", None),
    ("How long do I have to report a broken item?", r"48 hours", None),
    ("How many loyalty points equal one rupee?", r"(?i)1 loyalty point|1 INR", None),
    ("Is a payment retry allowed after my card failed?", r"(?i)retry|attempt", None),
    ("What is the status of order ORD-9999?", r"could not find an order", r"cancel"),
    ("Who is the prime minister of India?", r"don't know", None),
]


def main() -> int:
    bad = 0
    for q, must, mustnot in CASES:
        ans = core_pipeline(q)["answer"]
        ok = bool(re.search(must, ans)) and not (mustnot and re.search(mustnot, ans))
        bad += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {q}\n       -> {ans}")
    print(f"\n{len(CASES) - bad}/{len(CASES)} passed")
    return bad


if __name__ == "__main__":
    raise SystemExit(main())
