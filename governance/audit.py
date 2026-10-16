"""Governance audit trail: one JSON line per governance decision (budget rejected, access denied, ...).

Only decision metadata is stored - never the customer's text - so no PII can reach this file.
"""
import json
import os
import threading
from datetime import datetime, timezone

AUDIT_PATH = os.environ.get("GOVERNANCE_LOG", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                           "logs", "governance.jsonl"))
_lock = threading.Lock()
EVENTS: list[dict] = []  # in-memory copy, handy for tests and demos


def audit(event: str, **fields) -> dict:
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
    EVENTS.append(entry)
    os.makedirs(os.path.dirname(AUDIT_PATH), exist_ok=True)
    with _lock, open(AUDIT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry
