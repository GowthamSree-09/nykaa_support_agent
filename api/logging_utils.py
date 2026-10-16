"""Task 12 - JSON-Lines request log (one line per request, trace id + timing).

The question is masked with the SAME mask_pii used on the model's input, so a phone number or
card last-4 never reaches disk. (Names/addresses are free text and out of scope for masking.)
"""
import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone

from agents.guardrails import mask_pii

LOG_PATH = os.environ.get("REQUEST_LOG", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                      "logs", "requests.jsonl"))
_lock = threading.Lock()


def new_trace_id() -> str:
    return uuid.uuid4().hex


def log_request(trace_id: str, endpoint: str, text: str, started: float, **fields) -> dict:
    safe_text, _ = mask_pii(text)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trace_id": trace_id,
        "endpoint": endpoint,
        "request_text": safe_text,
        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        **fields,
    }
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with _lock, open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry
