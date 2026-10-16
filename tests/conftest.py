import os

import llm  # noqa: F401  (sets telemetry-off env vars before anything imports crewai)

os.environ.setdefault("REQUEST_LOG", os.path.join(os.path.dirname(__file__), "_test_requests.jsonl"))
