"""Importing anything from `llm` first turns CrewAI/OpenTelemetry telemetry OFF (zero outbound calls)."""
import os

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TRACING_ENABLED"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ.setdefault("MOCK_LLM", "true")
