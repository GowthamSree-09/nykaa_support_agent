"""CrewAI tool wrappers. Each tool declares its argument schema (Pydantic) - the mock LLM
reads that schema to decide what to pass, instead of guessing from the tool's name."""
import json
from typing import Type

import llm  # noqa: F401  (turns telemetry off before crewai is imported)
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from agents.order_tool import check_order_status
from governance.cache import cached_answer

TOOL_LOG: list[dict] = []  # audit trail: which tool ran with which arguments


class RagInput(BaseModel):
    query: str = Field(..., description="The customer's policy question")


class OrderInput(BaseModel):
    record_id: str = Field(..., description="Order id such as ORD-1023")


class RagLookupTool(BaseTool):
    name: str = "rag_lookup"
    description: str = "Search the support policy knowledge base and return a grounded answer as JSON."
    args_schema: Type[BaseModel] = RagInput

    def _run(self, query: str) -> str:
        TOOL_LOG.append({"tool": self.name, "args": {"query": query}})
        r = cached_answer(query)
        return json.dumps({"kind": "policy", "answer": r["answer"], "grounded": r["grounded"],
                           "sources": r["sources"], "top_similarity": r["top_similarity"],
                           "context": [c["text"] for c in r["chunks"] if c["similarity"] >= 0.32]})


class CheckOrderStatusTool(BaseTool):
    name: str = "check_order_status"
    description: str = "Look up one order by id: status, value and escalation score, as JSON."
    args_schema: Type[BaseModel] = OrderInput

    def _run(self, record_id: str) -> str:
        TOOL_LOG.append({"tool": self.name, "args": {"record_id": record_id}})
        return json.dumps({"kind": "order", **check_order_status(record_id)})
