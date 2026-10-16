"""MOCK_LLM for AutoGen: a deterministic ChatCompletionClient (no network, no key).

Two behaviours, chosen by the agent's own system message:
  * Policy-Compliance-Reviewer: reads "DRAFT:" and "CONTEXT:" from the task, checks every draft
    sentence against the context and replies with a JSON list of unsupported sentences.
  * Final-Editor: reads the reviewer's reply and returns a Verdict JSON. If nothing was
    unsupported it approves the draft unchanged; otherwise it removes the unsupported sentences.
"""
import json
from typing import Any, AsyncGenerator, Mapping, Optional, Sequence, Union

import llm  # noqa: F401
from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient, CreateResult, ModelFamily, RequestUsage

from agents.guardrails import _sentences, groundedness_check
from rag.generate import FALLBACK

_USAGE = RequestUsage(prompt_tokens=0, completion_tokens=0)


def _text(m) -> str:
    c = getattr(m, "content", "")
    return c if isinstance(c, str) else str(c)


def split_task(task_text: str) -> tuple[str, list[str]]:
    draft = task_text.split("DRAFT:", 1)[1].split("CONTEXT:", 1)[0].strip()
    context = [c.strip() for c in task_text.split("CONTEXT:", 1)[1].split("\n") if c.strip()]
    return draft, context


class MockAutogenClient(ChatCompletionClient):
    def __init__(self):
        self.calls = 0

    @property
    def model_info(self):
        return {"vision": False, "function_calling": False, "json_output": True,
                "family": ModelFamily.UNKNOWN, "structured_output": True}

    @property
    def capabilities(self):  # deprecated alias
        return self.model_info

    async def create(self, messages: Sequence[Any], *, tools=[], tool_choice="auto", json_output=None,
                     extra_create_args: Mapping[str, Any] = {},
                     cancellation_token: Optional[CancellationToken] = None) -> CreateResult:
        self.calls += 1
        system = _text(messages[0])
        body = [_text(m) for m in messages[1:]]
        task = next(t for t in body if "DRAFT:" in t and "CONTEXT:" in t)
        draft, context = split_task(task)

        if "Final-Editor" in system:
            reviewer_reply = next((t for t in reversed(body) if "UNSUPPORTED:" in t), "UNSUPPORTED: []")
            bad = json.loads(reviewer_reply.split("UNSUPPORTED:", 1)[1].strip().splitlines()[0])
            if not bad:
                verdict = {"approved": True, "final_answer": draft,
                           "reason": "Every sentence of the draft is supported by the retrieved context."}
            else:
                kept = [s for s in _sentences(draft) if s not in bad]
                verdict = {"approved": False, "final_answer": " ".join(kept) if kept else FALLBACK,
                           "reason": f"Removed {len(bad)} unsupported claim(s): " + " | ".join(bad)}
            content = json.dumps(verdict)
        else:  # Policy-Compliance-Reviewer
            _, bad = groundedness_check(draft, context)
            content = "Compliance review of the draft.\nUNSUPPORTED: " + json.dumps(bad)
        return CreateResult(finish_reason="stop", content=content, usage=_USAGE, cached=False)

    async def create_stream(self, messages, *, tools=[], tool_choice="auto", json_output=None,
                            extra_create_args={}, cancellation_token=None
                            ) -> AsyncGenerator[Union[str, CreateResult], None]:
        yield await self.create(messages, json_output=json_output)

    async def close(self) -> None:
        return None

    def actual_usage(self) -> RequestUsage:
        return _USAGE

    def total_usage(self) -> RequestUsage:
        return _USAGE

    def count_tokens(self, messages, *, tools=[]) -> int:
        return sum(len(_text(m)) // 4 for m in messages)

    def remaining_tokens(self, messages, *, tools=[]) -> int:
        return 8000 - self.count_tokens(messages)
