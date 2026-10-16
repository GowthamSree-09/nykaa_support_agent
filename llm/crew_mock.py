"""MOCK_LLM for CrewAI: a BaseLLM subclass that follows CrewAI's ReAct text protocol.

Real LLM vs mock: a real model reads the prompt and *decides* what to do. This mock follows
fixed rules, so every run is identical and needs no network:
  1. If the agent has a tool, first reply with  Thought/Action/Action Input  (a tool call).
  2. When CrewAI runs the tool it appends "...Observation: <result>" to our own assistant message.
     We read ONLY assistant messages (never the system prompt, which contains the template
     text "Observation: the result of the action" - pitfall #1 in the brief).
  3. Then reply with  Final Answer: <observation>.
  4. An agent with no tools (the Composer) merges the earlier agents' JSON into a SupportResponse.
Tool arguments are chosen from the tool's declared args_schema field names (pitfall #2).
"""
import json
import re
from typing import Any

import llm  # noqa: F401  (telemetry off)
from crewai.llms.base_llm import BaseLLM

from agents.routing import classify_intent, needs_policy, policy_query
from agents.schemas import (ASK_ORDER_ID, CLARIFY_QUESTION, PARTIAL_ANSWER, SupportResponse, not_found_sentence,
                            order_sentences)
from rag.generate import FALLBACK

ORDER_ID_RE = re.compile(r"ORD-\d+", re.I)


def _text(messages) -> list[dict]:
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    return [{"role": m.get("role", "user"), "content": str(m.get("content", ""))} for m in messages]


def _question(msgs: list[dict]) -> str:
    """Pull the customer question out of the task prompt (the user message)."""
    for m in msgs:
        if m["role"] == "user" and "Customer question:" in m["content"]:
            after = m["content"].split("Customer question:", 1)[1]
            return after.split("\n", 1)[0].strip()
    return ""


def _json_objects(text: str) -> list[dict]:
    dec, found, i = json.JSONDecoder(), [], 0
    while (i := text.find("{", i)) != -1:
        try:
            obj, end = dec.raw_decode(text[i:])
            if isinstance(obj, dict):
                found.append(obj)
            i += end
        except ValueError:
            i += 1
    return found


def compose(context_text: str, question: str) -> SupportResponse:
    """Merge the earlier agents' JSON into one answer, guided by what the customer actually asked."""
    intent = classify_intent(question)
    objs = _json_objects(context_text)
    policy = next((o for o in objs if o.get("kind") == "policy" and not o.get("skipped")), None)
    order_objs = [o for o in objs if o.get("kind") == "order"]
    order = next((o for o in order_objs if o.get("found")), None)
    missing = next((o for o in order_objs if o.get("found") is False and o.get("record_id")), None)

    parts, sources, grounded = [], [], True
    if order:
        parts.extend(order_sentences(order["record_id"], order["status"], order["order_value_inr"],
                                     order["escalation_score"], order["recommend_escalation"]))
    elif missing:
        parts.append(not_found_sentence(missing["record_id"]))
    elif intent == "order_id_missing":
        parts.append(ASK_ORDER_ID)
    elif intent == "clarify":
        parts.append(CLARIFY_QUESTION)  # ambiguous: ask instead of guessing

    if intent in ("policy", "mixed"):
        if policy and policy["grounded"]:
            parts.append(policy["answer"])
            sources = policy["sources"]
            if policy.get("unanswered"):  # part of the question had no supporting policy text
                parts.append(PARTIAL_ANSWER)
        elif parts:  # an order part exists but nothing supports the policy part
            parts.append(PARTIAL_ANSWER)
        else:
            parts.append(FALLBACK)
            grounded = False
    return SupportResponse(
        answer=" ".join(parts), grounded=grounded, sources=sources,
        order_id=order["record_id"] if order else None,
        order_status=order["status"] if order else None,
        order_value_inr=order["order_value_inr"] if order else None,
        escalation_score=order["escalation_score"] if order else None,
        escalate=bool(order and order["recommend_escalation"]),
    )


class MockCrewLLM(BaseLLM):
    llm_type: str = "mock"
    model: str = "mock-llm"
    is_litellm: bool = False
    call_count: int = 0

    def supports_function_calling(self) -> bool:
        return False  # force CrewAI's ReAct text loop

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 8000

    def call(self, messages, tools=None, callbacks=None, available_functions=None,
             from_task=None, from_agent=None, response_model=None) -> Any:
        self.call_count += 1
        msgs = _text(messages)
        # Step 3: a tool result came back. CrewAI stores it inside the latest ASSISTANT message
        # (our own earlier output + "Observation: <result>"), followed by a user nudge. We read only
        # assistant messages, never the system/user template text.
        own = [m for m in msgs if m["role"] == "assistant"]
        if own and "Observation:" in own[-1]["content"]:
            observation = own[-1]["content"].rsplit("Observation:", 1)[1].strip()
            return f"Thought: I now know the final answer\nFinal Answer: {observation}"

        agent_tools = list(getattr(from_agent, "tools", None) or [])
        question = _question(msgs)
        if agent_tools:  # Step 1: choose a tool call from its declared argument schema
            tool = agent_tools[0]
            fields = set(tool.args_schema.model_fields)
            if "record_id" in fields:
                m = ORDER_ID_RE.search(question)
                if not m:
                    return ("Thought: no order id in the question, so no lookup is needed\n"
                            "Final Answer: {\"kind\": \"order\", \"found\": false, \"note\": \"no order id given\"}")
                args = {"record_id": m.group(0).upper()}
            elif "query" in fields:
                if not needs_policy(question):  # order-only question: nothing to look up in the policy KB
                    return ("Thought: this is an order-only question, no policy search is needed\n"
                            "Final Answer: {\"kind\": \"policy\", \"skipped\": true, \"grounded\": false, "
                            "\"answer\": \"\", \"sources\": []}")
                args = {"query": policy_query(question)}
            else:
                return "Thought: I cannot use my tool for this\nFinal Answer: {}"
            return (f"Thought: I should use my tool\nAction: {tool.name}\n"
                    f"Action Input: {json.dumps(args)}")

        # Step 4: Composer (no tools) merges the earlier agents' outputs.
        context_text = "\n".join(m["content"] for m in msgs if m["role"] == "user")
        return compose(context_text, question).model_dump_json()
