"""Task 14 - AutoGen review stage: Policy-Compliance-Reviewer then Final-Editor, bounded to 2 turns."""
import asyncio

import llm  # noqa: F401
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from pydantic import BaseModel

from llm.autogen_mock import MockAutogenClient


class Verdict(BaseModel):
    approved: bool
    final_answer: str
    reason: str


def build_team() -> RoundRobinGroupChat:
    client = MockAutogenClient()
    reviewer = AssistantAgent(
        "Policy_Compliance_Reviewer", model_client=client,
        system_message="You are the Policy-Compliance-Reviewer. Flag draft sentences that are ungrounded, "
                       "irrelevant to the question, unsafe, or leave the question unanswered.")
    editor = AssistantAgent(
        "Final_Editor", model_client=client, output_content_type=Verdict,
        system_message="You are the Final-Editor. Approve the draft unchanged, or remove unsupported claims.")
    return RoundRobinGroupChat(
        [reviewer, editor],
        max_turns=2,  # reviewer speaks once, editor speaks once
        custom_message_types=[StructuredMessage[Verdict]],  # required for the editor's structured output
    )


async def review_draft_async(draft: str, context: list[str], question: str = "") -> Verdict:
    """The team receives the customer's question, the Composer's draft and the retrieved context."""
    task = ("Review this draft answer.\nQUESTION:\n" + question + "\n\nDRAFT:\n" + draft
            + "\n\nCONTEXT:\n" + "\n".join(context))
    result = await build_team().run(task=task)
    final = result.messages[-1]
    assert isinstance(final, StructuredMessage), "Final-Editor must reply with a structured message"
    return Verdict.model_validate(final.content.model_dump())


def review_draft(draft: str, context: list[str], question: str = "") -> Verdict:
    """Sync wrapper (safe when no event loop is running, e.g. in scripts or FastAPI sync endpoints)."""
    return asyncio.run(review_draft_async(draft, context, question))
