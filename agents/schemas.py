"""Task 9 - the structured output every crew response must follow."""
from pydantic import BaseModel, Field


class SupportResponse(BaseModel):
    answer: str = Field(..., min_length=1, description="Final customer-facing answer")
    grounded: bool = Field(..., description="True if the policy part is supported by retrieved context")
    sources: list[str] = Field(default_factory=list, description="Knowledge-base document ids used")
    order_id: str | None = None
    order_status: str | None = None
    order_value_inr: int | None = None
    escalation_score: float | None = Field(default=None, ge=0, le=1)
    escalate: bool = False


def order_sentences(order_id: str, status: str, value: int, score: float, escalate: bool) -> list[str]:
    """The one place that words order facts, so the Composer and the groundedness guard agree."""
    out = [f"Order {order_id} is currently {status} (order value {value} INR)."]
    if escalate:
        out.append(f"It has a high escalation score ({score}), so we are escalating it to a Level 2 team lead.")
    return out


ASK_ORDER_ID = "Please share your order id (for example ORD-1001) so I can check the status."
PARTIAL_ANSWER = "I could not find a policy answer for the rest of your question, so a human agent can help with that."


CLARIFY_QUESTION = "Could you tell me which item or order you are asking about, so I can find the right policy or contact?"
NOT_FOUND_PREFIX ="I could not find an order with id"


def not_found_sentence(order_id: str) -> str:
    return f"{NOT_FOUND_PREFIX} {order_id}, so please check the order number."
