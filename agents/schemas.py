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
