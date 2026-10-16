"""Task 15 (Runtime layer) - per-request token/cost budget cap.

Before the crew runs we ESTIMATE the request's size (about 4 characters = 1 token, plus a fixed
overhead for the agent prompts). If the estimate is above the cap the request is rejected up front,
so it can never silently exceed the budget.
"""
import math

MAX_TOKENS_PER_REQUEST = 1000
PROMPT_OVERHEAD_TOKENS = 600          # fixed cost of the three agents' prompts
COST_PER_1K_TOKENS_USD = 0.0005       # illustrative price used for the cost estimate
MAX_COST_PER_REQUEST_USD = MAX_TOKENS_PER_REQUEST / 1000 * COST_PER_1K_TOKENS_USD


class BudgetExceeded(Exception):
    pass


def estimate_tokens(*texts: str) -> int:
    return PROMPT_OVERHEAD_TOKENS + sum(math.ceil(len(t) / 4) for t in texts)


def estimate_cost_usd(tokens: int) -> float:
    return round(tokens / 1000 * COST_PER_1K_TOKENS_USD, 6)


def enforce_budget(*texts: str) -> dict:
    tokens = estimate_tokens(*texts)
    cost = estimate_cost_usd(tokens)
    if tokens > MAX_TOKENS_PER_REQUEST or cost > MAX_COST_PER_REQUEST_USD:
        raise BudgetExceeded(f"Request rejected: estimated {tokens} tokens (${cost}) exceeds the "
                             f"budget of {MAX_TOKENS_PER_REQUEST} tokens (${MAX_COST_PER_REQUEST_USD}).")
    return {"estimated_tokens": tokens, "estimated_cost_usd": cost}
