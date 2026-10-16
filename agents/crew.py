"""Task 7 - the CrewAI crew: Retrieval Agent, Order Lookup Agent, Response Composer."""
import llm  # noqa: F401  (telemetry off before crewai import)
from crewai import Agent, Crew, Process, Task

from agents.schemas import SupportResponse
from agents.tools import CheckOrderStatusTool, RagLookupTool
from governance.access import LOOKUP_AGENT_ROLE, assert_tool_allowed
from llm.crew_mock import MockCrewLLM

RETRIEVAL_ROLE = "Policy Retrieval Agent"
COMPOSER_ROLE = "Response Composer"

_llm = MockCrewLLM(model="mock-llm")  # swap for a real LLM here if you ever set an API key (optional, not needed)


def make_agent(role: str, goal: str, backstory: str, tools: list) -> Agent:
    for t in tools:
        assert_tool_allowed(role, t.name)  # least autonomy: refuse before the agent exists
    return Agent(role=role, goal=goal, backstory=backstory, tools=tools, llm=_llm,
                 allow_delegation=False, verbose=False, max_iter=4)


def build_crew(question: str) -> Crew:
    retrieval = make_agent(RETRIEVAL_ROLE, "Find policy facts that answer the customer question",
                           "You search the knowledge base and never invent policy.", [RagLookupTool()])
    lookup = make_agent(LOOKUP_AGENT_ROLE, "Look up the customer's order when an order id is given",
                        "You are the only agent allowed to read order records.", [CheckOrderStatusTool()])
    composer = make_agent(COMPOSER_ROLE, "Write one clear, grounded draft answer",
                          "You combine policy facts and order details; you never add new facts.", [])

    t_policy = Task(description=f"Find the relevant policy.\nCustomer question: {question}",
                    expected_output="JSON with the grounded policy answer", agent=retrieval)
    t_order = Task(description=f"Look up the order if the question contains an order id.\n"
                               f"Customer question: {question}",
                   expected_output="JSON with the order status and escalation score", agent=lookup)
    t_compose = Task(description=f"Combine the policy answer and order details into one draft.\n"
                                 f"Customer question: {question}",
                     expected_output="A SupportResponse JSON object", agent=composer,
                     context=[t_policy, t_order], output_pydantic=SupportResponse)
    return Crew(agents=[retrieval, lookup, composer], tasks=[t_policy, t_order, t_compose],
                process=Process.sequential, verbose=False)


def run_crew(question: str) -> SupportResponse:
    result = build_crew(question).kickoff()
    if result.pydantic is None:
        raise ValueError("Crew output did not match SupportResponse")
    return SupportResponse.model_validate(result.pydantic.model_dump())  # validate in code, as required
