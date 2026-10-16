"""Task 15 (Application layer) - principle of least autonomy.

Only the Order Lookup Agent may hold `check_order_status`. Every agent in the crew is built
through `make_agent`, which calls `assert_tool_allowed` for each tool it is given. Wiring the
lookup tool to any other role raises PermissionError *before* the agent exists, so it can never
call the tool. (The tool is also simply never passed to the other agents in build_crew.)
"""
LOOKUP_AGENT_ROLE = "Order Lookup Agent"
TOOL_PERMISSIONS = {"check_order_status": {LOOKUP_AGENT_ROLE}}


def assert_tool_allowed(agent_role: str, tool_name: str) -> None:
    allowed = TOOL_PERMISSIONS.get(tool_name)
    if allowed is not None and agent_role not in allowed:
        raise PermissionError(f"Least-autonomy violation: '{agent_role}' may not use tool '{tool_name}'")
