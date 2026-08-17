from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from app.agents.state import AgentState
from app.tools.registry import ALL_TOOLS
from app.agents.nodes import (
    supervisor_node, 
    specialist_node, 
    reviewer_node, 
    escalation_node,
    memorize_node
)

def route_after_supervisor(state: AgentState) -> str:
    plan = state["plan"]
    if plan.requires_human_approval or plan.confidence_score < 0.7:
        return "escalation_node"
    return "specialist_node"

def route_after_specialist(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "reviewer_node"

def route_after_reviewer(state: AgentState) -> str:
    if state.get("reviewer_feedback"):
        return "specialist_node"
        
    plan = state["plan"]
    if state["current_task_index"] >= len(plan.subtasks):
        # ALL TASKS DONE -> Route to Memory
        return "memorize_node"
        
    return "specialist_node"

workflow = StateGraph(AgentState)

workflow.add_node("supervisor_node", supervisor_node)
workflow.add_node("specialist_node", specialist_node)
workflow.add_node("tools", ToolNode(ALL_TOOLS))
workflow.add_node("reviewer_node", reviewer_node)
workflow.add_node("escalation_node", escalation_node)
workflow.add_node("memorize_node", memorize_node)

workflow.add_edge(START, "supervisor_node")

workflow.add_conditional_edges(
    "supervisor_node",
    route_after_supervisor,
    {"escalation_node": "escalation_node", "specialist_node": "specialist_node"}
)

workflow.add_edge("escalation_node", "specialist_node")

workflow.add_conditional_edges(
    "specialist_node",
    route_after_specialist,
    {"tools": "tools", "reviewer_node": "reviewer_node"}
)

workflow.add_edge("tools", "specialist_node")

workflow.add_conditional_edges(
    "reviewer_node",
    route_after_reviewer,
    {"specialist_node": "specialist_node", "memorize_node": "memorize_node"}
)

# After saving to memory, the graph finally ends
workflow.add_edge("memorize_node", END)

app_graph = workflow.compile(interrupt_before=["escalation_node"])
