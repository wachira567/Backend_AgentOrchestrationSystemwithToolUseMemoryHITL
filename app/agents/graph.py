from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from app.agents.state import AgentState
from app.tools.registry import ALL_TOOLS
from app.agents.nodes import (
    supervisor_node, 
    specialist_node, 
    reviewer_node, 
    escalation_node
)

# --- Routing Logic ---

def route_after_supervisor(state: AgentState) -> str:
    plan = state["plan"]
    # Trigger human-in-the-loop if confidence is low or task is sensitive
    if plan.requires_human_approval or plan.confidence_score < 0.7:
        return "escalation_node"
    return "specialist_node"

def route_after_specialist(state: AgentState) -> str:
    last_message = state["messages"][-1]
    # If the LLM decided to use a tool, route to the execution node
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    # Otherwise, the specialist has formulated an answer, send to reviewer
    return "reviewer_node"

def route_after_reviewer(state: AgentState) -> str:
    # If the reviewer rejected the output, send back to the specialist to fix
    if state.get("reviewer_feedback"):
        return "specialist_node"
        
    # If all subtasks in the plan are completed, end the workflow
    plan = state["plan"]
    if state["current_task_index"] >= len(plan.subtasks):
        return END
        
    # Move to the next subtask
    return "specialist_node"

# --- Build the Graph ---

workflow = StateGraph(AgentState)

# Add Agent and Tool Nodes
workflow.add_node("supervisor_node", supervisor_node)
workflow.add_node("specialist_node", specialist_node)
workflow.add_node("tools", ToolNode(ALL_TOOLS))
workflow.add_node("reviewer_node", reviewer_node)
workflow.add_node("escalation_node", escalation_node)

# Wire the execution flow
workflow.add_edge(START, "supervisor_node")

# Supervisor delegates, or escalates to human
workflow.add_conditional_edges(
    "supervisor_node",
    route_after_supervisor,
    {
        "escalation_node": "escalation_node",
        "specialist_node": "specialist_node"
    }
)

# Resuming from a human escalation routes directly to the specialist
workflow.add_edge("escalation_node", "specialist_node")

# Specialist either uses a tool or submits work to reviewer
workflow.add_conditional_edges(
    "specialist_node",
    route_after_specialist,
    {
        "tools": "tools",
        "reviewer_node": "reviewer_node"
    }
)

# After a tool executes, flow goes back to the specialist to interpret the result
workflow.add_edge("tools", "specialist_node")

# Reviewer either approves and loops, or rejects and loops
workflow.add_conditional_edges(
    "reviewer_node",
    route_after_reviewer,
    {
        "specialist_node": "specialist_node",
        END: END
    }
)

# Compile the graph
# We declare 'escalation_node' as an interruption point. The graph will pause here.
app_graph = workflow.compile(interrupt_before=["escalation_node"])
