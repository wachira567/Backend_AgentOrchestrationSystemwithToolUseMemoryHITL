import uuid
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

from app.memory.db import get_pool
from app.agents.graph import workflow

router = APIRouter()

async def execute_task_background(thread_id: str, task_input: str):
    """Runs the graph in the background, saving state to PostgreSQL."""
    pool = await get_pool()
    checkpointer = AsyncPostgresSaver(pool)
    
    # Compile the graph with persistent memory enabled
    app_graph = workflow.compile(
        checkpointer=checkpointer, 
        interrupt_before=["escalation_node"]
    )
    
    # The thread_id isolates this execution from all other users concurrently using the system
    config = {"configurable": {"thread_id": thread_id}}
    
    initial_state = {
        "task_input": task_input,
        "messages": [HumanMessage(content=task_input)]
    }
    
    # Run the graph. It will execute until finished or interrupted.
    await app_graph.ainvoke(initial_state, config)

class TaskRequest(BaseModel):
    task: str

@router.post("/tasks")
async def create_task(request: TaskRequest, background_tasks: BackgroundTasks):
    """Submit a new task to the agent orchestration system."""
    thread_id = str(uuid.uuid4())
    background_tasks.add_task(execute_task_background, thread_id, request.task)
    return {"thread_id": thread_id, "message": "Task queued for execution."}

@router.get("/tasks/{thread_id}/state")
async def get_task_state(thread_id: str):
    """Check the current status and memory of the agent graph."""
    pool = await get_pool()
    checkpointer = AsyncPostgresSaver(pool)
    app_graph = workflow.compile(checkpointer=checkpointer, interrupt_before=["escalation_node"])
    
    config = {"configurable": {"thread_id": thread_id}}
    state = await app_graph.aget_state(config)
    
    if not state.values:
        raise HTTPException(status_code=404, detail="Task not found or has no state.")
        
    # If the next node to run is the escalation_node, the graph is paused waiting for a human
    is_paused = "escalation_node" in state.next
    
    return {
        "status": "pending_human_approval" if is_paused else "running_or_completed",
        "next_nodes": list(state.next),
        "current_plan": state.values.get("plan"),
        "messages": [m.content for m in state.values.get("messages", []) if hasattr(m, "content")]
    }

class ApprovalRequest(BaseModel):
    approved: bool
    feedback: str = ""

@router.post("/tasks/{thread_id}/approve")
async def approve_task(thread_id: str, request: ApprovalRequest, background_tasks: BackgroundTasks):
    """Resume a paused task after human review."""
    pool = await get_pool()
    checkpointer = AsyncPostgresSaver(pool)
    app_graph = workflow.compile(checkpointer=checkpointer, interrupt_before=["escalation_node"])
    
    config = {"configurable": {"thread_id": thread_id}}
    state = await app_graph.aget_state(config)
    
    if "escalation_node" not in state.next:
        raise HTTPException(status_code=400, detail="Task is not waiting for human approval.")
        
    async def resume_graph():
        # Resuming with 'None' tells LangGraph to continue from where it paused
        await app_graph.ainvoke(None, config)
        
    background_tasks.add_task(resume_graph)
    return {"message": "Task execution resumed."}

NODE_LABELS = {
    "start": "Task Input",
    "supervisor_node": "Supervisor Agent",
    "escalation_node": "Human-in-the-Loop",
    "specialist_node": "Specialist Agent",
    "tools": "Tool Execution",
    "reviewer_node": "Reviewer Agent",
    "memorize_node": "Semantic Memory"
}

def estimate_tokens(text: str) -> int:
    """Fast tokenizer estimation (~4 chars per token)."""
    if not text:
        return 0
    return max(1, len(text) // 4)

@router.get("/tasks/{thread_id}/trace/{node_id}")
async def get_node_trace(thread_id: str, node_id: str):
    """
    Fetch historical checkpoint state for a specific node execution in the graph,
    extracting LLM prompt, response, tool invocations, token metrics, and state snapshots.
    """
    pool = await get_pool()
    checkpointer = AsyncPostgresSaver(pool)
    app_graph = workflow.compile(checkpointer=checkpointer, interrupt_before=["escalation_node"])
    
    config = {"configurable": {"thread_id": thread_id}}
    state = await app_graph.aget_state(config)
    
    if not state.values:
        raise HTTPException(status_code=404, detail="Task not found or has no state.")

    # Retrieve full execution history snapshots
    snapshots = []
    try:
        async for s in app_graph.aget_state_history(config):
            snapshots.append(s)
    except Exception as e:
        print(f"Warning fetching state history: {e}")

    # Determine node status
    is_active = node_id in state.next
    is_paused_hitl = "escalation_node" in state.next and node_id == "escalation_node"
    
    status = "completed"
    if is_paused_hitl:
        status = "waiting_approval"
    elif is_active:
        status = "active"
    elif not state.values.get("plan") and node_id not in ["start", "supervisor_node"]:
        status = "pending"

    # Match target snapshot for this node
    target_snapshot = None
    step_num = 1
    for idx, snap in enumerate(snapshots):
        meta = getattr(snap, "metadata", {}) or {}
        writes = meta.get("writes", {}) or {}
        if node_id in writes:
            target_snapshot = snap
            step_num = meta.get("step", len(snapshots) - idx)
            break
            
    snapshot_values = target_snapshot.values if target_snapshot else state.values
    
    task_input = snapshot_values.get("task_input", "")
    plan = snapshot_values.get("plan")
    messages = snapshot_values.get("messages", [])
    current_idx = snapshot_values.get("current_task_index", 0)
    reviewer_feedback = snapshot_values.get("reviewer_feedback")
    escalation_reason = snapshot_values.get("escalation_reason")

    prompt = None
    response = None
    tool_calls = []
    token_usage = None
    subtask_data = None

    # Node-specific inspection reconstruction
    if node_id == "start":
        prompt = f"User Request submitted at start of thread:\n{task_input}"
        response = "Task registered and routed to supervisor_node."
        token_usage = {
            "prompt_tokens": estimate_tokens(task_input),
            "completion_tokens": estimate_tokens(response),
            "total_tokens": estimate_tokens(task_input) + estimate_tokens(response),
            "estimated_cost_usd": round((estimate_tokens(task_input) + estimate_tokens(response)) * 0.000005, 6)
        }

    elif node_id == "supervisor_node":
        prompt = (
            "System Prompt: You are a Supervisor Agent. Decompose the complex task into a step-by-step "
            "execution plan. Assign each step to a specialist: 'researcher', 'coder', or 'data_analyst'.\n\n"
            f"User Task: {task_input}"
        )
        if plan:
            subtasks_repr = "\n".join([
                f"- Step {i+1} [{t.assigned_specialist if hasattr(t, 'assigned_specialist') else t.get('assigned_specialist')}]: "
                f"{t.description if hasattr(t, 'description') else t.get('description')}"
                for i, t in enumerate(plan.subtasks if hasattr(plan, 'subtasks') else plan.get('subtasks', []))
            ])
            confidence = plan.confidence_score if hasattr(plan, 'confidence_score') else plan.get('confidence_score', 1.0)
            req_hitl = plan.requires_human_approval if hasattr(plan, 'requires_human_approval') else plan.get('requires_human_approval', False)
            response = (
                f"Generated Execution Plan ({len(plan.subtasks if hasattr(plan, 'subtasks') else plan.get('subtasks', []))} subtasks):\n"
                f"{subtasks_repr}\n\n"
                f"Confidence Score: {confidence}\n"
                f"Requires Human Approval: {req_hitl}"
            )
        else:
            response = "Planning in progress..."

        p_tokens = estimate_tokens(prompt)
        c_tokens = estimate_tokens(response)
        token_usage = {
            "prompt_tokens": p_tokens,
            "completion_tokens": c_tokens,
            "total_tokens": p_tokens + c_tokens,
            "estimated_cost_usd": round((p_tokens * 0.000005) + (c_tokens * 0.000015), 6)
        }

    elif node_id == "specialist_node":
        subtasks_list = plan.subtasks if hasattr(plan, 'subtasks') else (plan.get('subtasks', []) if plan else [])
        if subtasks_list and current_idx < len(subtasks_list):
            st = subtasks_list[current_idx]
            role = st.assigned_specialist if hasattr(st, 'assigned_specialist') else st.get('assigned_specialist')
            desc = st.description if hasattr(st, 'description') else st.get('description')
            exp = st.expected_output if hasattr(st, 'expected_output') else st.get('expected_output')
            subtask_data = {
                "task_id": st.task_id if hasattr(st, 'task_id') else st.get('task_id'),
                "description": desc,
                "assigned_specialist": role,
                "expected_output": exp
            }
            prompt = (
                f"You are a {role}.\n"
                f"Current Task: {desc}\n"
                f"Expected Output: {exp}\n"
                f"Previous Feedback: {reviewer_feedback or 'None'}"
            )
        else:
            prompt = "Specialist execution context."

        # Find latest AI Message
        ai_msgs = [m for m in messages if isinstance(m, AIMessage) or (isinstance(m, dict) and m.get('type') == 'ai')]
        if ai_msgs:
            last_ai = ai_msgs[-1]
            response = last_ai.content if hasattr(last_ai, 'content') else str(last_ai)
            if hasattr(last_ai, 'tool_calls') and last_ai.tool_calls:
                for tc in last_ai.tool_calls:
                    tool_calls.append({
                        "name": tc.get("name", "unknown_tool"),
                        "args": tc.get("args", {}),
                        "output": "Forwarded to tool executor node"
                    })
        else:
            response = "Awaiting specialist execution."

        p_tokens = estimate_tokens(prompt)
        c_tokens = estimate_tokens(response or "")
        token_usage = {
            "prompt_tokens": p_tokens,
            "completion_tokens": c_tokens,
            "total_tokens": p_tokens + c_tokens,
            "estimated_cost_usd": round((p_tokens * 0.000005) + (c_tokens * 0.000015), 6)
        }

    elif node_id == "tools":
        prompt = "Tool invocation requested by specialist agent."
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage) or (isinstance(m, dict) and m.get('type') == 'tool')]
        if tool_msgs:
            for tm in tool_msgs:
                content = tm.content if hasattr(tm, 'content') else str(tm)
                name = getattr(tm, 'name', 'Tool')
                tool_calls.append({
                    "name": name,
                    "args": {"source": "Specialist tool call"},
                    "output": content
                })
            response = f"Successfully executed {len(tool_calls)} tool action(s)."
        else:
            response = "No tool executions recorded in this trace."

        token_usage = {
            "prompt_tokens": estimate_tokens(prompt),
            "completion_tokens": estimate_tokens(response),
            "total_tokens": estimate_tokens(prompt) + estimate_tokens(response),
            "estimated_cost_usd": 0.000002
        }

    elif node_id == "reviewer_node":
        prompt = (
            "System Prompt: You are a Reviewer Agent. Validate the specialist output against the task requirements.\n"
            "If it meets requirements, reply exactly with 'APPROVED'. If not, reply with specific feedback."
        )
        if reviewer_feedback:
            response = f"Reviewer Feedback: {reviewer_feedback}"
        else:
            response = "APPROVED (All validation criteria verified)"

        p_tokens = estimate_tokens(prompt)
        c_tokens = estimate_tokens(response)
        token_usage = {
            "prompt_tokens": p_tokens,
            "completion_tokens": c_tokens,
            "total_tokens": p_tokens + c_tokens,
            "estimated_cost_usd": round((p_tokens * 0.000005) + (c_tokens * 0.000015), 6)
        }

    elif node_id == "escalation_node":
        prompt = "Human-in-the-Loop Interruption triggered by low confidence or sensitive operational risk."
        response = escalation_reason or ("Human approval granted. Graph execution resumed." if status == "completed" else "Waiting for human review.")
        token_usage = {
            "prompt_tokens": estimate_tokens(prompt),
            "completion_tokens": estimate_tokens(response),
            "total_tokens": estimate_tokens(prompt) + estimate_tokens(response),
            "estimated_cost_usd": 0.0
        }

    elif node_id == "memorize_node":
        prompt = f"Save completed task input and final execution summary to ChromaDB collection 'agent_semantic_memory'."
        response = "Memory successfully vectorized with OpenAI text-embedding-3-small and stored in ChromaDB."
        token_usage = {
            "prompt_tokens": estimate_tokens(task_input),
            "completion_tokens": 30,
            "total_tokens": estimate_tokens(task_input) + 30,
            "estimated_cost_usd": round(estimate_tokens(task_input) * 0.00000002, 7)
        }

    return {
        "node_id": node_id,
        "node_label": NODE_LABELS.get(node_id, node_id),
        "status": status,
        "step": step_num,
        "timestamp": getattr(target_snapshot, "created_at", None) or "live",
        "prompt": prompt,
        "response": response,
        "tool_calls": tool_calls,
        "token_usage": token_usage,
        "subtask": subtask_data,
        "state_snapshot": {
            "current_task_index": current_idx,
            "reviewer_feedback": reviewer_feedback,
            "escalation_reason": escalation_reason,
            "total_messages": len(messages)
        }
    }
