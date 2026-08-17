import uuid
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from app.memory.db import get_pool
from app.agents.graph import workflow
from app.worker.celery_app import execute_agent_workflow

router = APIRouter()

class TaskRequest(BaseModel):
    task: str

@router.post("/tasks")
async def create_task(request: TaskRequest):
    """Submit a new task to the agent orchestration system via Celery and Redis."""
    thread_id = str(uuid.uuid4())
    
    # Enqueue execution in Celery worker queue
    task_job = execute_agent_workflow.delay(thread_id, request.task, False)
    
    return {
        "thread_id": thread_id, 
        "celery_task_id": task_job.id,
        "message": "Task queued in Redis for Celery worker execution."
    }

@router.get("/tasks/{thread_id}/state")
async def get_task_state(thread_id: str):
    """Check the current status and memory of the agent graph from PostgreSQL checkpointer."""
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
async def approve_task(thread_id: str, request: ApprovalRequest):
    """Resume a paused task after human review using Celery worker."""
    pool = await get_pool()
    checkpointer = AsyncPostgresSaver(pool)
    app_graph = workflow.compile(checkpointer=checkpointer, interrupt_before=["escalation_node"])
    
    config = {"configurable": {"thread_id": thread_id}}
    state = await app_graph.aget_state(config)
    
    if "escalation_node" not in state.next:
        raise HTTPException(status_code=400, detail="Task is not waiting for human approval.")
        
    # Dispatch resume task to Celery worker
    task_job = execute_agent_workflow.delay(thread_id, None, True)
    
    return {
        "thread_id": thread_id,
        "celery_task_id": task_job.id,
        "message": "Task execution resumed and dispatched to Celery worker."
    }

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
    checkpoint_id = None
    if target_snapshot:
        snap_cfg = getattr(target_snapshot, "config", {}) or {}
        configurable = snap_cfg.get("configurable", {}) if isinstance(snap_cfg, dict) else {}
        checkpoint_id = configurable.get("checkpoint_id") or getattr(target_snapshot, "checkpoint_id", None)
    if not checkpoint_id:
        state_cfg = getattr(state, "config", {}) or {}
        configurable = state_cfg.get("configurable", {}) if isinstance(state_cfg, dict) else {}
        checkpoint_id = configurable.get("checkpoint_id") or getattr(state, "checkpoint_id", str(uuid.uuid4()))
    
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
        "checkpoint_id": str(checkpoint_id),
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
            "task_input": task_input,
            "current_task_index": current_idx,
            "reviewer_feedback": reviewer_feedback,
            "escalation_reason": escalation_reason,
            "total_messages": len(messages)
        }
    }

class ReplayRequest(BaseModel):
    state_updates: Optional[Dict[str, Any]] = None
    modified_prompt: Optional[str] = None
    modified_response: Optional[str] = None
    fork_new_thread: bool = False

@router.post("/tasks/{thread_id}/replay/{checkpoint_id}")
async def replay_from_checkpoint(
    thread_id: str, 
    checkpoint_id: str, 
    request: ReplayRequest
):
    """
    Time-travel debugging endpoint:
    Loads the graph state at checkpoint_id, applies user-supplied context/state modifications,
    and resumes execution from that specific moment via Celery worker.
    """
    pool = await get_pool()
    checkpointer = AsyncPostgresSaver(pool)
    app_graph = workflow.compile(checkpointer=checkpointer, interrupt_before=["escalation_node"])

    target_thread_id = str(uuid.uuid4()) if request.fork_new_thread else thread_id
    config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id
        }
    }

    # Verify target checkpoint exists
    target_state = await app_graph.aget_state(config)
    if not target_state.values:
        config = {"configurable": {"thread_id": thread_id}}
        target_state = await app_graph.aget_state(config)
        if not target_state.values:
            raise HTTPException(status_code=404, detail="Target checkpoint state not found.")

    # Prepare state modifications
    updates: Dict[str, Any] = request.state_updates or {}
    
    if request.modified_prompt:
        messages = list(target_state.values.get("messages", []))
        messages.append(HumanMessage(content=request.modified_prompt))
        updates["messages"] = messages
        if "task_input" not in updates:
            updates["task_input"] = request.modified_prompt

    if request.modified_response:
        messages = list(updates.get("messages", target_state.values.get("messages", [])))
        messages.append(AIMessage(content=request.modified_response))
        updates["messages"] = messages

    # Apply updates to graph checkpointer
    if updates:
        await app_graph.aupdate_state(config, updates)

    # Dispatch to Celery worker to resume execution from the updated checkpoint
    task_job = execute_agent_workflow.delay(target_thread_id, None, True)

    return {
        "thread_id": target_thread_id,
        "checkpoint_id": checkpoint_id,
        "celery_task_id": task_job.id,
        "message": f"Graph replaying successfully via Celery worker from checkpoint {checkpoint_id}.",
        "forked": request.fork_new_thread
    }
