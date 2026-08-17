import uuid
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.messages import HumanMessage

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
        "next_nodes": state.next,
        "current_plan": state.values.get("plan"),
        "messages": [m.content for m in state.values.get("messages", [])]
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
