import asyncio
from typing import Any, Dict, Optional
from celery import Celery
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.core.config import settings
from app.memory.db import get_pool
from app.agents.graph import workflow

# Initialize Celery app with Redis broker and backend
celery_app = Celery(
    "agent_worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour timeout
    result_expires=86400,  # 24 hours
)

async def _run_agent_workflow(
    thread_id: str, 
    task_input: Optional[str] = None, 
    resume: bool = False
):
    """
    Asynchronous executor for LangGraph with PostgreSQL persistence.
    Interrupts before HITL escalation node to allow human review.
    """
    pool = await get_pool()
    checkpointer = AsyncPostgresSaver(pool)
    app_graph = workflow.compile(
        checkpointer=checkpointer, 
        interrupt_before=["escalation_node"]
    )
    
    config = {"configurable": {"thread_id": thread_id}}
    
    if resume:
        # Resuming execution after HITL approval or time-travel state update
        await app_graph.ainvoke(None, config)
    else:
        # Fresh task initiation
        initial_state = {
            "task_input": task_input or "",
            "messages": [HumanMessage(content=task_input or "")]
        }
        await app_graph.ainvoke(initial_state, config)

@celery_app.task(name="execute_agent_workflow", bind=True)
def execute_agent_workflow(
    self, 
    thread_id: str, 
    task_input: Optional[str] = None, 
    resume: bool = False
) -> Dict[str, Any]:
    """
    Celery task that wraps the async LangGraph workflow execution inside an asyncio event loop.
    Supports task initiation, HITL resumption, and time-travel replay execution.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(
            _run_agent_workflow(thread_id=thread_id, task_input=task_input, resume=resume)
        )
        return {
            "status": "COMPLETED",
            "thread_id": thread_id,
            "resumed": resume
        }
    except Exception as e:
        return {
            "status": "FAILED",
            "thread_id": thread_id,
            "error": str(e)
        }
    finally:
        loop.close()
