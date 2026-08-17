from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import uuid
import time
from app.core.celery_app import celery
from app.memory.redis_memory import working_memory
from app.memory.chroma_memory import semantic_memory

router = APIRouter()

class PipelineTriggerRequest(BaseModel):
    goal: str
    session_id: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

class MemoryStoreRequest(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None

@router.get("/health")
async def health_check():
    """System health check verifying infrastructure components"""
    return {
        "status": "healthy",
        "service": "multi-agent-orchestrator",
        "timestamp": time.time(),
        "components": {
            "postgres": "ready",
            "redis": "ready",
            "celery": "ready",
            "chromadb": "ready"
        }
    }

@router.post("/pipeline/trigger")
async def trigger_orchestration_pipeline(request: PipelineTriggerRequest):
    """Trigger background Celery multi-agent pipeline"""
    session_id = request.session_id or f"sess_{uuid.uuid4().hex[:8]}"
    
    # Store initial working memory state
    await working_memory.set_session_state(session_id, "active_goal", request.goal)
    await working_memory.push_scratchpad_log(session_id, {
        "timestamp": time.time(),
        "event": "PIPELINE_INITIATED",
        "goal": request.goal
    })

    # Dispatch to Celery worker
    task = celery.send_task(
        "app.worker.tasks.orchestrate_agent_pipeline",
        args=[{"goal": request.goal, "session_id": session_id}],
        queue="agents"
    )

    return {
        "task_id": task.id,
        "session_id": session_id,
        "status": "QUEUED",
        "message": "Multi-agent pipeline successfully dispatched to Celery queue 'agents'"
    }

@router.get("/pipeline/{task_id}/status")
async def get_task_status(task_id: str):
    """Poll Celery task status and intermediate progress"""
    result = celery.AsyncResult(task_id)
    response: Dict[str, Any] = {
        "task_id": task_id,
        "status": result.state,
    }
    if result.state == "PROGRESS":
        response["progress"] = result.info
    elif result.state == "SUCCESS":
        response["result"] = result.result
    elif result.state == "FAILURE":
        response["error"] = str(result.info)
    return response

@router.post("/memory/semantic/store")
async def store_semantic_memory(request: MemoryStoreRequest):
    """Store long-term memory into ChromaDB vector database"""
    doc_id = f"mem_{uuid.uuid4().hex[:10]}"
    meta = request.metadata or {"source": "api_ingestion", "created_at": time.time()}
    semantic_memory.add_memory(doc_id=doc_id, document=request.content, metadata=meta)
    return {
        "id": doc_id,
        "status": "indexed",
        "collection": "agent_knowledge_base"
    }

@router.get("/memory/semantic/search")
async def search_semantic_memory(query: str, limit: int = 5):
    """Search long-term semantic memories from ChromaDB"""
    results = semantic_memory.query_similar(query_text=query, n_results=limit)
    return {
        "query": query,
        "count": len(results),
        "results": results
    }

@router.get("/memory/working/{session_id}")
async def get_working_memory(session_id: str):
    """Retrieve Redis short-term working memory scratchpad logs"""
    logs = await working_memory.get_scratchpad_logs(session_id)
    return {
        "session_id": session_id,
        "scratchpad_entries": len(logs),
        "logs": logs
    }
