from celery import Celery
from app.core.config import settings

celery = Celery(
    "orchestration_worker",
    broker=settings.CELERY_BROKER_URL or "redis://:redispassword@localhost:6379/1",
    backend=settings.CELERY_RESULT_BACKEND or "redis://:redispassword@localhost:6379/2",
    include=["app.worker.tasks"]
)

celery.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max per task
    task_routes={
        "app.worker.tasks.orchestrate_agent_pipeline": {"queue": "agents"},
        "app.worker.tasks.embed_and_index_memory": {"queue": "memory"},
        "app.worker.tasks.*": {"queue": "default"},
    }
)
