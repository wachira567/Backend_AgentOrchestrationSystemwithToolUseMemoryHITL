import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Agent Orchestration API"
    # Using the credentials from our docker-compose.yml
    POSTGRES_URI: str = os.getenv(
        "POSTGRES_URI", 
        "postgresql://agent_user:agent_password@localhost:5432/agent_memory"
    )
    REDIS_URL: str = os.getenv(
        "REDIS_URL", 
        "redis://localhost:6379/0"
    )
    CELERY_BROKER_URL: str = os.getenv(
        "CELERY_BROKER_URL", 
        "redis://localhost:6379/0"
    )
    CELERY_RESULT_BACKEND: str = os.getenv(
        "CELERY_RESULT_BACKEND", 
        "redis://localhost:6379/1"
    )
    
settings = Settings()
