from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Agent Orchestration API"
    # Using the credentials from our docker-compose.yml
    POSTGRES_URI: str = "postgresql://agent_user:agent_password@localhost:5432/agent_memory"
    
settings = Settings()
