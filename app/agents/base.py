from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class AgentMessage(BaseModel):
    sender: str
    recipient: str
    content: str
    role: str = "assistant"
    metadata: Optional[Dict[str, Any]] = None

class BaseAgent(ABC):
    def __init__(self, name: str, role: str, system_prompt: str):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt

    @abstractmethod
    async def process_task(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Process an assigned task and return the generated outcome."""
        pass
