import operator
from typing import Annotated, TypedDict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage

# --- Pydantic Models for Structured Output ---

class SubTask(BaseModel):
    task_id: str = Field(..., description="Unique identifier for the subtask, e.g., 'task_1'")
    description: str = Field(..., description="Clear description of what needs to be done")
    assigned_specialist: str = Field(..., description="Which specialist: 'researcher', 'coder', or 'data_analyst'")
    expected_output: str = Field(..., description="What the expected output format looks like")

class ExecutionPlan(BaseModel):
    subtasks: List[SubTask] = Field(..., description="List of subtasks to execute in order")
    confidence_score: float = Field(..., description="Supervisor's confidence in this plan (0.0 to 1.0)")
    requires_human_approval: bool = Field(..., description="Set to True if task involves sensitive actions")

# --- LangGraph State Definition ---

class AgentState(TypedDict):
    # 'messages' uses operator.add so new messages are appended to the list, not overwritten
    messages: Annotated[list[AnyMessage], operator.add]
    
    # The user's original request
    task_input: str
    
    # The supervisor's active plan
    plan: Optional[ExecutionPlan]
    
    # State tracking
    current_task_index: int
    escalation_reason: Optional[str]
    reviewer_feedback: Optional[str]
