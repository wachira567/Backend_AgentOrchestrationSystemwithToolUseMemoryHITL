import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from app.agents.state import AgentState, ExecutionPlan
from app.tools.registry import ALL_TOOLS

# Initialize the LLM (ensure OPENAI_API_KEY is in your environment)
llm = ChatOpenAI(model="gpt-4o", temperature=0)

async def supervisor_node(state: AgentState) -> dict:
    """Decomposes the complex task into a step-by-step execution plan."""
    task = state["task_input"]
    
    system_prompt = (
        "You are a Supervisor Agent. Decompose the following complex task into a step-by-step "
        "execution plan. Assign each step to a specialist: 'researcher', 'coder', or 'data_analyst'. "
        "Score your confidence in this plan (0.0 to 1.0). If the task involves sensitive operations "
        "(like deleting data, spending money, or external emails), set requires_human_approval to true."
    )
    
    structured_llm = llm.with_structured_output(ExecutionPlan)
    plan = await structured_llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=task)
    ])
    
    # Initialize the plan in the state
    return {"plan": plan, "current_task_index": 0, "escalation_reason": None}

async def specialist_node(state: AgentState) -> dict:
    """Executes the current subtask using tools."""
    plan = state["plan"]
    idx = state["current_task_index"]
    
    if not plan or idx >= len(plan.subtasks):
        return {}
        
    current_task = plan.subtasks[idx]
    
    # Bind the available tools to the LLM so it can execute actions
    worker_llm = llm.bind_tools(ALL_TOOLS)
    
    prompt = (
        f"You are a {current_task.assigned_specialist}. "
        f"Your current task: {current_task.description}\n"
        f"Expected output format: {current_task.expected_output}\n"
        f"Previous Reviewer Feedback (if any): {state.get('reviewer_feedback', 'None')}"
    )
    
    response = await worker_llm.ainvoke([SystemMessage(content=prompt)] + state["messages"])
    return {"messages": [response]}

async def reviewer_node(state: AgentState) -> dict:
    """Validates the specialist's output before returning to the supervisor."""
    plan = state["plan"]
    idx = state["current_task_index"]
    last_message = state["messages"][-1]
    
    current_task = plan.subtasks[idx]
    
    prompt = (
        "You are a Reviewer Agent. Validate the following output against the task requirements.\n"
        f"Task: {current_task.description}\n"
        f"Expected Output: {current_task.expected_output}\n"
        f"Output to review: {last_message.content}\n\n"
        "If it meets requirements, reply exactly with 'APPROVED'. If not, reply with specific feedback."
    )
    
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    
    if "APPROVED" in response.content.upper():
        return {
            "current_task_index": idx + 1,
            "reviewer_feedback": None,
            "messages": [AIMessage(content=f"Subtask {current_task.task_id} approved by Reviewer.")]
        }
    else:
        return {
            "reviewer_feedback": response.content,
            "messages": [AIMessage(content=f"Reviewer rejected output. Feedback: {response.content}")]
        }

async def escalation_node(state: AgentState) -> dict:
    """Landing pad for Human-in-the-Loop interruptions."""
    # When execution resumes after a human approves, it flows through this node.
    return {"escalation_reason": "Resolved by human operator."}
