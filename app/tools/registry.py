from langchain_core.tools import tool

@tool
def web_search_tool(query: str) -> str:
    """
    Search the web for real-time information. 
    Use this when you need current facts, news, or to verify information.
    """
    # In production, replace with Tavily, DuckDuckGo, or Google API.
    return f"Mock search results for: {query}. The latest data suggests this is highly relevant."

@tool
def execute_python_code(code: str) -> str:
    """
    Execute a python script in a sandboxed environment and return the standard output.
    Useful for data analysis, math, or complex string manipulation.
    """
    # WARNING: Do not use `exec()` in production without a secure Docker sandbox!
    # This is a safe mock implementation.
    return f"Successfully executed script. Output: \n[Mock Execution Result of {len(code)} bytes of code]"

@tool
def query_database(sql_query: str) -> str:
    """
    Run a read-only SQL query against the internal database to extract business data.
    """
    # Mock database read
    return f"Executed query: {sql_query}\nResult: 5 rows returned. [Row 1: Data...]"

# Group tools by specialist type
RESEARCHER_TOOLS = [web_search_tool]
CODER_TOOLS = [execute_python_code]
DATA_ANALYST_TOOLS = [query_database, execute_python_code]

# All tools for easy binding
ALL_TOOLS = RESEARCHER_TOOLS + CODER_TOOLS + DATA_ANALYST_TOOLS
