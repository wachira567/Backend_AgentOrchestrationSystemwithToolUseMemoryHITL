import uuid
import chromadb
from langchain_openai import OpenAIEmbeddings

# Connect to the ChromaDB container defined in docker-compose.yml
chroma_client = chromadb.HttpClient(host="localhost", port=8000)
collection = chroma_client.get_or_create_collection(name="agent_semantic_memory")

# Using OpenAI's optimized embedding model
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

def retrieve_past_context(task_input: str) -> str:
    """Fetch similar past tasks to inform the Supervisor's execution plan."""
    try:
        query_embedding = embeddings_model.embed_query(task_input)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        
        if not results["documents"] or not results["documents"][0]:
            return "No relevant past memories found."
            
        memories = "\n".join(results["documents"][0])
        return f"Relevant past context from memory:\n{memories}"
    except Exception as e:
        return f"Memory retrieval failed: {str(e)}"

def save_memory(task_input: str, summary: str):
    """Save a completed task and its final summary into long-term memory."""
    try:
        doc_id = str(uuid.uuid4())
        content = f"Task: {task_input}\nOutcome/Summary: {summary}"
        embedding = embeddings_model.embed_query(content)
        
        collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[{"task": task_input}]
        )
    except Exception as e:
        print(f"Failed to save memory: {e}")
