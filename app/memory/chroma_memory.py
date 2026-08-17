from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.core.config import settings

class ChromaSemanticMemory:
    """
    Long-Term Semantic Memory using ChromaDB.
    Indexes documents, research outputs, and agent insights with embeddings for RAG retrieval.
    """
    def __init__(self):
        self.host = settings.CHROMA_HOST
        self.port = settings.CHROMA_PORT
        self.collection_name = settings.CHROMA_COLLECTION_NAME
        self._client: Optional[chromadb.HttpClient] = None

    def get_client(self) -> chromadb.HttpClient:
        if self._client is None:
            try:
                self._client = chromadb.HttpClient(
                    host=self.host,
                    port=self.port,
                    settings=ChromaSettings(anonymized_telemetry=False)
                )
            except Exception as e:
                # Fallback in memory client for lightweight testing
                self._client = chromadb.EphemeralClient()
        return self._client

    def get_or_create_collection(self):
        client = self.get_client()
        return client.get_or_create_collection(name=self.collection_name)

    def add_memory(self, doc_id: str, document: str, metadata: Dict[str, Any]):
        collection = self.get_or_create_collection()
        collection.upsert(
            ids=[doc_id],
            documents=[document],
            metadatas=[metadata]
        )

    def query_similar(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        collection = self.get_or_create_collection()
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        memories = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
            ids = results["ids"][0] if "ids" in results else [""] * len(docs)
            for i in range(len(docs)):
                memories.append({
                    "id": ids[i],
                    "content": docs[i],
                    "metadata": metas[i] if i < len(metas) else {}
                })
        return memories

semantic_memory = ChromaSemanticMemory()
