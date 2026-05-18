from __future__ import annotations

import uuid
from typing import Any


class VectorStore:
    """Simple vector store wrapper supporting local embeddings (sentence-transformers) or OpenAI embeddings."""

    def __init__(self) -> None:
        self._documents: list[str] = []
        self._metadatas: list[dict] = []
        self._embedding_provider: str = "local"
        self._embedding_model: Any = None
        self._openai_api_key: str | None = None

    def set_embedding_provider(self, provider: str, openai_api_key: str | None = None) -> None:
        self._embedding_provider = provider
        if provider == "openai":
            self._openai_api_key = openai_api_key
        elif provider == "local":
            self._load_local_model()

    def _load_local_model(self) -> None:
        if self._embedding_model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install with: pip install sentence-transformers"
            )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self._embedding_provider == "openai":
            return self._embed_openai(texts)
        return self._embed_local(texts)

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        self._load_local_model()
        embeddings = self._embedding_model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        if not self._openai_api_key:
            raise ValueError("OpenAI API key is required for OpenAI embeddings.")
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")
        client = OpenAI(api_key=self._openai_api_key)
        response = client.embeddings.create(model="text-embedding-3-small", input=texts)
        return [d.embedding for d in response.data]

    def add_documents(self, documents: list[str], metadatas: list[dict] | None = None) -> None:
        if metadatas is None:
            metadatas = [{}] * len(documents)
        self._documents.extend(documents)
        self._metadatas.extend(metadatas)

    def search(self, query: str, k: int = 5) -> list[dict]:
        if not self._documents:
            return []

        import numpy as np

        query_embedding = np.array(self._embed([query])[0])
        doc_embeddings = np.array(self._embed(self._documents))

        similarities = np.dot(doc_embeddings, query_embedding) / (
            np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-10
        )

        top_indices = np.argsort(similarities)[-k:][::-1]

        results: list[dict] = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append({
                    "content": self._documents[idx],
                    "metadata": self._metadatas[idx],
                    "score": float(similarities[idx]),
                })
        return results

    def clear(self) -> None:
        self._documents = []
        self._metadatas = []


class ChromaVectorStore(VectorStore):
    """ChromaDB-backed vector store for larger paper collections."""

    def __init__(self, collection_name: str = "paper_chunks", persist_dir: str | None = None) -> None:
        super().__init__()
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("chromadb is required. Install with: pip install chromadb")

        settings = Settings(anonymized_telemetry=False)
        if persist_dir:
            self._client = chromadb.PersistentClient(path=persist_dir, settings=settings)
        else:
            self._client = chromadb.Client(settings=settings)

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._id_counter = 0

    def add_documents(self, documents: list[str], metadatas: list[dict] | None = None) -> None:
        if metadatas is None:
            metadatas = [{}] * len(documents)
        ids = [str(uuid.uuid4()) for _ in documents]
        self._collection.add(documents=documents, metadatas=metadatas, ids=ids)

    def search(self, query: str, k: int = 5) -> list[dict]:
        results = self._collection.query(query_texts=[query], n_results=k)
        output: list[dict] = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0
                output.append({"content": doc, "metadata": metadata, "score": 1 - distance})
        return output

    def clear(self) -> None:
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection.name,
            metadata={"hnsw:space": "cosine"},
        )
