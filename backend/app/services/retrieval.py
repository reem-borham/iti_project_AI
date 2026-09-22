import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("rag_backend.retrieval")


class RetrievalService:
    """Manages Chroma vector store connection and passage retrieval."""

    def __init__(
        self,
        vector_store_dir: Path,
        collection_name: str = "rag_textbooks",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        k: int = 4,
    ):
        self.vector_store_dir = Path(vector_store_dir)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.k = k
        self.collection = None
        self.is_ready = False
        self._initialize_vector_store()

    def _initialize_vector_store(self) -> None:
        """Initialize ChromaDB persistent client and collection."""
        try:
            import chromadb
            from chromadb.utils import embedding_functions

            logger.info(f"Connecting to Chroma store at: {self.vector_store_dir}")
            self.vector_store_dir.mkdir(parents=True, exist_ok=True)
            
            client = chromadb.PersistentClient(path=str(self.vector_store_dir))
            
            embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model_name
            )

            # Check if collection exists
            existing_collections = [c.name for c in client.list_collections()]
            if self.collection_name in existing_collections:
                self.collection = client.get_collection(
                    name=self.collection_name,
                    embedding_function=embedding_fn,
                )
                count = self.collection.count()
                logger.info(f"Chroma collection '{self.collection_name}' loaded with {count} chunks.")
                self.is_ready = True
            else:
                logger.warning(
                    f"Collection '{self.collection_name}' not found in {self.vector_store_dir}. "
                    "Empty collection initialized. Chunks should be indexed from raw textbooks."
                )
                self.collection = client.get_or_create_collection(
                    name=self.collection_name,
                    embedding_function=embedding_fn,
                )
                self.is_ready = True
        except Exception as exc:
            logger.error(f"Failed to initialize Chroma vector store: {exc}")
            self.collection = None
            self.is_ready = False

    def retrieve(self, query: str, k: int = None) -> Tuple[str, List[str]]:
        """
        Retrieve relevant chunks for a query.
        Returns:
            context_str: Formatted context passages for prompt injection.
            sources: List of unique source citations (e.g. ['book.pdf, p.10']).
        """
        target_k = k or self.k
        if not self.is_ready or self.collection is None:
            logger.warning("Retrieval attempted but vector store is not ready.")
            return "", []

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=target_k,
            )

            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]

            if not documents:
                return "", []

            context_parts: List[str] = []
            sources: List[str] = []

            for i, (doc, meta) in enumerate(zip(documents, metadatas), 1):
                meta = meta or {}
                src = meta.get("source", "unknown_source")
                page = meta.get("page", "?")
                citation = f"{src}, p.{page}"
                if citation not in sources:
                    sources.append(citation)

                context_parts.append(f"[Passage {i} | Source: {src}, p.{page}]\n{doc}")

            formatted_context = "\n\n".join(context_parts)
            return formatted_context, sources

        except Exception as exc:
            logger.error(f"Error querying vector store: {exc}")
            return "", []
