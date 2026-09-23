from .embedding import EmbeddingService
from .pgvector_store import PgVectorStore


class Retriever:
    def __init__(
        self,
        embedder: EmbeddingService,
        vector_store: PgVectorStore
    ):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 30
    ) -> list[dict]:

        query_embedding = self.embedder.embed_query(
            query
        )

        return self.vector_store.search(
            query_embedding,
            top_k
        )