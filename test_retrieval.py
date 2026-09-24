import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from backend.retrieval.embedding import EmbeddingService
from backend.retrieval.reranker import Reranker


DATABASE_URL = (
    "postgresql://postgres:postgres@127.0.0.1:5432/rag"
)

VECTOR_TOP_K = 30
FINAL_TOP_K = 5


print("Loading embedding model...")

embedder = EmbeddingService(
    "BAAI/bge-m3"
)

print("Loading reranker...")

reranker = Reranker()

print("Models loaded.")


def vector_search(
    query_embedding: list[float],
    top_k: int = VECTOR_TOP_K
) -> list[dict]:

    query_vector = Vector(query_embedding)

    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)

        with conn.cursor() as cursor:

            cursor.execute(
                """
                SELECT
                    id,
                    document_id,
                    filename,
                    page,
                    content,
                    1 - (embedding <=> %s) AS similarity
                FROM chunks
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (
                    query_vector,
                    query_vector,
                    top_k
                )
            )

            rows = cursor.fetchall()

    results = []

    for row in rows:
        results.append({
            "id": row[0],
            "document_id": row[1],
            "filename": row[2],
            "page": row[3],
            "content": row[4],
            "vector_score": float(row[5])
        })

    return results


def search(
    query: str,
    vector_top_k: int = VECTOR_TOP_K,
    final_top_k: int = FINAL_TOP_K
) -> list[dict]:

    print("\nCreating query embedding...")

    query_embedding = embedder.embed_query(
        query
    )

    print(
        f"Searching PostgreSQL with pgvector: "
        f"Top-{vector_top_k}"
    )

    candidates = vector_search(
        query_embedding,
        vector_top_k
    )

    print(
        f"Found {len(candidates)} candidates"
    )

    print(
        f"Reranking {len(candidates)} candidates..."
    )

    results = reranker.rerank(
        query,
        candidates,
        top_n=final_top_k
    )

    return results


if __name__ == "__main__":

    query = input(
        "Введите запрос: "
    )

    results = search(
        query,
        vector_top_k=30,
        final_top_k=5
    )

    print("\nFINAL RESULTS\n")

    for i, result in enumerate(
        results,
        start=1
    ):

        print(
            f"--- #{i} ---"
        )

        print(
            "File:",
            result["filename"]
        )

        print(
            "Page:",
            result["page"]
        )

        print(
            "Vector score:",
            f"{result['vector_score']:.4f}"
        )

        print(
            "Rerank score:",
            f"{result['rerank_score']:.4f}"
        )

        print(
            "Content:",
            result["content"][:500]
        )

        print()