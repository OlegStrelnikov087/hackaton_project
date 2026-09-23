from backend.retrieval.embedding import EmbeddingService

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector


DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/rag"

embedder = EmbeddingService("BAAI/bge-m3")


def search(query: str, top_k: int = 5):
    # Превращаем запрос пользователя в embedding
    query_embedding = embedder.embed_query(query)
    query_vector = Vector(query_embedding)

    # Подключаемся к PostgreSQL
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

    return [
        {
            "id": row[0],
            "document_id": row[1],
            "filename": row[2],
            "page": row[3],
            "content": row[4],
            "similarity": float(row[5])
        }
        for row in rows
    ]


if __name__ == "__main__":
    query = input("Введите запрос: ")

    results = search(query, top_k=5)

    print("\nРезультаты:\n")

    for i, result in enumerate(results, 1):
        print(f"--- #{i} ---")
        print(f"Файл: {result['filename']}")
        print(f"Страница: {result['page']}")
        print(f"Similarity: {result['similarity']:.4f}")
        print(f"Текст: {result['content']}")
        print()