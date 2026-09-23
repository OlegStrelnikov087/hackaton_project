import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector


DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/rag"


class PgVectorStore:
    def __init__(self):
        self.connection = psycopg.connect(
            DATABASE_URL
        )

        register_vector(self.connection)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 30
    ) -> list[dict]:

        query_vector = Vector(query_embedding)

        with self.connection.cursor() as cursor:
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
                "vector_score": float(row[5])
            }
            for row in rows
        ]

    def close(self):
        self.connection.close()