import json
from pathlib import Path

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector


DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/rag"

# backend/retrieval/import_embeddings.py
# parents[0] = retrieval
# parents[1] = backend
BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "output" / "embedded_chunks.json"


def load_embeddings():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Файл не найден: {INPUT_FILE}"
        )

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            "JSON должен содержать список chunks."
        )

    return data


def validate_chunk(chunk, index):
    required_fields = [
        "document_id",
        "filename",
        "content",
        "embedding"
    ]

    for field in required_fields:
        if field not in chunk:
            raise ValueError(
                f"Chunk #{index} не содержит поле '{field}'."
            )

    if len(chunk["embedding"]) != 1024:
        raise ValueError(
            f"Chunk #{index}: embedding имеет размер "
            f"{len(chunk['embedding'])}, ожидалось 1024."
        )


def insert_chunks(chunks):
    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)

        with conn.cursor() as cursor:
            for index, chunk in enumerate(chunks):
                validate_chunk(chunk, index)

                embedding = Vector(chunk["embedding"])

                cursor.execute(
                    """
                    INSERT INTO chunks (
                        document_id,
                        filename,
                        page,
                        section,
                        content,
                        embedding
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        chunk["document_id"],
                        chunk["filename"],
                        chunk.get("page"),
                        chunk.get("section"),
                        chunk["content"],
                        embedding
                    )
                )

                if (index + 1) % 100 == 0:
                    print(
                        f"Inserted: {index + 1}/{len(chunks)}"
                    )

        conn.commit()


def main():
    print(f"Loading: {INPUT_FILE}")

    chunks = load_embeddings()

    print(f"Chunks loaded: {len(chunks)}")

    insert_chunks(chunks)

    print("Import completed successfully.")


if __name__ == "__main__":
    main()