import json
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from pgvector import Vector


DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/rag"

INPUT_FILE = (
    Path(__file__).resolve().parents[1]
    / "output"
    / "embedded_chunks.json"
)


def load_chunks():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Файл не найден: {INPUT_FILE}"
        )

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not isinstance(chunks, list):
        raise ValueError(
            "embedded_chunks.json должен содержать список"
        )

    return chunks


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
                f"Chunk #{index} не содержит поле '{field}'"
            )

    if len(chunk["embedding"]) != 1024:
        raise ValueError(
            f"Chunk #{index}: embedding имеет "
            f"{len(chunk['embedding'])} измерений, "
            f"а ожидалось 1024"
        )


def insert_chunks(chunks):
    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)

        with conn.cursor() as cursor:

            for index, chunk in enumerate(chunks):
                validate_chunk(chunk, index)

                cursor.execute(
                    """
                    INSERT INTO chunks (
                        document_id,
                        filename,
                        page,
                        content,
                        embedding
                    )
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        chunk["document_id"],
                        chunk["filename"],
                        chunk.get("page"),
                        chunk["content"],
                        Vector(chunk["embedding"])
                    )
                )

                if (index + 1) % 100 == 0:
                    print(
                        f"Inserted: {index + 1}/{len(chunks)}"
                    )

        conn.commit()


def main():
    print(f"Loading: {INPUT_FILE}")

    chunks = load_chunks()

    print(f"Loaded chunks: {len(chunks)}")

    insert_chunks(chunks)

    print("Import completed successfully.")


if __name__ == "__main__":
    main()