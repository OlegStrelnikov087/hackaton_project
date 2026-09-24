import json
from pathlib import Path

import numpy as np

from backend.retrieval.embedding import EmbeddingService


INPUT_FILE = Path("backend/output/embedded_chunks.json")


def load_chunks():
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


def search(query: str, top_k: int = 5):
    chunks = load_chunks()

    embedder = EmbeddingService("BAAI/bge-m3")

    # Делаем embedding только пользовательского запроса
    query_embedding = embedder.embed_query(query)

    results = []

    for chunk in chunks:
        similarity = cosine_similarity(
            query_embedding,
            chunk["embedding"]
        )

        results.append({
            "document_id": chunk["document_id"],
            "filename": chunk["filename"],
            "page": chunk.get("page"),
            "content": chunk["content"],
            "similarity": float(similarity)
        })

    results.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    return results[:top_k]


if __name__ == "__main__":  
    query = input("Введите запрос: ")

    results = search(query, top_k=5)

    print("\nРезультаты:\n")

    for i, result in enumerate(results, 1):
        print(f"--- #{i} ---")
        print("File:", result["filename"])
        print("Page:", result["page"])
        print("Similarity:", f"{result['similarity']:.4f}")
        print("Content:", result["content"][:500])
        print()