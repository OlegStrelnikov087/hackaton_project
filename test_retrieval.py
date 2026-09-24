import json
from pathlib import Path

import numpy as np

from backend.retrieval.embedding import EmbeddingService
from backend.retrieval.reranker import Reranker


INPUT_FILE = Path(
    "backend/output/embedded_chunks.json"
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


def load_chunks():
    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def cosine_similarity(a, b):
    a = np.asarray(
        a,
        dtype=np.float32
    )

    b = np.asarray(
        b,
        dtype=np.float32
    )

    return np.dot(a, b) / (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )


def vector_search(
    query_embedding,
    chunks,
    top_k
):
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
            "vector_score": float(similarity)
        })

    results.sort(
        key=lambda x: x["vector_score"],
        reverse=True
    )

    return results[:top_k]


def search(
    query: str,
    vector_top_k: int = VECTOR_TOP_K,
    final_top_k: int = FINAL_TOP_K
):

    chunks = load_chunks()

    print(
        f"Loaded chunks: {len(chunks)}"
    )

    # ---------------------------
    # 1. Embedding запроса
    # ---------------------------

    print(
        "Creating query embedding..."
    )

    query_embedding = embedder.embed_query(
        query
    )

    # ---------------------------
    # 2. Vector search
    # ---------------------------

    print(
        f"Vector search: Top-{vector_top_k}"
    )

    candidates = vector_search(
        query_embedding,
        chunks,
        vector_top_k
    )

    # ---------------------------
    # 3. Reranking
    # ---------------------------

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

    results = search(query)

    print(
        "\nFINAL RESULTS\n"
    )

    for i, result in enumerate(
        results,
        1
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