import gc

import torch
import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from backend.retrieval.embedding import EmbeddingService
from backend.retrieval.reranker import Reranker
from backend.llm.llm import LocalLLM


# =========================
# CONFIG
# =========================

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/rag"

EMBEDDING_MODEL = "BAAI/bge-m3"

VECTOR_TOP_K = 30
FINAL_TOP_K = 5


# =========================
# MEMORY CLEANUP
# =========================

def free_memory():
    """
    Освобождает память после работы модели.
    """

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# =========================
# VECTOR SEARCH
# =========================

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
        results.append(
            {
                "id": row[0],
                "document_id": row[1],
                "filename": row[2],
                "page": row[3],
                "content": row[4],
                "vector_score": float(row[5])
            }
        )

    return results


# =========================
# EMBEDDING
# =========================

def create_query_embedding(query: str) -> list[float]:

    print("\n[1/4] Загружаем BGE-M3...")

    embedder = EmbeddingService(EMBEDDING_MODEL)

    print("[2/4] Создаём embedding запроса...")

    query_embedding = embedder.embed_query(query)

    # BGE больше не нужен
    del embedder

    print("[2/4] Выгружаем BGE-M3...")

    free_memory()

    return query_embedding


# =========================
# RERANKING
# =========================

def rerank_results(
    query: str,
    candidates: list[dict]
) -> list[dict]:

    print("\n[3/4] Загружаем CrossEncoder...")

    reranker = Reranker()

    print("[3/4] Выполняем reranking...")

    results = reranker.rerank(
        query=query,
        results=candidates,
        top_n=FINAL_TOP_K
    )

    # CrossEncoder больше не нужен
    del reranker

    print("[3/4] Выгружаем CrossEncoder...")

    free_memory()

    return results


# =========================
# LLM
# =========================

def generate_answer(
    query: str,
    results: list[dict]
) -> str:

    print("\n[4/4] Загружаем Qwen3...")

    llm = LocalLLM()

    print("[4/4] Генерируем ответ...")

    answer = llm.answer(
        query=query,
        results=results
    )

    # Qwen после ответа больше не нужен
    del llm

    free_memory()

    return answer


# =========================
# FULL RAG
# =========================

def search(
    query: str,
    vector_top_k: int = VECTOR_TOP_K,
    final_top_k: int = FINAL_TOP_K
) -> tuple[str, list[dict]]:

    # -------------------------
    # 1. BGE-M3
    # -------------------------

    query_embedding = create_query_embedding(query)

    # -------------------------
    # 2. pgvector
    # -------------------------

    print("\n[2/4] Ищем документы в PostgreSQL...")

    candidates = vector_search(
        query_embedding=query_embedding,
        top_k=vector_top_k
    )

    # Embedding больше не нужен
    del query_embedding

    free_memory()

    if not candidates:
        return (
            "В предоставленных документах недостаточно информации для ответа.",
            []
        )

    print(f"[2/4] Найдено кандидатов: {len(candidates)}")

    # -------------------------
    # 3. CrossEncoder
    # -------------------------

    results = rerank_results(
        query=query,
        candidates=candidates
    )

    # Кандидаты больше не нужны
    del candidates

    free_memory()

    print(f"[3/4] Отобрано лучших чанков: {len(results)}")

    # -------------------------
    # 4. Qwen
    # -------------------------

    answer = generate_answer(
        query=query,
        results=results
    )

    return answer, results


# =========================
# CONSOLE
# =========================

def main():

    print("=" * 80)
    print("LOCAL RAG")
    print("=" * 80)

    while True:

        query = input(
            "\nВведите вопрос (или 'exit' для выхода): "
        ).strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        try:

            answer, results = search(query)

            print()
            print("=" * 80)
            print("ОТВЕТ")
            print("=" * 80)

            print(answer)

            print()
            print("=" * 80)
            print("ИСПОЛЬЗОВАННЫЕ ЧАНКИ")
            print("=" * 80)

            for i, result in enumerate(results, start=1):

                print()
                print(f"[{i}] {result['filename']}")

                if result["page"] is not None:
                    print(f"Страница: {result['page']}")

                print(
                    f"Vector score: "
                    f"{result['vector_score']:.4f}"
                )

                print(
                    f"Rerank score: "
                    f"{result['rerank_score']:.4f}"
                )

                print("-" * 80)
                print(result["content"])

        except Exception as error:

            print()
            print("=" * 80)
            print("ОШИБКА")
            print("=" * 80)
            print(error)
            print("=" * 80)


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    main()