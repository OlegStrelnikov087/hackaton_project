import gc
import importlib.util
from pathlib import Path

import psycopg
import torch
from pgvector import Vector
from pgvector.psycopg import register_vector


PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_ROOT / "backend"

EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
LLM_MODEL = "qwen3:4b-instruct"

VECTOR_TOP_K = 30
FINAL_TOP_K = 5


def load_class_from_backend(filename: str, class_name: str):
    matches = list(BACKEND_DIR.rglob(filename))

    if not matches:
        raise ModuleNotFoundError(
            f"Не найден файл '{filename}' внутри '{BACKEND_DIR}'."
        )

    module_path = matches[0]

    spec = importlib.util.spec_from_file_location(
        f"_rag_{filename.replace('.py', '')}",
        module_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Не удалось загрузить модуль: {module_path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, class_name):
        raise ImportError(
            f"В файле '{module_path}' нет класса '{class_name}'."
        )

    return getattr(module, class_name)


EmbeddingService = load_class_from_backend(
    "embedding.py",
    "EmbeddingService",
)

Reranker = load_class_from_backend(
    "reranker.py",
    "Reranker",
)

LocalLLM = load_class_from_backend(
    "llm.py",
    "LocalLLM",
)


DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/rag"


def progress(callback, percent, message):
    if callback is not None:
        callback(percent, message)


def free_memory():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def vector_search(query_embedding, top_k=30):
    conn = psycopg.connect(DATABASE_URL)

    try:
        register_vector(conn)

        with conn.cursor() as cursor:
            query_vector = Vector(query_embedding)

            cursor.execute(
                """
                SELECT
                    id,
                    document_id,
                    filename,
                    page,
                    content
                FROM chunks
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (
                    query_vector,
                    top_k,
                ),
            )

            rows = cursor.fetchall()

            return [
                {
                    "id": row[0],
                    "document_id": row[1],
                    "filename": row[2],
                    "page": row[3],
                    "content": row[4],
                }
                for row in rows
            ]

    finally:
        conn.close()


def create_query_embedding(query):
    embedder = EmbeddingService(
        model_name=EMBEDDING_MODEL,
    )

    embedding = embedder.embed_query(query)

    del embedder
    free_memory()

    return embedding


def rerank_results(query, candidates, top_n=5):
    reranker = Reranker()

    results = reranker.rerank(
        query=query,
        results=candidates,
        top_n=top_n,
    )

    del reranker
    free_memory()

    return results


def generate_answer(query, results):
    llm = LocalLLM(
        model_name=LLM_MODEL,
    )

    return llm.answer(
        query=query,
        results=results,
    )


def search(
    query,
    vector_top_k=VECTOR_TOP_K,
    final_top_k=FINAL_TOP_K,
    progress_callback=None,
):
    # ------------------------------------------------------
    # 1
    # ------------------------------------------------------

    progress(
        progress_callback,
        5,
        "Принимаю вопрос и определяю, какую информацию нужно найти.",
    )

    # ------------------------------------------------------
    # 2
    # ------------------------------------------------------

    progress(
        progress_callback,
        20,
        "Ищу связанные фрагменты в корпоративных документах.",
    )

    query_embedding = create_query_embedding(query)

    candidates = vector_search(
        query_embedding=query_embedding,
        top_k=vector_top_k,
    )

    if not candidates:
        progress(
            progress_callback,
            100,
            "В документах не найдено подходящей информации.",
        )

        return (
            "В предоставленных документах недостаточно информации для ответа.",
            [],
        )

    # ------------------------------------------------------
    # 3
    # ------------------------------------------------------

    progress(
        progress_callback,
        50,
        "Проверяю найденные фрагменты и выбираю наиболее подходящие.",
    )

    results = rerank_results(
        query=query,
        candidates=candidates,
        top_n=final_top_k,
    )

    # ------------------------------------------------------
    # 4
    # ------------------------------------------------------

    progress(
        progress_callback,
        70,
        "Собираю найденные сведения и связываю их с источниками.",
    )

    # ------------------------------------------------------
    # 5
    # ------------------------------------------------------

    progress(
        progress_callback,
        85,
        "Формирую ответ на основе найденной информации.",
    )

    answer = generate_answer(
        query=query,
        results=results,
    )

    # ------------------------------------------------------
    # 6
    # ------------------------------------------------------

    progress(
        progress_callback,
        95,
        "Проверяю источники, которые использованы в ответе.",
    )

    progress(
        progress_callback,
        100,
        "Готово.",
    )

    return answer, results


def main():
    query = input("Введите запрос: ")

    answer, results = search(query)

    print("\nОтвет:")
    print(answer)

    print("\nИсточники:")

    for result in results:
        filename = result.get(
            "filename",
            "Неизвестный файл",
        )

        page = result.get("page")

        if page is not None:
            print(f"- {filename}, стр. {page}")
        else:
            print(f"- {filename}")


if __name__ == "__main__":
    main()
