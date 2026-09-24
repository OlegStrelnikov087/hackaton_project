import json
from pathlib import Path
import os

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


os.environ["HF_HUB_OFFLINE"] = "1"

MODEL_NAME = "BAAI/bge-m3"

INPUT_FILE = "backend/output/chunks.json"
OUTPUT_FILE = "backend/output/embedded_chunks.json"

BATCH_SIZE = 16


class EmbeddingService:
    def __init__(self, model_name: str = MODEL_NAME):
        # Используем GPU, если CUDA доступна
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Device: {self.device}")

        self.model = SentenceTransformer(
            model_name,
            device=self.device
        )

    def embed_chunks(
        self,
        chunks: list[dict]
    ) -> list[dict]:

        # Достаём текст каждого chunk
        texts = [
            chunk["content"]
            for chunk in chunks
        ]

        print(f"Chunks: {len(texts)}")
        print("Creating embeddings...")

        embeddings = self.model.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        # Добавляем embedding обратно в каждый chunk
        result = []

        for chunk, embedding in zip(chunks, embeddings):

            chunk_with_embedding = {
                **chunk,
                "embedding": embedding.tolist()
            }

            result.append(chunk_with_embedding)

        return result

    def embed_query(self, query: str) -> list[float]:
        """
        Создаёт embedding для пользовательского запроса.
        """

        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        return embedding.tolist()


def load_chunks(path: str) -> list[dict]:
    """
    Загружает chunks из JSON.
    """

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            "JSON должен содержать список chunks."
        )

    for i, chunk in enumerate(data):

        if not isinstance(chunk, dict):
            raise ValueError(
                f"Chunk #{i} должен быть объектом."
            )

        if "content" not in chunk:
            raise ValueError(
                f"Chunk #{i} не содержит поле 'content'."
            )

    return data


def save_chunks(
    chunks: list[dict],
    path: str
) -> None:
    """
    Сохраняет chunks с embeddings в JSON.
    """

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )


def main():
    input_path = Path(INPUT_FILE)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Файл {INPUT_FILE} не найден."
        )

    # Загружаем chunks
    chunks = load_chunks(INPUT_FILE)

    # Загружаем модель
    embedder = EmbeddingService()

    # Создаём embeddings
    embedded_chunks = embedder.embed_chunks(chunks)

    # Проверяем размерность
    embedding_size = len(
        embedded_chunks[0]["embedding"]
    )

    print(
        f"\nEmbedding dimension: {embedding_size}"
    )

    # Сохраняем результат
    save_chunks(
        embedded_chunks,
        OUTPUT_FILE
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )

    # Показываем первый chunk
    print("\nFirst chunk:")
    print(
        json.dumps(
            embedded_chunks[0],
            ensure_ascii=False,
            indent=2
        )
    )


if __name__ == "__main__":
    main()