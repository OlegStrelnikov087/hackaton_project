import os

os.environ["HF_HOME"] = r"D:\hf_cache"
os.environ["HF_HUB_CACHE"] = r"D:\hf_cache\hub"

from sentence_transformers import CrossEncoder


MODEL_NAME = "BAAI/bge-reranker-v2-m3"


class Reranker:
    def __init__(self):
        self.model = CrossEncoder(
            MODEL_NAME,
            device="cpu"
        )

    def rerank(
        self,
        query: str,
        results: list[dict],
        top_n: int = 5
    ) -> list[dict]:

        pairs = [
            (query, result["content"])
            for result in results
        ]

        scores = self.model.predict(
            pairs,
            show_progress_bar=False
        )

        reranked = []

        for result, score in zip(results, scores):
            reranked.append({
                **result,
                "rerank_score": float(score)
            })

        reranked.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return reranked[:top_n]