import requests


MODEL_NAME = "qwen3:4b-instruct"
OLLAMA_URL = "http://localhost:11434/api/generate"


class LocalLLM:

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name

    def build_context(self, results: list[dict]) -> str:
        context_parts = []

        for i, result in enumerate(results, start=1):
            filename = result.get("filename", "Неизвестный файл")
            page = result.get("page")
            content = result.get("content", "")

            if page is not None:
                source = f"{filename}, стр. {page}"
            else:
                source = filename

            context_parts.append(
                f"[Источник {i}: {source}]\n{content}"
            )

        return "\n\n".join(context_parts)

    def answer(self, query: str, results: list[dict]) -> str:

        context = self.build_context(results)

        prompt = f"""
Ты — корпоративный ассистент, работающий с внутренними документами.

Твоя задача — отвечать на вопрос пользователя только на основании
предоставленного контекста.

ПРАВИЛА:

1. Отвечай только на основании предоставленного контекста.
2. Не придумывай факты, которых нет в контексте.
3. Не используй внешние знания и интернет.
4. Если информации недостаточно, напиши:
"В предоставленных документах недостаточно информации для ответа."
5. Отвечай на русском языке.
6. Отвечай кратко, точно и по существу.
7. Для важных утверждений указывай источник в формате [Источник N].
8. В конце добавь раздел "Источники".
9. В разделе "Источники" указывай только использованные источники.

КОНТЕКСТ:

{context}

ВОПРОС:

{query}

Сформируй итоговый ответ.
"""

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_ctx": 4096,
                "num_predict": 512
            }
        }

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=300
        )

        response.raise_for_status()

        data = response.json()

        return data["response"]
