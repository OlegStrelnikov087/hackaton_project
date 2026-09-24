import requests


MODEL_NAME = "qwen3:4b-instruct"
OLLAMA_URL = "http://localhost:11434/api/generate"
CONTEXT_SIZE = 4096


class LocalLLM:

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name

    def build_context(self, results: list[dict]) -> str:
        context_parts = []

        for index, result in enumerate(
            results,
            start=1,
        ):
            filename = result.get(
                "filename",
                "Неизвестный файл",
            )

            page = result.get("page")

            content = result.get(
                "content",
                "",
            )

            if page is not None:
                source = f"{filename}, стр. {page}"
            else:
                source = filename

            context_parts.append(
                f"[Источник {index}: {source}]\n{content}"
            )

        return "\n\n".join(context_parts)

    def answer(
        self,
        query: str,
        results: list[dict],
    ) -> str:

        context = self.build_context(results)

        prompt = f"""
Ты — корпоративный ассистент,
работающий с внутренними корпоративными документами.

Отвечай на вопрос пользователя только на основании
предоставленного контекста.

Правила:
1. Не придумывай факты.
2. Не используй интернет.
3. Не используй внешние знания.
4. Если информации недостаточно, напиши:
"В предоставленных документах недостаточно информации для ответа."
5. Отвечай на русском языке.
6. Отвечай кратко и по существу.
7. Для важных утверждений указывай источник в формате [Источник N].
8. В конце добавь раздел "Источники".
9. Указывай только реально использованные источники.
10. Не выдумывай страницы или номера источников.

Контекст:

{context}

Вопрос:

{query}

Сформируй итоговый ответ.
"""

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_ctx": CONTEXT_SIZE,
                "num_predict": 512,
            },
        }

        try:
            response = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=300,
            )

            response.raise_for_status()

            data = response.json()

            if "response" not in data:
                raise RuntimeError(
                    "Ollama не вернул поле 'response'."
                )

            return data["response"]

        except requests.exceptions.ConnectionError as error:
            raise RuntimeError(
                "Не удалось подключиться к Ollama. "
                "Убедитесь, что Ollama запущен."
            ) from error

        except requests.exceptions.Timeout as error:
            raise RuntimeError(
                "Ollama слишком долго генерирует ответ."
            ) from error

        except requests.exceptions.HTTPError as error:
            raise RuntimeError(
                f"Ollama вернул HTTP ошибку: {response.status_code}"
            ) from error

        except requests.exceptions.RequestException as error:
            raise RuntimeError(
                f"Ошибка HTTP при обращении к Ollama: {error}"
            ) from error

        except Exception as error:
            raise RuntimeError(
                f"Ошибка локальной LLM: {error}"
            ) from error
