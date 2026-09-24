import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from search import search


st.set_page_config(
    page_title="Corporate RAG",
    page_icon="📄",
    layout="wide"
)


st.title("📄 Corporate RAG")
st.caption("Поиск по внутренним корпоративным документам")


if "messages" not in st.session_state:
    st.session_state.messages = []


# Вывод предыдущих сообщений
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            results = message.get("results", [])

            if results:
                with st.expander("📚 Использованные источники"):

                    for i, result in enumerate(results, start=1):

                        filename = result.get(
                            "filename",
                            "Неизвестный файл"
                        )

                        page = result.get("page")

                        if page is not None:
                            source = f"{filename}, страница {page}"
                        else:
                            source = filename

                        st.markdown(
                            f"**[Источник {i}] {source}**"
                        )

                        st.caption(
                            f"Vector score: "
                            f"{result.get('vector_score', 0):.4f} | "
                            f"Rerank score: "
                            f"{result.get('rerank_score', 0):.4f}"
                        )

                        st.write(
                            result.get("content", "")
                        )

                        if i < len(results):
                            st.divider()


# Поле ввода
query = st.chat_input(
    "Введите вопрос по документам..."
)


if query:

    # Сообщение пользователя
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)


    # Генерация ответа
    with st.chat_message("assistant"):

        with st.spinner("Ищу информацию в документах..."):

            try:

                answer, results = search(query)

                st.markdown(answer)


                # Источники
                if results:

                    with st.expander(
                        "📚 Использованные источники"
                    ):

                        for i, result in enumerate(
                            results,
                            start=1
                        ):

                            filename = result.get(
                                "filename",
                                "Неизвестный файл"
                            )

                            page = result.get("page")

                            if page is not None:
                                source = (
                                    f"{filename}, "
                                    f"страница {page}"
                                )
                            else:
                                source = filename

                            st.markdown(
                                f"**[Источник {i}] {source}**"
                            )

                            st.caption(
                                f"Vector score: "
                                f"{result.get('vector_score', 0):.4f} | "
                                f"Rerank score: "
                                f"{result.get('rerank_score', 0):.4f}"
                            )

                            st.write(
                                result.get("content", "")
                            )

                            if i < len(results):
                                st.divider()


                # Сохраняем ответ
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "results": results
                })


            except Exception as error:

                error_message = (
                    "❌ Ошибка при обработке запроса:\n\n"
                    f"`{error}`"
                )

                st.error(error_message)