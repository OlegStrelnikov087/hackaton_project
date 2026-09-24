import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from search import (
    search,
    VECTOR_TOP_K,
    FINAL_TOP_K,
)


st.set_page_config(
    page_title="Corporate RAG",
    page_icon="📄",
    layout="wide",
)


st.title("Corporate RAG")

st.caption(
    "Поиск и ответы по внутренним корпоративным документам"
)


# ==========================================================
# DOCUMENTS
# ==========================================================

with st.sidebar:
    st.header("Документы")

    uploaded_files = st.file_uploader(
        "Загрузите документы",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        upload_dir = (
            PROJECT_ROOT
            / "backend"
            / "documents"
            / "uploads"
        )

        upload_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        upload_progress = st.progress(
            0,
            text="Сохраняю документы...",
        )

        total_files = len(uploaded_files)

        for index, uploaded_file in enumerate(
            uploaded_files,
            start=1,
        ):
            file_path = upload_dir / uploaded_file.name

            with open(
                file_path,
                "wb",
            ) as file:
                file.write(
                    uploaded_file.getbuffer()
                )

            percent = int(
                index / total_files * 100
            )

            upload_progress.progress(
                percent,
                text=(
                    f"Сохраняю документ "
                    f"{index} из {total_files}..."
                ),
            )

        upload_progress.progress(
            100,
            text="Документы загружены.",
        )

        st.success(
            f"Загружено файлов: {total_files}"
        )


# ==========================================================
# CHAT
# ==========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


query = st.chat_input(
    "Задайте вопрос по корпоративным документам..."
)


if query:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        st.markdown("Обработка запроса")

        progress_bar = st.progress(
            0,
            text="Начинаю обработку...",
        )

        process_text = st.empty()

        def update_progress(percent, message):
            progress_bar.progress(
                percent,
                text=message,
            )

            process_text.markdown(
                f"**{message}**"
            )

        try:
            answer, results = search(
                query=query,
                vector_top_k=VECTOR_TOP_K,
                final_top_k=FINAL_TOP_K,
                progress_callback=update_progress,
            )

        except Exception as error:
            progress_bar.empty()
            process_text.empty()

            st.error(str(error))
            st.stop()

        progress_bar.progress(
            100,
            text="Готово.",
        )

        process_text.markdown(
            "**Ответ сформирован.**"
        )

        st.divider()

        st.markdown("### Ответ")
        st.markdown(answer)

        # ==================================================
        # SOURCES
        # ==================================================

        if results:
            st.markdown("### Источники")

            for index, result in enumerate(
                results,
                start=1,
            ):
                filename = result.get(
                    "filename",
                    "Неизвестный файл",
                )

                page = result.get("page")

                if page is not None:
                    st.write(
                        f"{index}. {filename}, "
                        f"страница {page}"
                    )
                else:
                    st.write(
                        f"{index}. {filename}"
                    )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )
