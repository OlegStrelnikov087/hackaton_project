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
# PROGRESS BAR STYLE
# ==========================================================

st.markdown(
    """
    <style>
    .rag-progress {
        width: 100%;
        height: 10px;
        background: #e6e6e6;
        border-radius: 999px;
        overflow: hidden;
        margin: 8px 0 14px 0;
    }

    .rag-progress-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(
            110deg,
            #1f9cff 0%,
            #67c4ff 35%,
            #ffffff 50%,
            #67c4ff 65%,
            #1f9cff 100%
        );
        background-size: 200% 100%;
        animation: rag-shimmer 1.4s linear infinite;
        transition: width 0.35s ease;
    }

    @keyframes rag-shimmer {
        from {
            background-position: 200% 0;
        }
        to {
            background-position: -200% 0;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_progress(container, percent, message):
    percent = max(0, min(100, int(percent)))

    container.markdown(
        f"""
        <div style="font-weight: 600; margin-bottom: 6px;">
            {message}
        </div>
        <div class="rag-progress">
            <div
                class="rag-progress-fill"
                style="width: {percent}%;">
            </div>
        </div>
        """,
        unsafe_allow_html=True,
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

        progress_container = st.empty()

        def update_progress(percent, message):
            render_progress(
                progress_container,
                percent,
                message,
            )

        render_progress(
            progress_container,
            0,
            "Начинаю обработку...",
        )

        try:
            answer, results = search(
                query=query,
                vector_top_k=VECTOR_TOP_K,
                final_top_k=FINAL_TOP_K,
                progress_callback=update_progress,
            )

        except Exception as error:
            progress_container.empty()
            st.error(str(error))
            st.stop()

        render_progress(
            progress_container,
            100,
            "Готово.",
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
