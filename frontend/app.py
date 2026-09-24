import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from search import search


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Corporate RAG",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

DOCUMENTS_DIR = PROJECT_ROOT / "backend" / "documents"
UPLOADS_DIR = DOCUMENTS_DIR / "uploads"

UPLOADS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# HEADER
# ============================================================

st.title("📄 Corporate RAG")
st.caption("Поиск по внутренним корпоративным документам")


# ============================================================
# FILE UPLOAD
# ============================================================

with st.sidebar:

    st.header("📁 Документы")

    uploaded_files = st.file_uploader(
        "Перетащите документы сюда",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
        help="Можно загрузить несколько TXT, PDF и DOCX файлов."
    )

    if uploaded_files:

        st.write(
            f"Выбрано файлов: **{len(uploaded_files)}**"
        )

        if st.button(
            "📥 Загрузить документы",
            use_container_width=True
        ):

            saved_count = 0

            for uploaded_file in uploaded_files:

                file_path = UPLOADS_DIR / uploaded_file.name

                file_path.write_bytes(
                    uploaded_file.getbuffer()
                )

                saved_count += 1

            st.success(
                f"Загружено файлов: {saved_count}"
            )

            st.rerun()


# ============================================================
# SHOW UPLOADED FILES
# ============================================================

with st.sidebar:

    st.divider()

    st.subheader("📚 Загруженные документы")

    existing_files = [
        file
        for file in UPLOADS_DIR.iterdir()
        if file.is_file()
        and file.suffix.lower() in [".txt", ".pdf", ".docx"]
    ]

    if existing_files:

        for file in existing_files:
            st.write(f"📄 {file.name}")

    else:

        st.caption(
            "Документы пока не загружены."
        )


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )

        if message["role"] == "assistant":

            results = message.get(
                "results",
                []
            )

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

                        page = result.get(
                            "page"
                        )

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
                            result.get(
                                "content",
                                ""
                            )
                        )


                        if i < len(results):
                            st.divider()


# ============================================================
# CHAT INPUT
# ============================================================

query = st.chat_input(
    "Введите вопрос по документам..."
)


if query:

    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })


    with st.chat_message("user"):

        st.markdown(query)


    # --------------------------------------------------------
    # ASSISTANT
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Ищу информацию в документах..."
        ):

            try:

                answer, results = search(
                    query
                )


                st.markdown(
                    answer
                )


                # ------------------------------------------------
                # SOURCES
                # ------------------------------------------------

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

                            page = result.get(
                                "page"
                            )


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
                                result.get(
                                    "content",
                                    ""
                                )
                            )


                            if i < len(results):
                                st.divider()


                # ------------------------------------------------
                # SAVE MESSAGE
                # ------------------------------------------------

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


                st.error(
                    error_message
                )