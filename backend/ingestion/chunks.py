import os
import re
import json
from pathlib import Path
import pymupdf


# ============================================================
# НАСТРОЙКИ
# ============================================================

CHUNK_SIZE = 500
CHUNK_OVERLAP = 150

# Если страница содержит меньше этого количества символов,
# считаем, что текста может быть недостаточно и нужен OCR.
MIN_TEXT_LENGTH = 30


# ============================================================
# 1. OCR
# ============================================================

def ocr_page(page):
    """
    Распознаёт страницу PDF через OCR.
    Требует установленный Tesseract и пакет pytesseract.
    """

    try:
        import pytesseract
        from PIL import Image

        # Получаем изображение страницы
        pix = page.get_pixmap(
            matrix=pymupdf.Matrix(2, 2),
            alpha=False
        )

        # Преобразуем изображение в PIL
        image = Image.frombytes(
            "RGB",
            [pix.width, pix.height],
            pix.samples
        )

        # OCR
        text = pytesseract.image_to_string(
            image,
            lang="rus+eng"
        )

        return text.strip()

    except Exception as e:
        print(f"[WARNING] OCR не сработал: {e}")
        return ""


# ============================================================
# 2. Таблицы
# ============================================================

def table_to_text(table_data):
    """
    Преобразует таблицу в обычный текст.

    Например:

    [
        ["Материал", "Температура"],
        ["Сталь", "10-20 °C"]
    ]

    превращается в:

    Материал: Сталь; Температура: 10-20 °C.
    """

    if not table_data:
        return ""

    rows = []

    # Первая строка может быть заголовком
    headers = table_data[0]

    headers = [
        clean_text(str(cell)) if cell is not None else ""
        for cell in headers
    ]

    for row in table_data[1:]:

        values = [
            clean_text(str(cell)) if cell is not None else ""
            for cell in row
        ]

        parts = []

        for header, value in zip(headers, values):

            if value:

                if header:
                    parts.append(f"{header}: {value}")
                else:
                    parts.append(value)

        if parts:
            rows.append("; ".join(parts) + ".")

    # Если заголовков нет или таблица странная
    if not rows:

        for row in table_data:

            values = []

            for cell in row:

                if cell is not None:

                    value = clean_text(str(cell))

                    if value:
                        values.append(value)

            if values:
                rows.append("; ".join(values) + ".")

    return " ".join(rows)


def extract_tables(page):
    """
    Извлекает таблицы со страницы PDF.
    Возвращает список текстовых представлений таблиц.
    """

    tables_result = []

    try:

        page_tables = page.find_tables()

        for table_number, table in enumerate(
            page_tables.tables,
            start=1
        ):

            data = table.extract()

            text = table_to_text(data)

            if text:

                tables_result.append({
                    "table_number": table_number,
                    "text": text
                })

    except Exception as e:

        print(
            f"[WARNING] Не удалось извлечь таблицу: {e}"
        )

    return tables_result


# ============================================================
# 3. Очистка текста
# ============================================================

def clean_text(text):

    if text is None:
        return ""

    text = str(text)

    # Windows переносы
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Табуляции
    text = text.replace("\t", " ")

    # Убираем последовательности =====
    text = re.sub(r"={3,}", " ", text)

    # Убираем лишние пробелы
    text = re.sub(r"[ ]+", " ", text)

    # Убираем слишком много пустых строк
    text = re.sub(r"\n{2,}", "\n", text)

    return text.strip()


# ============================================================
# 4. Разбиение текста на chunks
# ============================================================

def split_text(
    text,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
):

    chunks = []

    if not text:
        return chunks

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # Защита от бесконечного цикла
        if end >= len(text):
            break

        start += chunk_size - chunk_overlap

    return chunks


# ============================================================
# 5. Создание чанков
# ============================================================

def create_chunks_from_content(
    document_id,
    filename,
    page,
    content
):

    chunks = []

    content = clean_text(content)

    if not content:
        return chunks

    text_chunks = split_text(content)

    for chunk in text_chunks:

        chunks.append({
            "document_id": document_id,
            "filename": filename,
            "page": page,
            "content": chunk
        })

    return chunks


# ============================================================
# 6. Обработка PDF
# ============================================================

def process_pdf(
    file_path,
    document_id,
    filename
):

    chunks = []

    pdf = pymupdf.open(file_path)

    for page_number, page in enumerate(pdf):

        page_number = page_number + 1

        print(
            f"Обработка {filename}, "
            f"страница {page_number}/{len(pdf)}"
        )

        # ----------------------------------------------------
        # Обычный текст
        # ----------------------------------------------------

        text = page.get_text()

        text = clean_text(text)

        # ----------------------------------------------------
        # Если текста мало — пробуем OCR
        # ----------------------------------------------------

        if len(text) < MIN_TEXT_LENGTH:

            print(
                "  → мало текста, запускаем OCR"
            )

            ocr_text = ocr_page(page)

            if ocr_text:

                text = clean_text(ocr_text)

                chunks.extend(
                    create_chunks_from_content(
                        document_id=document_id,
                        filename=filename,
                        page=page_number,
                        content=text
                    )
                )

        # ----------------------------------------------------
        # Если обычный текст есть
        # ----------------------------------------------------

        elif text:

            chunks.extend(
                create_chunks_from_content(
                    document_id=document_id,
                    filename=filename,
                    page=page_number,
                    content=text
                )
            )

        # ----------------------------------------------------
        # Таблицы
        # ----------------------------------------------------

        tables = extract_tables(page)

        for table in tables:

            table_chunks = create_chunks_from_content(
                document_id=document_id,
                filename=filename,
                page=page_number,
                content=table["text"]
            )

            chunks.extend(table_chunks)

    pdf.close()

    return chunks


# ============================================================
# 7. Обработка TXT
# ============================================================

def process_txt(
    file_path,
    document_id,
    filename
):

    chunks = []

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:

        text = f.read()

    text = clean_text(text)

    if not text:
        return chunks

    chunks.extend(
        create_chunks_from_content(
            document_id=document_id,
            filename=filename,
            page=None,
            content=text
        )
    )

    return chunks


# ============================================================
# 8. Загрузка всех документов
# ============================================================

def load_documents(folder_path):

    chunks = []

    document_id = 1

    for filename in sorted(os.listdir(folder_path)):

        file_path = os.path.join(
            folder_path,
            filename
        )

        # Пропускаем папки
        if not os.path.isfile(file_path):
            continue

        # ----------------------------------------------------
        # PDF
        # ----------------------------------------------------

        if filename.lower().endswith(".pdf"):

            print(f"\nPDF: {filename}")

            pdf_chunks = process_pdf(
                file_path=file_path,
                document_id=document_id,
                filename=filename
            )

            chunks.extend(pdf_chunks)

            document_id += 1

        # ----------------------------------------------------
        # TXT
        # ----------------------------------------------------

        elif filename.lower().endswith(".txt"):

            print(f"\nTXT: {filename}")

            txt_chunks = process_txt(
                file_path=file_path,
                document_id=document_id,
                filename=filename
            )

            chunks.extend(txt_chunks)

            document_id += 1

    return chunks


# ============================================================
# 9. MAIN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

documents_dir = BASE_DIR / "documents"

output_dir = BASE_DIR / "output"

output_dir.mkdir(
    exist_ok=True
)

output_file = output_dir / "chunks.json"


# ------------------------------------------------------------
# Проверка папки documents
# ------------------------------------------------------------

if not documents_dir.exists():

    raise FileNotFoundError(
        f"Папка с документами не найдена: {documents_dir}"
    )


# ------------------------------------------------------------
# Запуск ingestion
# ------------------------------------------------------------

chunks = load_documents(
    documents_dir
)


# ------------------------------------------------------------
# Сохранение JSON
# ------------------------------------------------------------

with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        chunks,
        f,
        ensure_ascii=False,
        indent=4,
        allow_nan=False
    )


# ------------------------------------------------------------
# Статистика
# ------------------------------------------------------------

print("\n" + "=" * 50)

print(
    f"JSON сохранён: {output_file}"
)

print(
    f"Всего чанков: {len(chunks)}"
)

print("=" * 50)