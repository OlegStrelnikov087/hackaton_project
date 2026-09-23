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

# Если на странице меньше символов — запускаем OCR
MIN_TEXT_LENGTH = 30


# ============================================================
# 1. OCR (Tesseract)
# ============================================================

def ocr_page(page):
    """
    Распознаёт страницу PDF через OCR.
    """
    try:
        import pytesseract
        from PIL import Image

        pix = page.get_pixmap(
            matrix=pymupdf.Matrix(2, 2),
            alpha=False
        )

        image = Image.frombytes(
            "RGB",
            [pix.width, pix.height],
            pix.samples
        )

        text = pytesseract.image_to_string(
            image,
            lang="rus+eng"
        )

        return text.strip()

    except Exception as e:
        print(f"[WARNING] OCR не сработал: {e}")
        return ""


# ============================================================
# 2. Обработка Таблиц
# ============================================================

def table_to_text(table_data):
    """
    Преобразует структуру таблицы в связный текст.
    """
    if not table_data:
        return ""

    rows = []
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
        print(f"[WARNING] Не удалось извлечь таблицу: {e}")

    return tables_result


# ============================================================
# 3. Нормализация и очистка текста
# ============================================================

def clean_text(text):
    """
    Чистит неразрывные пробелы (\xa0), переносы строк и спецсимволы,
    которые вызывают рваные слова в PDF.
    """
    if text is None:
        return ""

    text = str(text)

    # 1. Заменяем неразрывные пробелы (\xa0) и невидимые символы
    text = re.sub(r'[\xa0\u200b\u200e\u200f]', ' ', text)

    # 2. Нормализуем переносы и табуляции
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", " ")

    # 3. Чистим разделители мусора (=====)
    text = re.sub(r"={3,}", " ", text)

    # 4. Убираем дублирующиеся пробелы
    text = re.sub(r"[ ]+", " ", text)

    # 5. Оставляем не более двух переносов строки подряд (для границ абзацев)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# 4. Умное контекстное чанкование (БЕЗ обрезки слов)
# ============================================================

def _find_best_split_point(text, max_pos):
    """
    Ищет идеальное место для разреза текста слева от max_pos:
    1. Граница абзаца (\n\n)
    2. Граница строки (\n)
    3. Конец предложения (. ! ?)
    4. Обычный пробел (гарантия не разрезать слово)
    """
    if max_pos >= len(text):
        return len(text)

    search_window = text[:max_pos]

    # 1. Ищем абзац
    paragraph_match = list(re.finditer(r'\n\n', search_window))
    if paragraph_match:
        return paragraph_match[-1].end()

    # 2. Ищем перенос строки
    newline_match = list(re.finditer(r'\n', search_window))
    if newline_match:
        return newline_match[-1].end()

    # 3. Ищем конец предложения (. ! ? с последующим пробелом)
    sentence_match = list(re.finditer(r'[.!?]\s+', search_window))
    if sentence_match:
        return sentence_match[-1].end()

    # 4. Фолбэк: ищем любой пробельный символ (слово НЕ режется)
    space_match = list(re.finditer(r'\s+', search_window))
    if space_match:
        return space_match[-1].start()

    # Если пробелов вообще нет в окне (одно гигантское слово)
    return max_pos


def split_text_by_context(
    text,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
):
    """
    Разбивает текст по смысловым блокам, гарантируя отсутствие
    обрезанных слов на концах и стыках чанков.
    """
    chunks = []
    if not text:
        return chunks

    start = 0
    text_len = len(text)

    while start < text_len:
        # Если остаток текста меньше размера чанка — забираем целиком
        if start + chunk_size >= text_len:
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Ищем наилучшую контекстную точку разрыва в пределах chunk_size
        end = start + _find_best_split_point(text[start:], chunk_size)

        # Страховка от зацикливания
        if end <= start:
            end = start + chunk_size

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Рассчитываем старт следующего чанка с учетом overlap
        new_start = end - chunk_overlap

        if new_start > start and new_start < text_len:
            # Корректируем начало overlap-чанка по пробелу ВПРАВО,
            # чтобы не отрезать половину слова в НАЧАЛЕ чанка
            match_space = re.search(r'\s+', text[new_start:end])
            if match_space:
                start = new_start + match_space.end()
            else:
                start = new_start
        else:
            start = end

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

    # Запускаем контекстное чанкование
    text_chunks = split_text_by_context(content)

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

        text = page.get_text()
        text = clean_text(text)

        # Если текста слишком мало — пробуем OCR
        if len(text) < MIN_TEXT_LENGTH:
            print("  → мало текста, запускаем OCR")
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

        elif text:
            chunks.extend(
                create_chunks_from_content(
                    document_id=document_id,
                    filename=filename,
                    page=page_number,
                    content=text
                )
            )

        # Извлечение таблиц
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

        if not os.path.isfile(file_path):
            continue

        if filename.lower().endswith(".pdf"):
            print(f"\nPDF: {filename}")
            pdf_chunks = process_pdf(
                file_path=file_path,
                document_id=document_id,
                filename=filename
            )
            chunks.extend(pdf_chunks)
            document_id += 1

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

output_dir.mkdir(exist_ok=True)
output_file = output_dir / "chunks.json"

if not documents_dir.exists():
    raise FileNotFoundError(
        f"Папка с документами не найдена: {documents_dir}"
    )

chunks = load_documents(documents_dir)

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

print("\n" + "=" * 50)
print(f"JSON сохранён: {output_file}")
print(f"Всего чанков: {len(chunks)}")
print("=" * 50)