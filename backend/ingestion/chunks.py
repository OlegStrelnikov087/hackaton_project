import os
import re
import pandas as pd
import json

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_documents(folder_path):

    documents = []

    document_id = 1

    for filename in os.listdir(folder_path):

        file_path = os.path.join(folder_path, filename)

        if filename.lower().endswith(".pdf"):

            pdf = fitz.open(file_path)

            for page_number, page in enumerate(pdf):

                text = page.get_text()

                documents.append({
                    "document_id": document_id,
                    "filename": filename,
                    "page": page_number + 1,
                    "text": text
                })

            pdf.close()

            document_id += 1

        elif filename.lower().endswith(".txt"):

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            documents.append({
                "document_id": document_id,
                "filename": filename,
                "page": None,
                "text": text
            })

            document_id += 1

    return documents

def clean_documents(documents):

    df = pd.DataFrame(documents)

    if df.empty:
        return []

    df["text"] = df["text"].fillna("")

    df["text"] = (
        df["text"]
        .str.replace(r"={3,}", " ", regex=True)
        .str.replace("\r", " ", regex=False)
        .str.replace("\n", " ", regex=False)
        .str.replace("\t", " ", regex=False)
        .str.replace(r"[ ]+", " ", regex=True)
        .str.strip()
    )

    df = df[df["text"].str.len() > 0]

    return df.to_dict("records")

def split_text(text, chunk_size=500, chunk_overlap=150):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - chunk_overlap

    return chunks


def create_chunks(documents):

    chunks = []

    for document in documents:

        text_chunks = split_text(document["text"])

        for chunk in text_chunks:

            chunks.append({
                "document_id": document["document_id"],
                "filename": document["filename"],
                "page": document["page"],
                "section": None,
                "content": chunk
            })

    return chunks




if __name__ == "__main__":

    folder_path = r"D:\VSC_Code\Python"

    # 1. Загрузка
    print("\n[1] Загружаем документы...")

    documents = load_documents(folder_path)

    print(f"Найдено документов/страниц: {len(documents)}")

    if not documents:
        print("❌ Документы не найдены!")
        exit()

    # 2. Очистка
    print("\n[2] Очищаем документы...")

    clean_documents_data = clean_documents(documents)

    print(f"После очистки: {len(clean_documents_data)}")

    if not clean_documents_data:
        print("❌ После очистки не осталось текста!")
        exit()

    # 3. Chunking
    print("\n[3] Создаём chunks...")

    chunks = create_chunks(clean_documents_data)

    print(f"Создано chunks: {len(chunks)}")

    if not chunks:
        print("❌ Chunks не созданы!")
        exit()

    # 4. Показываем первый chunk
    print("\n[4] Первый chunk:")
    print("-" * 60)

    print(f"document_id : {chunks[0]['document_id']}")
    print(f"filename    : {chunks[0]['filename']}")
    print(f"page        : {chunks[0]['page']}")
    print(f"section     : {chunks[0]['section']}")
    print(f"content     : {chunks[0]['content'][:500]}")

    print("-" * 60)

    # 5. Проверяем metadata
    print("\n[5] Проверяем metadata...")

    required_fields = [
        "document_id",
        "filename",
        "page",
        "section",
        "content"
    ]

    for field in required_fields:

        if field in chunks[0]:
            print(f"✅ {field}")
        else:
            print(f"❌ {field}")

print(json.dumps(chunks[:3], ensure_ascii=False, indent=4))