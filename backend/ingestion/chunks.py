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

BASE_DIR = Path(__file__).resolve().parent.parent
output_dir = BASE_DIR / "output"

output_dir.mkdir(exist_ok=True)

output_file = output_dir / "chunks.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=4)

print(f"JSON сохранён: {output_file}")

print(json.dumps(chunks, ensure_ascii=False, indent=4))