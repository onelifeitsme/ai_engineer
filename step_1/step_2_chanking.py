# step2_ingest_pdfs.py
import os
import re
from PyPDF2 import PdfReader
import tiktoken
import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PDF_FILES = [
    os.path.join(BASE_DIR, 'docs', "geely.pdf"),
    os.path.join(BASE_DIR, 'docs', "lada.pdf")
]

CHUNK_SIZE_WORDS = 300  # количество слов в чанке


# --- Функции ---
def read_pdf(file_path):
    reader = PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append({"text": text, "page": i+1})
    return pages


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def chunk_text(text, max_words=CHUNK_SIZE_WORDS):
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]


def ingest_pdf(file_path):
    pages = read_pdf(file_path)
    chunks = []
    for page in pages:
        cleaned = clean_text(page["text"])
        page_chunks = chunk_text(cleaned, max_words=CHUNK_SIZE_WORDS)
        for idx, chunk in enumerate(page_chunks):
            chunks.append({
                "text": chunk,
                "metadata": {
                    "source": os.path.basename(file_path),
                    "page": page["page"],
                    "chunk_id": idx + 1
                }
            })
    return chunks


enc = tiktoken.get_encoding("cl100k_base")


def add_tokens_count(chunks):
    for chunk in chunks:
        tokens = enc.encode(chunk["text"])
        chunk["metadata"]["num_tokens"] = len(tokens)
    return chunks


def prepare_for_chroma(chunks):
    ids = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        ids.append(f"chunk-{i}")
        documents.append(chunk["text"])
        metadatas.append(chunk["metadata"])

    return ids, documents, metadatas


if __name__ == "__main__":
    # --- Чтение и разбиение PDF ---
    all_chunks = []
    for pdf in PDF_FILES:
        pdf_chunks = ingest_pdf(pdf)
        all_chunks.extend(pdf_chunks)

    all_chunks = add_tokens_count(all_chunks)

    for i in all_chunks:
        if "отстегнуть ремень безопасности во" in i["text"]:
            pass

    print(f"Всего чанков: {len(all_chunks)}")

    # --- Подготовка данных для Chroma ---
    ids, documents, metadatas = prepare_for_chroma(all_chunks)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("Генерирую embeddings...")
    embeddings = model.encode(documents).tolist()

    # --- Инициализация Chroma ---
    client = chromadb.PersistentClient(path="./chroma_store")

    collection = client.get_or_create_collection(
        name="cars_manuals",
        metadata={"hnsw:space": "cosine"},
        embedding_function=None
    )

    # --- Загрузка в ChromaDB ---
    # collection.add(
    #     ids=ids,
    #     documents=documents,
    #     metadatas=metadatas,
    #     embeddings=embeddings
    # )

    # print("Загружено:", collection.count(), "документов")
    q = "Если отстегнуть ремень безопасности водителя, открыть водительскую дверь или выключить электропитание автомобиля"
    query_emb = model.encode([q]).tolist()

    results = collection.query(
        query_embeddings=query_emb,
        n_results=5
    )

    print(results)
