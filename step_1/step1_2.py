import os
from sentence_transformers import SentenceTransformer
from mistralai import Mistral
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
model = SentenceTransformer("sberbank-ai/sbert_large_mt_nlu_ru")
tokenizer = model.tokenizer
chunks = []


def get_chunks(path):
    chunks = []
    filename = os.path.basename(path)
    with open(path, mode='r', encoding='utf-8') as f:
        current_chunk = ''
        chunk_start = None
        cursor = 0

        for line in f.readlines():
            line_len = len(line)
            if line == '\n':
                cursor += line_len
                continue

            if all([line[0].isdigit(), line[1] == '.']):
                if current_chunk and all([current_chunk[0].isdigit(), current_chunk[1] == '.']):
                    chunks.append({
                        "id": f"{filename}::chunk_{len(chunks)}",
                        "text": current_chunk.strip('\n'),
                        "source": filename,
                        "chunk_index": len(chunks),
                        "char_start": chunk_start,
                        "char_end": cursor,
                        "token_count": len(tokenizer.encode(current_chunk)),
                    })

                current_chunk = line
                chunk_start = cursor
            else:
                current_chunk += line

            cursor += line_len

        # последний чанк
        if current_chunk:
            chunks.append({
                "id": f"{filename}::chunk_{len(chunks)}",
                "text": current_chunk.strip('\n'),
                "source": filename,
                "chunk_index": len(chunks),
                "char_start": chunk_start,
                "char_end": cursor,
                "token_count": len(tokenizer.encode(current_chunk)),
            })
    return chunks



def get_answer(promt: str, context=None):
    key = 'e8vQnlZ9jgk7YQ5v1M2nIqioS7t38bXj'
    messages=[{"content": promt, "role": "user"}]
    if context:
        messages.append({"role": "system", "content": f"Для ответа на вопросы бери только информацию из контекста и в своём ответе указывай название файла и пункт. Начало контекста <{context}>Конец контекста"})
    with Mistral(api_key=key) as mistral:
        response = mistral.chat.complete(
            model="mistral-small-latest",
            messages=messages,
            stream=False,
        )
        return response.choices[0].message.content


# chunks = get_chunks(os.path.join(BASE_DIR, 'docs', 'rules.txt'))

for filename in os.listdir(DOCS_DIR):
    if filename.endswith(".txt"):
        file_path = os.path.join(DOCS_DIR, filename)
        chunks.extend(get_chunks(file_path))


texts = [chunk["text"] for chunk in chunks]
if os.path.exists('embeddings.npy'): 
    embeddings = np.load('embeddings.npy') 
else:
    embeddings = model.encode(texts, normalize_embeddings=True)
    np.save('embeddings.npy', embeddings)


question = "Можно ли сегодня забирать наколенники? если да, то где?"
question_embedding = model.encode(
    [question],
    normalize_embeddings=True
)[0]

similarities = cosine_similarity(
    [question_embedding],
    embeddings
)[0]

top_n_indices = np.argsort(similarities)[-10:][::-1]

context = "\n".join([
    f"[{chunks[idx]['source']}::{chunks[idx]['chunk_index']}] {chunks[idx]['text']}"
    for idx in top_n_indices
])


answer = get_answer(
    promt='Можно ли сегодня забирать наколенники? если да, то где?',
    context=context
)

print(answer)
