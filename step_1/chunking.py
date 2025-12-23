import os
from sentence_transformers import SentenceTransformer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = SentenceTransformer("sberbank-ai/sbert_large_mt_nlu_ru")
tokenizer = model.tokenizer
chunks = []


def get_chunks(path):
    chunks = []
    with open(path, mode='r', encoding='utf-8') as f:
        current_chunk = ''
        chunk_start = None
        cursor = 0  # абсолютная позиция в файле

        for line in f.readlines():

            line_len = len(line)

            if line == '\n':
                cursor += line_len
                continue

            if all([line[0].isdigit(), line[1] == '.']):

                if current_chunk and all([current_chunk[0].isdigit(), current_chunk[1] == '.']):
                    chunks.append({
                        "id": f"rules.txt::chunk_{len(chunks)}",
                        "text": current_chunk.strip('\n'),
                        "source": "rules.txt",
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

        if current_chunk:
            chunks.append({
                "id": f"rules.txt::chunk_{len(chunks)}",
                "text": current_chunk.strip('\n'),
                "source": "rules.txt",
                "chunk_index": len(chunks),
                "char_start": chunk_start,
                "char_end": cursor,
                "token_count": len(tokenizer.encode(current_chunk)),
            })
    return chunks
