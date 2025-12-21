
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
chunks = []


with open(os.path.join(BASE_DIR, 'docs', 'rules.txt'), mode='r') as f:
    current_chunk = ''
    for line in f.readlines():
        if line == '\n':
            continue
        if all([line[0].isdigit(), line[1] == '.']):
            if current_chunk and all([current_chunk[0].isdigit(), current_chunk[1] == '.']):
                chunks.append(current_chunk.strip('\n'))
            current_chunk = line
        else:
            current_chunk += line


from sentence_transformers import SentenceTransformer

# Загрузите модель (будет скачана автоматически при первом запуске)
model = SentenceTransformer('sberbank-ai/sbert_large_mt_nlu_ru')

# Получение эмбедингов для чанков
embeddings = model.encode(chunks)

# embeddings — это numpy массив, где каждая строка — вектор для соответствующего чанка
print(embeddings.shape)

import numpy as np

np.save('embeddings.npy', embeddings)

