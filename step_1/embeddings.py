import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from chunking import get_chunks
import os
from mistral import get_answer


embeddings = np.load('embeddings.npy')
model = SentenceTransformer('sberbank-ai/sbert_large_mt_nlu_ru')
question = "В каком спорте основное время составляет 80 минут"  # Замените на ваш вопрос
question_embedding = model.encode([question])[0]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
chunks = get_chunks(os.path.join(BASE_DIR, 'docs', 'rules.txt'))

# Вычисляем сходство между вопросом и всеми чанками
similarities = cosine_similarity(
    [question_embedding],
    embeddings
)[0]

# # Находим индекс чанка с максимальным сходством
# most_similar_idx = np.argmax(similarities)
# most_similar_chunk = chunks[most_similar_idx]

# print(most_similar_chunk)

top_n_indices = np.argsort(similarities)[-10:][::-1]  # Берем 5 самых больших значений

# Выводим топ-5 чанков

context = ''.join([chunks[idx] for idx in top_n_indices])

answer = get_answer(
    promt='В каком спорте основное время составляет 80 минут',
    context=context
)

print(answer)
