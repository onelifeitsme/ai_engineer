import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from chunking import get_chunks
import os
from mistral import get_answer


MODEL_NAME = 'sberbank-ai/sbert_large_mt_nlu_ru'
model = SentenceTransformer(MODEL_NAME)
EMBEDDINGS_FILE = 'embeddings.npy'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
chunks = get_chunks(os.path.join(BASE_DIR, 'docs', 'rules.txt'))

if os.path.exists(EMBEDDINGS_FILE): 
    embeddings = np.load(EMBEDDINGS_FILE) 
else:
    embeddings = model.encode(chunks) 
    np.save(EMBEDDINGS_FILE, embeddings)


embeddings = np.load('embeddings.npy')
model = SentenceTransformer('sberbank-ai/sbert_large_mt_nlu_ru')
question = "В каком спорте основное время составляет 80 минут"  # Замените на ваш вопрос
question_embedding = model.encode([question])[0]




# Вычисляем сходство между вопросом и всеми чанками
similarities = cosine_similarity(
    [question_embedding],
    embeddings
)[0]

top_n_indices = np.argsort(similarities)[-10:][::-1]  # Берем 10 самых больших значений

# Выводим топ-5 чанков

context = ''.join([chunks[idx] for idx in top_n_indices])

answer = get_answer(
    promt='В каком спорте основное время составляет 80 минут',
    context=context
)

print(answer)
