from sentence_transformers import SentenceTransformer


def create_embeddings(chanks):
    # Загрузите модель (будет скачана автоматически при первом запуске)
    model = SentenceTransformer('sberbank-ai/sbert_large_mt_nlu_ru')
    # Получение эмбедингов для чанков
    embeddings = model.encode(chanks)
    return embeddings
