from sentence_transformers import SentenceTransformer


def create_embeddings(chanks):
    model = SentenceTransformer('sberbank-ai/sbert_large_mt_nlu_ru')
    embeddings = model.encode(chanks)
    return embeddings
