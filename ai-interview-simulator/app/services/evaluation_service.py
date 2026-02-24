import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.services.embedding_service import get_embedding

def semantic_score(candidate_answer: str, ideal_answer: str):
    emb1 = np.array(get_embedding(candidate_answer)).reshape(1, -1)
    emb2 = np.array(get_embedding(ideal_answer)).reshape(1, -1)
    score = cosine_similarity(emb1, emb2)[0][0]
    return float(score * 100)
