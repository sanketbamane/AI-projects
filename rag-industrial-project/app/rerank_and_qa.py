from typing import List, Dict, Any
try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

import nltk
nltk.download('punkt', quiet=True)

_cross_encoder_cache = {}

def get_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    if model_name in _cross_encoder_cache:
        return _cross_encoder_cache[model_name]
    if CrossEncoder is None:
        raise RuntimeError("CrossEncoder not available")
    model = CrossEncoder(model_name)
    _cross_encoder_cache[model_name] = model
    return model

def rerank(question: str, candidates: List[Dict[str, Any]], top_n: int = 5, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    if not candidates:
        return []
    try:
        model = get_cross_encoder(model_name)
        inputs = [[question, c["text"]] for c in candidates]
        scores = model.predict(inputs)
        scored = []
        for c, s in zip(candidates, scores):
            cc = c.copy()
            cc["_rerank_score"] = float(s)
            scored.append(cc)
        scored.sort(key=lambda x: x["_rerank_score"], reverse=True)
        return scored[:top_n]
    except Exception:
        out = []
        for c in candidates[:top_n]:
            c["_rerank_score"] = 0.0
            out.append(c)
        return out

import re
from collections import Counter

def extractive_answer(question: str, top_chunks: List[Dict[str, Any]], top_k_sentences: int = 2):
    def tokenize(s):
        return [t.lower() for t in re.findall(r"\w+", s)]
    q_tokens = tokenize(question)
    q_counter = Counter(q_tokens)

    evidences = []
    for chunk in top_chunks:
        from nltk import sent_tokenize
        sents = sent_tokenize(chunk["text"])
        for sent in sents:
            tokens = tokenize(sent)
            score = sum(q_counter.get(t, 0) for t in tokens)
            score = score + 0.001 * len(tokens)
            evidences.append({
                "doc_id": chunk.get("doc_id"),
                "chunk_id": chunk.get("id"),
                "sentence": sent,
                "score": float(score),
                "metadata": chunk.get("metadata", {})
            })
    evidences.sort(key=lambda x: x["score"], reverse=True)
    if not evidences:
        return ("I couldn't find relevant sentences in the documents.", [])
    top = evidences[:top_k_sentences]
    answer = " ".join([t["sentence"] for t in top])
    for t in top:
        title = t.get("metadata", {}).get("title") or t.get("doc_id")
        t["citation"] = f"[{title}]"
    return (answer, top)
