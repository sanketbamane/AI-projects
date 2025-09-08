import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
from .schemas import UploadResponse, QueryIn
from .celery_app import celery
from .tasks import ingest_and_index_task
from .db import SessionLocal
from .models import Chunk, Document
from .embedders import Embedder
from .vectorstore import VectorStore
from .rerank_and_qa import rerank, extractive_answer
from .utils import build_prompt

app = FastAPI(title="RAG Assistant - Industrial Skeleton")

@app.post("/upload", response_model=UploadResponse)
async def upload(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    job_id = uuid.uuid4().hex
    for f in files:
        raw = await f.read()
        _ = ingest_and_index_task.delay(raw, f.filename)
    return {"job_id": job_id, "indexed_chunks_estimate": -1}

@app.post("/query")
def query(q: QueryIn):
    q_text = q.question.strip()
    if not q_text:
        raise HTTPException(status_code=400, detail="question required")
    embedder = Embedder(model_name=os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2"),
                        use_openai=os.environ.get("USE_OPENAI_EMB", "false").lower() in ("1","true","yes"))
    q_emb = embedder.embed_texts([q_text])
    vs = VectorStore(dim=q_emb.shape[1])
    hits = vs.query(q_emb, top_k=q.top_k)
    if not hits:
        return {"answer": "No documents indexed.", "sources": []}
    db = SessionLocal()
    try:
        candidates = []
        for h in hits:
            chunk_id = h["id"]
            chunk = db.get(Chunk, chunk_id)
            if chunk:
                candidates.append({"id": chunk.id, "text": chunk.text, "metadata": {"title": chunk.document.title if chunk.document else None, "source": chunk.document.source if chunk.document else None}, "doc_id": chunk.doc_id, "score": h["score"]})
    finally:
        db.close()
    reranked = rerank(q_text, candidates, top_n=q.rerank_top_n)
    if q.gen and os.environ.get("OPENAI_API_KEY"):
        prompt_chunks = [{"id": c["id"], "text": c["text"], "metadata": c.get("metadata", {})} for c in reranked]
        prompt = build_prompt(q_text, prompt_chunks)
        import openai
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        resp = openai.ChatCompletion.create(model=os.environ.get("GEN_MODEL", "gpt-4o-mini"),
                                            messages=[{"role": "system", "content": prompt}],
                                            max_tokens=512, temperature=0.0)
        answer = resp["choices"][0]["message"]["content"].strip()
        sources = [{"id": c["id"], "title": c.get("metadata", {}).get("title"), "score": c.get("_rerank_score", 0.0)} for c in reranked]
        return {"answer": answer, "sources": sources}
    answer_text, evidence = extractive_answer(q_text, reranked, top_k_sentences=2)
    sources = [{"id": c["id"], "title": c.get("metadata", {}).get("title"), "score": c.get("_rerank_score", 0.0)} for c in reranked]
    return {"answer": answer_text, "sources": sources}
