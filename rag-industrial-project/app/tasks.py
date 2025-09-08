import os
from .celery_app import celery
from .ingest import ingest_file_to_chunks
from .db import SessionLocal, engine, Base
from .models import Document, Chunk
from .embedders import Embedder
from .vectorstore import VectorStore
import tempfile
import json

Base.metadata.create_all(bind=engine)

@celery.task(bind=True, name="app.tasks.ingest_and_index_task")
def ingest_and_index_task(self, file_bytes: bytes, filename: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=filename) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        tmp_path = tmp.name

    records = ingest_file_to_chunks(tmp_path, title=filename)

    db = SessionLocal()
    try:
        if not records:
            return {"status": "no_chunks"}
        doc_id = records[0]["doc_id"]
        doc = Document(id=doc_id, title=records[0]["metadata"]["title"], source=records[0]["metadata"].get("source"))
        db.add(doc)
        db.commit()
        for rec in records:
            chunk = Chunk(id=rec["id"], doc_id=rec["doc_id"], chunk_index=rec["metadata"]["chunk_index"], text=rec["text"])
            db.add(chunk)
        db.commit()
    finally:
        db.close()

    embedder = Embedder(model_name=os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2"),
                        use_openai=os.environ.get("USE_OPENAI_EMB", "false").lower() in ("1","true","yes"))
    texts = [r["text"] for r in records]
    ids = [r["id"] for r in records]
    metas = [r["metadata"] for r in records]
    vectors = embedder.embed_texts(texts)

    vs = VectorStore(dim=vectors.shape[1])
    vs.upsert(ids=ids, vectors=vectors, metas=metas)

    try:
        os.remove(tmp_path)
    except Exception:
        pass

    return {"status": "ok", "indexed": len(records)}
