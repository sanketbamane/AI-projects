import os
import json
import numpy as np

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENVIRONMENT")
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "rag-index")
INDEX_DIR = os.environ.get("INDEX_DIR", "./indexes")

if PINECONE_API_KEY and PINECONE_ENV:
    import pinecone
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    PINECONE_ENABLED = True
else:
    PINECONE_ENABLED = False

try:
    import faiss
except Exception:
    faiss = None

os.makedirs(INDEX_DIR, exist_ok=True)

class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        if PINECONE_ENABLED:
            if INDEX_NAME not in pinecone.list_indexes():
                pinecone.create_index(INDEX_NAME, dimension=dim, metric="cosine")
            self.index = pinecone.Index(INDEX_NAME)
            print("Using Pinecone index:", INDEX_NAME)
        else:
            if faiss is None:
                raise RuntimeError("faiss not installed and Pinecone not configured")
            self.index_path = os.path.join(INDEX_DIR, "faiss.index")
            self.index = faiss.IndexFlatIP(dim)
            self.metadata = []
            meta_path = os.path.join(INDEX_DIR, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)

    def upsert(self, ids: list, vectors: np.ndarray, metas: list):
        if PINECONE_ENABLED:
            to_upsert = [(i, v.tolist(), m) for i, v, m in zip(ids, vectors, metas)]
            self.index.upsert(to_upsert)
        else:
            if vectors.shape[1] != self.dim:
                raise ValueError("Dimension mismatch")
            faiss.normalize_L2(vectors)
            self.index.add(vectors)
            self.metadata.extend(metas)
            meta_path = os.path.join(INDEX_DIR, "meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f)

    def query(self, vector: np.ndarray, top_k: int = 5):
        if PINECONE_ENABLED:
            res = self.index.query(vector.tolist(), top_k=top_k, include_metadata=True, include_values=False)
            matches = res["matches"]
            out = [{"id": m["id"], "score": m["score"], "metadata": m.get("metadata", {})} for m in matches]
            return out
        else:
            faiss.normalize_L2(vector)
            D, I = self.index.search(vector, top_k)
            res = []
            for idx, score in zip(I[0].tolist(), D[0].tolist()):
                if idx < 0 or idx >= len(self.metadata):
                    continue
                meta = self.metadata[idx]
                meta_copy = meta.copy()
                res.append({"id": meta_copy.get("id"), "score": float(score), "metadata": meta_copy})
            return res

    def delete_all(self):
        if PINECONE_ENABLED:
            self.index.delete(deleteAll=True)
        else:
            if hasattr(self, "index_path"):
                self.index = faiss.IndexFlatIP(self.dim)
                self.metadata = []
                meta_path = os.path.join(INDEX_DIR, "meta.json")
                if os.path.exists(meta_path):
                    os.remove(meta_path)
