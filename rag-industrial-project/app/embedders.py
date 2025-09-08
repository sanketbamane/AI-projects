import os
import numpy as np

OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
USE_OPENAI_EMB = os.environ.get("USE_OPENAI_EMB", "false").lower() in ("1", "true", "yes")

if OPENAI_KEY:
    import openai
    openai.api_key = OPENAI_KEY
else:
    openai = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_openai: bool = False):
        self.use_openai = use_openai and bool(openai)
        self.model_name = model_name
        if not self.use_openai:
            if SentenceTransformer is None:
                raise RuntimeError("sentence-transformers not installed")
            self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list) -> np.ndarray:
        if self.use_openai:
            embs = []
            for t in texts:
                resp = openai.Embedding.create(model="text-embedding-3-small", input=t)
                v = np.array(resp["data"][0]["embedding"], dtype="float32")
                embs.append(v)
            return np.vstack(embs)
        else:
            arr = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            if arr.dtype != np.float32:
                arr = arr.astype("float32")
            return arr
