from pydantic import BaseModel
from typing import List, Optional

class UploadResponse(BaseModel):
    job_id: str
    indexed_chunks_estimate: int

class QueryIn(BaseModel):
    question: str
    top_k: int = 10
    rerank_top_n: int = 5
    gen: bool = False

class SourceOut(BaseModel):
    id: str
    title: Optional[str]
    score: float

class QueryOut(BaseModel):
    answer: str
    sources: List[SourceOut]
