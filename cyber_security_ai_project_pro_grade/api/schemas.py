from pydantic import BaseModel

class IngestRequest(BaseModel):
    timestamp: str
    source: str
    message: str

class PredictResponse(BaseModel):
    score: float
    label: int
