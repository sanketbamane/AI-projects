from fastapi import FastAPI
from app.api import interview

app = FastAPI(title="AI Interview Simulator")

app.include_router(interview.router, prefix="/interview")
