from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.models.interview import InterviewSession
from app.services.interview_service import next_question
from app.services.evaluation_service import semantic_score

router = APIRouter()

@router.post("/start")
async def start_interview(user_id: str, db: AsyncSession = Depends(get_db)):
    session = InterviewSession(user_id=user_id, role_type="backend")
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session

@router.post("/next-question")
async def get_question(session_id: str, db: AsyncSession = Depends(get_db)):
    session = await db.get(InterviewSession, session_id)
    question = await next_question(session)
    return {"question": question}

@router.post("/submit-answer")
async def submit_answer(session_id: str, question: str, answer: str):
    score = semantic_score(answer, "Ideal answer placeholder")
    return {"semantic_score": score}
