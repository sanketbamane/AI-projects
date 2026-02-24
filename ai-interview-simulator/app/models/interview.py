from sqlalchemy import Column, String, Integer, ForeignKey, Float, DateTime, Text
from sqlalchemy.sql import func
from app.core.database import Base
import uuid

class InterviewSession(Base):
    __tablename__ = "interview_sessions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"))
    role_type = Column(String)
    difficulty_level = Column(Integer, default=1)
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)

class InterviewAnswer(Base):
    __tablename__ = "interview_answers"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey("interview_sessions.id"))
    question_text = Column(Text)
    transcript = Column(Text)
    semantic_score = Column(Float)
    rubric_score = Column(Float)
    confidence_score = Column(Float)
    behavioral_score = Column(Float)
