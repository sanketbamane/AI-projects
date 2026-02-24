from sqlalchemy import Column, String, Text, ForeignKey
from app.core.database import Base
import uuid

class Resume(Base):
    __tablename__ = "resumes"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"))
    raw_text = Column(Text)
    parsed_json = Column(Text)
    embedding_id = Column(String)
