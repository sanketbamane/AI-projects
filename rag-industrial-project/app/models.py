from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from .db import Base

class Document(Base):
    __tablename__ = "documents"
    id = Column(String, primary_key=True, index=True)  # uuid string
    title = Column(String, nullable=False)
    source = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")

class Chunk(Base):
    __tablename__ = "chunks"
    id = Column(String, primary_key=True, index=True)  # uuid string (docid_chunkidx)
    doc_id = Column(String, ForeignKey("documents.id"), index=True)
    chunk_index = Column(Integer)
    text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="chunks")
