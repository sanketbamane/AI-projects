from sqlalchemy import Column, Integer, String, JSON, DateTime, func
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class OptimizationRequest(Base):
    __tablename__ = "optimization_requests"
    id = Column(Integer, primary_key=True)
    status = Column(String, default="pending")
    request = Column(JSON)
    solution = Column(JSON, nullable=True)
    created_at = Column(DateTime, server_default=func.now())