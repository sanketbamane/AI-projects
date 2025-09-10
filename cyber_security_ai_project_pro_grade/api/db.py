from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Table, MetaData, Text
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from api.config import settings

engine = create_engine(settings.POSTGRES_DSN, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

metadata = MetaData()

alerts = Table(
    'alerts', metadata,
    Column('id', Integer, primary_key=True),
    Column('timestamp', String, nullable=False),
    Column('source', String, nullable=True),
    Column('message', String, nullable=False),
    Column('score', Float, nullable=False),
    Column('label', Integer, nullable=False),
    Column('explanation', Text, nullable=True)
)

def init_db():
    metadata.create_all(bind=engine)
