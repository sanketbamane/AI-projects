import os

POSTGRES_DSN = os.getenv("POSTGRES_DSN", "postgresql+asyncpg://postgres:postgres@postgres:5432/drom")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
CELERY_BROKER = os.getenv("CELERY_BROKER", REDIS_URL)
CELERY_BACKEND = os.getenv("CELERY_BACKEND", REDIS_URL)