from pydantic import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://admin:admin@localhost:5432/ai_interview"
    OPENAI_API_KEY: str = ""
    JWT_SECRET: str = "supersecret"
    REDIS_URL: str = "redis://localhost:6379/0"
    
    class Config:
        env_file = ".env"

settings = Settings()
