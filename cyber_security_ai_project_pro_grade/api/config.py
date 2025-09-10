import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    KAFKA_BOOTSTRAP: str = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
    POSTGRES_DSN: str = os.getenv("POSTGRES_DSN", "postgresql://threat:threatpass@localhost:5432/threatdb")
    JWT_SECRET: str = os.getenv("JWT_SECRET", "change-me")
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    OIDC_ISSUER: str = os.getenv("OIDC_ISSUER", "")
    OIDC_INTROSPECT: str = os.getenv("OIDC_INTROSPECT", "")
    OIDC_CLIENT_ID: str = os.getenv("OIDC_CLIENT_ID", "")
    OIDC_CLIENT_SECRET: str = os.getenv("OIDC_CLIENT_SECRET", "")
    METRICS_PATH: str = "/metrics"
settings = Settings()
