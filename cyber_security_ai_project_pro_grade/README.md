# AI Threat Detection — Complete Local Stack (Scaffold)

This repo provides a runnable local scaffold that includes:
- FastAPI API service with OIDC (Keycloak) introspection fallback, OpenTelemetry scaffold, Prometheus metrics
- Kafka + Zookeeper for async ingestion
- Postgres for alerts storage (includes explanation column)
- Model service with `/predict` and `/explain` endpoints using TF-IDF + RandomForest and fast surrogate explanations
- MLflow tracking server (local filesystem backend) and UI
- Keycloak as OIDC provider (dev mode) for testing OIDC flows
- Prometheus + Grafana for observability (Prometheus scrapes API metrics)

Quick start (local dev, requires Docker & docker-compose):

1. Build & start everything:
   ```bash
   docker compose up --build
   ```
2. Create training data and train a model:
   ```bash
   docker compose exec api python data/generate_logs.py --n 2000 --out data/logs.csv
   docker compose exec api python ml/train.py --input data/logs.csv --output ml/model.joblib
   ```
   This will also log a run to MLflow (MLflow UI available at http://localhost:5000).
3. Use API:
   - Keycloak admin UI: http://localhost:8080 (admin/keycloak)
   - MLflow UI: http://localhost:5000
   - Grafana: http://localhost:3000 (admin:admin)
   - Prometheus: http://localhost:9090
   - Threat API: http://localhost:8000 (use bearer token from Keycloak or local JWT)

Notes:
- This is a scaffold for development/testing. Before production, secure secrets, enable TLS, use a persistent model registry and object store, and tune SHAP explainability for text.
- For explanation we use a fast surrogate: mapping RandomForest feature importances to TF-IDF feature names.

Generated: 2025-09-10T05:30:55.992639Z
