# Dynamic Route Optimization - Full Industrial Project

This repository contains a production-ready starter for dynamic route optimization with:
- FastAPI backend with async Postgres persistence (SQLAlchemy + asyncpg).
- Celery worker for async optimization (Redis broker).
- OR-Tools optimizer with greedy fallback.
- Prometheus metrics endpoint.
- Minimal React UI.
- Docker Compose for local integration.

## Run locally (development)
1. Build & start services:
   docker-compose up --build
2. Wait for Postgres to initialize, then visit:
   - API: http://localhost:8000
   - UI: Serve ui via `npm start` in ui/ (separate)
3. To trigger sample request:
   curl -X POST http://localhost:8000/optimize -H "Content-Type: application/json" --data @data/sample_requests.json

## Notes
- This repo focuses on architecture and structure. For production hardening, add secrets management, TLS, monitoring, autoscaling, and more robust error handling.