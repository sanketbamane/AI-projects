from app.celery_app import celery
from planner.route_optimizer import RouteOptimizer
from app.db import AsyncSessionLocal
from app.models import OptimizationRequest
import asyncio
import json
import logging

logger = logging.getLogger("drom.tasks")

@celery.task(bind=True)
def optimize_async(self, request_id: int):
    # NOTE: Celery tasks cannot call async DB code directly in this simplified setup.
    # We'll use a simple synchronous DB update using SQLAlchemy's sync engine for demo purposes.
    from sqlalchemy import create_engine, MetaData, Table
    from sqlalchemy.sql import select, update
    from app.config import POSTGRES_DSN
    sync_dsn = POSTGRES_DSN.replace("asyncpg://", "")
    engine = create_engine(sync_dsn)
    metadata = MetaData(bind=engine)
    requests_table = Table("optimization_requests", metadata, autoload_with=engine)
    conn = engine.connect()
    try:
        res = conn.execute(select([requests_table.c.request]).where(requests_table.c.id==request_id)).fetchone()
        if not res:
            logger.error("Request id not found")
            return
        req = res[0]
        optimizer = RouteOptimizer()
        sol = optimizer.optimize(req)
        conn.execute(update(requests_table).where(requests_table.c.id==request_id).values(status="done", solution=sol))
        logger.info("Optimization finished for %s", request_id)
    finally:
        conn.close()
    return {"status":"done", "request_id": request_id}