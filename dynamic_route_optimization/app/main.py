from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
from planner.route_optimizer import RouteOptimizer
from app.db import AsyncSessionLocal, init_db
from app.models import OptimizationRequest
from sqlalchemy.future import select
import logging
from app.celery_app import celery
from app.config import POSTGRES_DSN
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import json, asyncio

logger = logging.getLogger("drom.api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Dynamic Route Optimization - Full")

# Metrics
REQUESTS_TOTAL = Counter("drom_requests_total", "Total optimization requests")

class Location(BaseModel):
    id: int
    lat: float
    lon: float
    demand: Optional[int] = 0
    ready_time: Optional[int] = 0
    due_time: Optional[int] = 24*60*60

class Vehicle(BaseModel):
    id: str
    start_index: int
    capacity: int = 100
    max_work_seconds: Optional[int] = 24*60*60

class OptimizeRequest(BaseModel):
    locations: List[Location]
    vehicles: List[Vehicle]
    depot_index: int = 0
    metric: Optional[str] = "distance"
    async_mode: Optional[bool] = True

@app.on_event("startup")
async def startup():
    await init_db()

@app.post("/optimize")
async def optimize(req: OptimizeRequest):
    REQUESTS_TOTAL.inc()
    payload = req.dict()
    async with AsyncSessionLocal() as session:
        db_obj = OptimizationRequest(status="pending", request=payload)
        session.add(db_obj)
        await session.commit()
        await session.refresh(db_obj)
        request_id = db_obj.id

    if payload.get("async_mode", True):
        # enqueue
        celery.send_task("app.tasks.optimize_async", args=[request_id])
        return {"status":"queued", "request_id": request_id}
    else:
        optimizer = RouteOptimizer()
        sol = optimizer.optimize(payload)
        # update db
        async with AsyncSessionLocal() as session:
            obj = await session.get(OptimizationRequest, request_id)
            obj.status = "done"
            obj.solution = sol
            session.add(obj)
            await session.commit()
        return {"status":"done", "solution": sol}

@app.get("/result/{request_id}")
async def get_result(request_id: int):
    async with AsyncSessionLocal() as session:
        obj = await session.get(OptimizationRequest, request_id)
        if not obj:
            raise HTTPException(status_code=404, detail="Not found")
        return {"id": obj.id, "status": obj.status, "solution": obj.solution}

@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
def health():
    return {"status":"healthy"}