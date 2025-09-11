from celery import Celery
from app.config import CELERY_BROKER, CELERY_BACKEND

celery = Celery("drom", broker=CELERY_BROKER, backend=CELERY_BACKEND)
celery.conf.task_routes = {"app.tasks.optimize_async": {"queue": "optimization"}}
celery.conf.worker_prefetch_multiplier = 1
celery.conf.task_acks_late = True