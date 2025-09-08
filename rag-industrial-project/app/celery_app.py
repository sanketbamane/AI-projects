import os
from celery import Celery

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
CELERY_BROKER = REDIS_URL
CELERY_BACKEND = REDIS_URL

celery = Celery("rag_tasks", broker=CELERY_BROKER, backend=CELERY_BACKEND)
celery.conf.task_routes = {
    'app.tasks.ingest_and_index_task': {'queue': 'ingest_queue'}
}
celery.conf.worker_max_tasks_per_child = 100
