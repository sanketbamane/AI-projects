FROM python:3.10-slim AS builder
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
FROM python:3.10-slim
COPY --from=builder /usr/local /usr/local
WORKDIR /app
COPY . /app
EXPOSE 8000 8501
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "api.main:app", "--bind", "0.0.0.0:8000", "--workers", "2"]
