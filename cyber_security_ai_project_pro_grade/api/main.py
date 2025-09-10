import os, json, time, requests
from fastapi import FastAPI, Depends, HTTPException, Header, Request
from api.schemas import IngestRequest, PredictResponse
from api.utils import logger, INCOMING, PROCESSED, PREDICTION_LATENCY
from api import kafka_producer, db, auth, oidc
from starlette_exporter import PrometheusMiddleware, handle_metrics
from api.opentelemetry_config import init_tracing

app = FastAPI(title='Threat Detection API')
app.add_middleware(PrometheusMiddleware)
app.add_route('/metrics', handle_metrics)

init_tracing('threat-api', otlp_endpoint=os.getenv('OTLP_ENDPOINT'))

db.init_db()

def get_current_user(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail='Missing authorization header')
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != 'bearer':
        raise HTTPException(status_code=401, detail='Invalid auth header')
    token = parts[1]
    sub = oidc.verify_oidc_token(token) or auth.verify_token(token)
    if not sub:
        raise HTTPException(status_code=401, detail='Invalid token')
    return sub

@app.get('/healthz')
def health():
    return {'status': 'ok'}

@app.post('/ingest', response_model=dict)
async def ingest_log(req: IngestRequest, user: str = Depends(get_current_user)):
    INCOMING.inc()
    payload = json.dumps({'timestamp': req.timestamp, 'source': req.source, 'message': req.message})
    kafka_producer.send_log('logs', payload)
    logger.info('ingested_log', user=user, source=req.source)
    return {'accepted': True}

@app.post('/predict', response_model=PredictResponse)
def predict_live(req: IngestRequest, user: str = Depends(get_current_user)):
    with PREDICTION_LATENCY.time():
        resp = requests.post(os.getenv('MODEL_SERVICE_URL', 'http://model-service:8501/predict'), json={'message': req.message, 'source': req.source}, timeout=5)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail='model-service error')
        data = resp.json()
        explanation = None
        if data.get('label') == 1:
            # request explanation from model-service
            try:
                er = requests.post(os.getenv('MODEL_SERVICE_URL', 'http://model-service:8501') + '/explain', json={'message': req.message, 'source': req.source}, timeout=5)
                if er.status_code == 200:
                    explanation = er.text
            except Exception as e:
                logger.error('explain_error', error=str(e))
            with db.engine.begin() as conn:
                conn.execute(db.alerts.insert().values(timestamp=req.timestamp, source=req.source, message=req.message, score=data['score'], label=1, explanation=explanation))
            PROCESSED.inc()
        return PredictResponse(score=data['score'], label=data['label'])
