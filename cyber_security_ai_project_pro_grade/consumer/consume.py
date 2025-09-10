import json, time, requests
from confluent_kafka import Consumer
from api.config import settings
from joblib import load
from api.db import engine, alerts
from api.utils import logger, PROCESSED, PREDICTION_LATENCY

c = Consumer({
    'bootstrap.servers': settings.KAFKA_BOOTSTRAP,
    'group.id': 'threat-consumer-group',
    'auto.offset.reset': 'earliest'
})
c.subscribe(['logs'])

MODEL_PATH = 'ml/model.joblib'
try:
    pipeline = load(MODEL_PATH)
except Exception as e:
    pipeline = None
    logger.error('no_model_loaded', error=str(e))

def run():
    while True:
        msg = c.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            logger.error('kafka_error', error=msg.error())
            continue
        data = json.loads(msg.value().decode('utf-8'))
        text = f"[{data.get('source')}] " + data.get('message', '')
        if pipeline:
            with PREDICTION_LATENCY.time():
                prob = pipeline.predict_proba([text])[0,1]
            label = int(prob >= 0.5)
        else:
            try:
                resp = requests.post('http://model-service:8501/predict', json={'message': data.get('message'), 'source': data.get('source')}, timeout=5)
                d = resp.json()
                prob = d.get('score', 0.0); label = d.get('label', 0)
            except Exception as e:
                logger.error('model_service_error', error=str(e)); prob=0.0; label=0
        explanation = None
        if label == 1:
            # try to fetch explanation
            try:
                er = requests.post('http://model-service:8501/explain', json={'message': data.get('message'), 'source': data.get('source')}, timeout=5)
                if er.status_code == 200:
                    explanation = er.text
            except Exception as e:
                logger.error('explain_error', error=str(e))
            with engine.begin() as conn:
                conn.execute(alerts.insert().values(timestamp=data.get('timestamp'), source=data.get('source'), message=data.get('message'), score=prob, label=label, explanation=explanation))
        PROCESSED.inc()

if __name__ == '__main__':
    run()
