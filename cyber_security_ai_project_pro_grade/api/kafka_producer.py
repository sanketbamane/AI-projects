from confluent_kafka import Producer
from api.config import settings

p = Producer({'bootstrap.servers': settings.KAFKA_BOOTSTRAP})

def send_log(topic: str, value: str, key: str = None):
    p.produce(topic, value=value, key=key)
    p.flush()
