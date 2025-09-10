import structlog, logging
from prometheus_client import Counter, Histogram

logging.basicConfig(level=logging.INFO)
structlog.configure(logger_factory=structlog.stdlib.LoggerFactory())

logger = structlog.get_logger()

INCOMING = Counter('threat_ingest_total', 'Total ingested logs')
PROCESSED = Counter('threat_processed_total', 'Total processed logs')
PREDICTION_LATENCY = Histogram('threat_prediction_seconds', 'Prediction latency seconds')
