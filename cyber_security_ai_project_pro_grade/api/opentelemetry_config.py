from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

def init_tracing(service_name='service', otlp_endpoint=None):
    resource = Resource.create({'service.name': service_name})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    if otlp_endpoint:
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)))
    trace.set_tracer_provider(provider)
    return trace.get_tracer(__name__)
