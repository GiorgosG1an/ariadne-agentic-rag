"""
Tracing and observability for the Ariadne AI assistant.

This file provides utilities for initializing Phoenix tracing and
instrumenting LlamaIndex with OpenTelemetry.

Author: Georgios Giannopoulos
"""

import logging
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

def init_phoenix_tracing(endpoint: str = "http://localhost:6006/v1/traces") -> None:
    """
    Initialize Phoenix tracing for LlamaIndex instrumentation.

    Args:
        endpoint (str, optional): The OpenTelemetry Protocol (OTLP) endpoint URL
            where traces will be sent. Defaults to "http://localhost:6006/v1/traces"
            (Phoenix local development server).
    """
    
    try:
        tracer_provider: TracerProvider = TracerProvider()
        tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
        
        LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
        print(f"Phoenix Traceability Enabled (Sending to {endpoint})")
        
        logging.getLogger("openinference.instrumentation.llama_index._handler").setLevel(logging.ERROR)
        
    except Exception as e:
        print(f"Tracing Initialization Failed: {e}")
