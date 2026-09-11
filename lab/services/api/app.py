"""
AOMB lab API — small multi-dependency service with OpenTelemetry.

Emits real spans/logs to the collector. Faults are controlled via env:
  FAULT_MODE=none|latency|errors|both
  FAULT_LATENCY_MS=500
  FAULT_ERROR_RATE=0.3
"""

from __future__ import annotations

import logging
import os
import random
import time
from contextlib import contextmanager
from typing import Iterator

from flask import Flask, jsonify, request

# Optional deps — image installs them; degrade gracefully for local unit import
try:
    import psycopg2
except ImportError:  # pragma: no cover
    psycopg2 = None  # type: ignore

try:
    import redis as redis_lib
except ImportError:  # pragma: no cover
    redis_lib = None  # type: ignore

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.flask import FlaskInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
    from opentelemetry.sdk.resources import SERVICE_NAME
except ImportError:  # pragma: no cover
    trace = None  # type: ignore


SERVICE_NAME_ENV = os.environ.get("SERVICE_NAME", "api")
OTLP_ENDPOINT = os.environ.get(
    "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"
).rstrip("/")
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://aomb:aomb@localhost:5433/aomb"
)
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")


def _fault_mode() -> str:
    return os.environ.get("FAULT_MODE", "none").lower()


def _apply_faults() -> None:
    mode = _fault_mode()
    if mode in {"latency", "both"}:
        ms = int(os.environ.get("FAULT_LATENCY_MS", "500"))
        time.sleep(ms / 1000.0)
    if mode in {"errors", "both"}:
        rate = float(os.environ.get("FAULT_ERROR_RATE", "0.3"))
        if random.random() < rate:
            raise RuntimeError("induced_fault_error")


def setup_otel(app: Flask) -> None:
    if trace is None:
        app.logger.warning("OpenTelemetry not installed — running without export")
        return
    resource = Resource.create({SERVICE_NAME: SERVICE_NAME_ENV})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=f"{OTLP_ENDPOINT}/v1/traces")
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    FlaskInstrumentor().instrument_app(app)

    logger_provider = LoggerProvider(resource=resource)
    log_exporter = OTLPLogExporter(endpoint=f"{OTLP_ENDPOINT}/v1/logs")
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
    handler = LoggingHandler(level=logging.INFO, logger_provider=logger_provider)
    logging.getLogger().addHandler(handler)


def create_app() -> Flask:
    app = Flask(__name__)
    logging.basicConfig(level=logging.INFO)
    setup_otel(app)
    tracer = trace.get_tracer(__name__) if trace else None

    @contextmanager
    def span(name: str) -> Iterator[None]:
        if tracer is None:
            yield
            return
        with tracer.start_as_current_span(name):
            yield

    def db_ping() -> str:
        if psycopg2 is None:
            return "psycopg2-missing"
        with span("db.ping"):
            conn = psycopg2.connect(DATABASE_URL)
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            finally:
                conn.close()
        return "ok"

    def cache_incr(key: str = "aomb:hits") -> int:
        if redis_lib is None:
            return -1
        with span("cache.incr"):
            r = redis_lib.from_url(REDIS_URL)
            return int(r.incr(key))

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "service": SERVICE_NAME_ENV})

    @app.get("/api/checkout")
    def checkout():
        app.logger.info("checkout_started")
        try:
            _apply_faults()
            with span("checkout"):
                db = db_ping()
                hits = cache_incr()
            app.logger.info("checkout_ok db=%s hits=%s", db, hits)
            return jsonify(
                {
                    "ok": True,
                    "db": db,
                    "cache_hits": hits,
                    "fault_mode": _fault_mode(),
                }
            )
        except Exception as exc:  # noqa: BLE001 — surface induced faults
            app.logger.error("checkout_failed error=%s", exc)
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.get("/api/catalog")
    def catalog():
        try:
            _apply_faults()
            with span("catalog"):
                db_ping()
            return jsonify(
                {
                    "items": [
                        {"id": 1, "name": "widget"},
                        {"id": 2, "name": "gadget"},
                    ]
                }
            )
        except Exception as exc:  # noqa: BLE001
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.post("/admin/fault")
    def set_fault():
        """Runtime fault switch (also used by inject_faults.sh via env recreate)."""
        body = request.get_json(force=True, silent=True) or {}
        mode = str(body.get("mode", "none"))
        os.environ["FAULT_MODE"] = mode
        if "latency_ms" in body:
            os.environ["FAULT_LATENCY_MS"] = str(body["latency_ms"])
        if "error_rate" in body:
            os.environ["FAULT_ERROR_RATE"] = str(body["error_rate"])
        app.logger.warning("fault_mode_set mode=%s", mode)
        return jsonify({"fault_mode": _fault_mode()})

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
