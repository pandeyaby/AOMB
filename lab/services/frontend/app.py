"""Lab frontend — proxies to API and emits its own OTel spans."""

from __future__ import annotations

import logging
import os
import urllib.request

from flask import Flask, jsonify, Response

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.flask import FlaskInstrumentor
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
except ImportError:  # pragma: no cover
    trace = None  # type: ignore

SERVICE = os.environ.get("SERVICE_NAME", "frontend")
API_URL = os.environ.get("API_URL", "http://localhost:8081").rstrip("/")
OTLP = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318").rstrip("/")


def setup_otel(app: Flask) -> None:
    if trace is None:
        return
    resource = Resource.create({SERVICE_NAME: SERVICE})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{OTLP}/v1/traces"))
    )
    trace.set_tracer_provider(provider)
    FlaskInstrumentor().instrument_app(app)


def create_app() -> Flask:
    app = Flask(__name__)
    logging.basicConfig(level=logging.INFO)
    setup_otel(app)
    tracer = trace.get_tracer(__name__) if trace else None

    @app.get("/")
    def index():
        html = """<!doctype html><html><body>
        <h1>AOMB Lab Store</h1>
        <p><a href="/shop/catalog">Catalog</a> · <a href="/shop/checkout">Checkout</a></p>
        </body></html>"""
        return Response(html, mimetype="text/html")

    def _proxy(path: str):
        url = f"{API_URL}{path}"
        if tracer:
            with tracer.start_as_current_span(f"proxy {path}"):
                with urllib.request.urlopen(url, timeout=10) as resp:
                    body = resp.read()
                    return Response(body, status=resp.status, content_type="application/json")
        with urllib.request.urlopen(url, timeout=10) as resp:
            body = resp.read()
            return Response(body, status=resp.status, content_type="application/json")

    @app.get("/shop/catalog")
    def catalog():
        try:
            return _proxy("/api/catalog")
        except Exception as exc:  # noqa: BLE001
            return jsonify({"ok": False, "error": str(exc)}), 502

    @app.get("/shop/checkout")
    def checkout():
        try:
            return _proxy("/api/checkout")
        except Exception as exc:  # noqa: BLE001
            return jsonify({"ok": False, "error": str(exc)}), 502

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "service": SERVICE})

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
