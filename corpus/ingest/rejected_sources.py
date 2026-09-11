"""
Rejected / non-flagship public sources.

These are synthetic or testbed-only and must NOT be presented as the
AOMB flagship reference corpus. Kept here as an explicit denylist for docs
and future adapter registration checks.
"""

REJECTED_PUBLIC_SOURCES = [
    {
        "id": "smithclay/otel-demo-telemetry",
        "reason": "OTel Demo capture — demo/testbed, not production flagship",
    },
    {
        "id": "open-telemetry/opentelemetry-demo",
        "reason": "Synthetic demo microservices",
    },
    {
        "id": "opentelemetry-tracegen",
        "reason": "Synthetic trace generator",
    },
    {
        "id": "sock-shop+chaos-mesh zenodo testbeds",
        "reason": "Chaos/testbed workloads, not production telemetry",
    },
    {
        "id": "DeathStarBench packs",
        "reason": "Benchmark/testbed packs, not production flagship",
    },
    {
        "id": "generate_observability_corpus.py",
        "reason": "Synthetic smoke/CI only",
    },
]
