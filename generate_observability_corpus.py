"""
generate_observability_corpus.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generates synthetic Cisco/Splunk/AppDynamics observability telemetry
as parquet shards compatible with autoresearch-macos prepare.py.

Each shard is a parquet file with a single "text" column, where every
row is one coherent observability "document" (a session of correlated
events: a business transaction trace, a cascade failure, a network
event burst, etc.).

Run once before `python prepare.py` and `uv run train.py`.

Output: ~/.cache/autoresearch/data/shard_NNNNN.parquet
  shard_00000 – shard_00019  : training shards (20 × 1000 docs each)
  shard_06542                : pinned validation shard (1000 docs)
"""

import os
import math
import random
import hashlib
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta

# ── paths (must match prepare.py constants) ─────────────────────────────────
CACHE_DIR    = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR     = os.path.join(CACHE_DIR, "data")
VAL_SHARD    = 6542          # pinned val shard index (matches prepare.py)
NUM_TRAIN_SHARDS = 20        # how many training shards to generate
DOCS_PER_SHARD   = 1000      # coherent sessions per shard

random.seed(42)
os.makedirs(DATA_DIR, exist_ok=True)

# ── Domain vocabulary ────────────────────────────────────────────────────────

SERVICES = [
    "payment-gateway", "auth-service", "user-api", "inventory-db",
    "recommendation-engine", "notification-svc", "checkout-flow",
    "search-index", "order-processor", "analytics-pipeline",
    "fraud-detector", "cart-service", "shipping-estimator", "tax-calculator",
    "session-manager", "cdn-edge", "api-gateway", "queue-worker",
    "llm-inference", "embedding-service", "vector-db", "prompt-router",
    "cost-tracker", "pii-scanner", "policy-enforcer", "rate-limiter",
]
HOSTS = (
    [f"web-{i:02d}" for i in range(1, 20)]
    + [f"db-{i:02d}" for i in range(1, 10)]
    + [f"gpu-node-{i:02d}" for i in range(1, 8)]
    + [f"cache-{i:02d}" for i in range(1, 6)]
    + [f"k8s-worker-{i:02d}" for i in range(1, 12)]
)
AWS_REGIONS  = ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1", "eu-central-1"]
BRANCHES     = [f"branch-{i:02d}" for i in range(1, 16)]
QOS_CLASSES  = ["ef", "af41", "af31", "cs3", "cs1", "be"]
HTTP_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH"]
HTTP_ERRORS  = [400, 401, 403, 404, 408, 429, 500, 502, 503, 504]
LOG_LEVELS   = ["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"]
SPAN_OPS     = [
    "http.request", "db.query", "db.write", "db.transaction",
    "cache.get", "cache.set", "rpc.call", "queue.publish",
    "queue.consume", "auth.validate", "llm.completion", "embedding.encode",
]
MPLS_PATHS   = ["mpls→inet", "inet→mpls", "mpls→mpls", "inet→inet", "4g→mpls", "lte→inet"]
ANOMALY_KINDS = [
    "timeout", "oom", "circuit_open", "rate_limited", "connection_refused",
    "ssl_handshake_fail", "dns_nxdomain", "memory_leak", "thread_pool_exhausted",
]

# ── Helpers ──────────────────────────────────────────────────────────────────

def _tid():
    return hashlib.md5(str(random.random()).encode()).hexdigest()[:16]

def _sid():
    return hashlib.md5(str(random.random()).encode()).hexdigest()[:8]

def _ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"

def _normal_latency() -> int:
    """Lognormal, P50≈80ms P99≈420ms."""
    return max(5, int(math.exp(random.gauss(4.4, 0.6))))

def _spike_latency() -> int:
    return random.randint(1800, 32000)

# ── Per-source event generators ───────────────────────────────────────────────

def _appd_normal(ts, trace_id, span_id) -> str:
    svc = random.choice(SERVICES)
    lat = _normal_latency()
    tok_in = random.randint(20, 512)
    tok_out = random.randint(10, tok_in)
    return (
        f"[ts={ts}] [src=AppD] [svc={svc}] "
        f"latency_ms={lat} error=none trace_id={trace_id} span_id={span_id} "
        f"http_status=200 http_method={random.choice(HTTP_METHODS)} "
        f"gpu_util={random.uniform(0.05, 0.65):.2f} "
        f"prompt_tokens={tok_in} response_tokens={tok_out} "
        f"cost_usd={tok_in*0.000003:.6f} drift_score={random.uniform(0, 0.12):.3f}"
    )

def _appd_anomaly(ts, trace_id, span_id) -> str:
    svc = random.choice(SERVICES)
    lat = _spike_latency()
    kind = random.choice(ANOMALY_KINDS)
    drift = random.uniform(0.65, 0.99)
    gpu = random.uniform(0.88, 0.99)
    tok_in = random.randint(256, 4096)
    return (
        f"[ts={ts}] [src=AppD] [svc={svc}] "
        f"latency_ms={lat} error={kind} trace_id={trace_id} span_id={span_id} "
        f"http_status={random.choice(HTTP_ERRORS)} http_method={random.choice(HTTP_METHODS)} "
        f"gpu_util={gpu:.2f} "
        f"prompt_tokens={tok_in} response_tokens=0 "
        f"cost_usd={tok_in*0.000003:.6f} drift_score={drift:.3f}"
    )

def _thousandeyes_normal(ts) -> str:
    region = random.choice(AWS_REGIONS)
    return (
        f"[ts={ts}] [src=ThousandEyes] path=internet→aws-{region} "
        f"latency_ms={random.randint(10, 200)} jitter_ms={random.randint(1, 8)} "
        f"packet_loss={random.uniform(0, 0.001):.4f} bgp_changes=0 "
        f"dns_ms={random.randint(5, 40)} hop_count={random.randint(7, 14)} "
        f"mtu=1500 reachable=true"
    )

def _thousandeyes_anomaly(ts) -> str:
    region = random.choice(AWS_REGIONS)
    return (
        f"[ts={ts}] [src=ThousandEyes] path=internet→aws-{region} "
        f"latency_ms={random.randint(800, 8000)} jitter_ms={random.randint(50, 800)} "
        f"packet_loss={random.uniform(0.05, 0.45):.4f} bgp_changes={random.randint(1, 12)} "
        f"dns_ms={random.randint(200, 5000)} hop_count={random.randint(18, 32)} "
        f"mtu={random.choice([576, 1400, 1500])} reachable={random.choice(['true', 'false'])}"
    )

def _splunk_normal(ts) -> str:
    host = random.choice(HOSTS)
    svc = random.choice(SERVICES)
    lvl = random.choices(LOG_LEVELS, weights=[30, 50, 15, 4, 1])[0]
    lat = _normal_latency()
    msgs = {
        "DEBUG": "handler_entered",
        "INFO": "request_completed",
        "WARN": "cache_miss_rate_elevated",
        "ERROR": "upstream_timeout",
        "CRITICAL": "circuit_breaker_open",
    }
    return (
        f"[ts={ts}] [src=Splunk] host={host} level={lvl} svc={svc} "
        f"msg={msgs[lvl]} latency_ms={lat} session_id={_sid()} "
        f"user_id=usr_{random.randint(10000, 99999)}"
    )

def _splunk_anomaly(ts) -> str:
    host = random.choice(HOSTS)
    svc = random.choice(SERVICES)
    kind = random.choice([
        "token_expired", "pii_leak_detected", "policy_violation",
        "cost_explosion", "oom_kill", "disk_full", "llm_jailbreak_attempt",
        "prompt_injection_detected", "data_exfil_blocked", "api_key_exposed",
    ])
    return (
        f"[ts={ts}] [src=Splunk] host={host} level=CRITICAL svc={svc} "
        f"msg={kind} latency_ms={_spike_latency()} session_id={_sid()} "
        f"alert=true escalated=true pagerduty=triggered "
        f"user_id=usr_{random.randint(10000, 99999)}"
    )

def _otel_span(ts, trace_id, span_id, parent) -> str:
    svc = random.choice(SERVICES)
    op = random.choice(SPAN_OPS)
    anomalous = random.random() < 0.05
    dur = _spike_latency() if anomalous else _normal_latency()
    st = "error" if (dur > 1000 or anomalous) else "ok"
    return (
        f"[ts={ts}] [src=OTel] trace_id={trace_id} span_id={span_id} parent={parent} "
        f"op={op} svc={svc} duration_ms={dur} status={st} "
        f"db={'postgres' if 'db' in op else 'n/a'} "
        f"rows={random.randint(0, 5000) if 'db' in op else 0} "
        f"k8s_pod={svc}-{_sid()} k8s_ns=prod"
    )

def _cisco_sdwan_normal(ts) -> str:
    site = random.choice(BRANCHES)
    return (
        f"[ts={ts}] [src=CiscoSDWAN] site={site} link={random.choice(MPLS_PATHS)} "
        f"qos={random.choice(QOS_CLASSES)} latency_ms={random.randint(10, 80)} "
        f"loss=0.000 jitter_ms={random.randint(1, 5)} "
        f"bw_util={random.uniform(0.10, 0.75):.3f} policy=active "
        f"interface=ge0/{random.randint(0, 3)}"
    )

def _cisco_sdwan_anomaly(ts) -> str:
    site = random.choice(BRANCHES)
    return (
        f"[ts={ts}] [src=CiscoSDWAN] site={site} link={random.choice(MPLS_PATHS)} "
        f"qos=be latency_ms={random.randint(500, 5000)} "
        f"loss={random.uniform(0.05, 0.40):.3f} jitter_ms={random.randint(80, 600)} "
        f"bw_util={random.uniform(0.92, 1.00):.3f} policy=VIOLATED "
        f"alert=link_degraded interface=ge0/{random.randint(0, 3)}"
    )

def _appdynamics_biz_txn(ts, trace_id) -> str:
    """AppDynamics-style business transaction health snapshot."""
    svc = random.choice(SERVICES)
    health = random.choices(["NORMAL", "SLOW", "VERY_SLOW", "STALL", "ERROR"],
                            weights=[70, 15, 8, 4, 3])[0]
    baseline = _normal_latency()
    actual = {
        "NORMAL": baseline,
        "SLOW": int(baseline * random.uniform(2, 4)),
        "VERY_SLOW": int(baseline * random.uniform(4, 10)),
        "STALL": int(baseline * random.uniform(10, 30)),
        "ERROR": _spike_latency(),
    }[health]
    return (
        f"[ts={ts}] [src=AppD-BizTxn] trace_id={trace_id} svc={svc} "
        f"txn=/{svc}/api/v2/{random.choice(['checkout','search','auth','recommend'])} "
        f"health={health} response_time_ms={actual} baseline_ms={baseline} "
        f"call_count={random.randint(1, 500)} error_pct={random.uniform(0, 0.05 if health=='NORMAL' else 0.8):.3f} "
        f"slow_pct={random.uniform(0, 0.1 if health=='NORMAL' else 0.9):.3f}"
    )

# ── Session generator ─────────────────────────────────────────────────────────

# Source weights: AppD:25%, TE:15%, Splunk:25%, OTel:20%, Cisco:10%, AppD-BizTxn:5%
SRC_CHOICES  = ["AppD", "TE", "Splunk", "OTel", "Cisco", "BizTxn"]
SRC_WEIGHTS  = [25, 15, 25, 20, 10, 5]

def generate_session(base_dt: datetime, anomalous: bool, cascade: bool = False) -> str:
    """
    Generate a coherent observability session.
    Returns a multi-line string (the "document").
    Normal sessions: 8–25 events, mostly healthy.
    Anomalous sessions: 10–40 events, showing failure patterns.
    Cascade sessions: 20–60 events, multi-service failure propagation.
    """
    trace_id = _tid()
    spans    = [_tid() for _ in range(random.randint(3, 12))]
    n_events = random.randint(20, 60) if cascade else (
               random.randint(10, 40) if anomalous else
               random.randint(8, 25))

    lines = []
    dt = base_dt

    # Cascade: anomaly starts after a few normal events
    cascade_trigger = random.randint(3, 6) if cascade else 9999

    for i in range(n_events):
        dt += timedelta(milliseconds=random.uniform(50, 2000))
        ts = _ts(dt)
        span_id = random.choice(spans)
        parent  = random.choice(spans)
        src     = random.choices(SRC_CHOICES, weights=SRC_WEIGHTS)[0]

        is_bad = anomalous or (cascade and i >= cascade_trigger)

        if src == "AppD":
            lines.append(_appd_anomaly(ts, trace_id, span_id) if is_bad
                         else _appd_normal(ts, trace_id, span_id))
        elif src == "TE":
            lines.append(_thousandeyes_anomaly(ts) if is_bad
                         else _thousandeyes_normal(ts))
        elif src == "Splunk":
            lines.append(_splunk_anomaly(ts) if is_bad
                         else _splunk_normal(ts))
        elif src == "OTel":
            lines.append(_otel_span(ts, trace_id, span_id, parent))
        elif src == "Cisco":
            lines.append(_cisco_sdwan_anomaly(ts) if is_bad
                         else _cisco_sdwan_normal(ts))
        elif src == "BizTxn":
            lines.append(_appdynamics_biz_txn(ts, trace_id))

    return "\n".join(lines)


def generate_shard(n_docs: int, anomaly_rate: float = 0.06,
                   cascade_rate: float = 0.03) -> list[str]:
    """Generate n_docs session documents for one shard."""
    docs = []
    base = datetime(2026, 1, 1, 0, 0, 0)
    for _ in range(n_docs):
        base += timedelta(seconds=random.uniform(1, 30))
        p = random.random()
        cascade  = p < cascade_rate
        anomalous = p < cascade_rate + anomaly_rate
        docs.append(generate_session(base, anomalous, cascade))
    return docs


def write_parquet_shard(shard_index: int, docs: list[str]) -> None:
    filename = f"shard_{shard_index:05d}.parquet"
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"  {filename} already exists, skipping")
        return
    table = pa.table({"text": pa.array(docs, type=pa.string())})
    pq.write_table(table, filepath, compression="snappy")
    size_mb = os.path.getsize(filepath) / 1e6
    print(f"  {filename}: {len(docs)} docs, {size_mb:.2f} MB")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Autonomous Observability Model Breeder")
    print("Generating synthetic telemetry corpus...")
    print(f"Output: {DATA_DIR}")
    print("=" * 60)

    # Training shards
    print(f"\nGenerating {NUM_TRAIN_SHARDS} training shards × {DOCS_PER_SHARD} docs each...")
    for i in range(NUM_TRAIN_SHARDS):
        docs = generate_shard(DOCS_PER_SHARD, anomaly_rate=0.06, cascade_rate=0.03)
        write_parquet_shard(i, docs)

    # Pinned validation shard
    print(f"\nGenerating pinned val shard (shard_{VAL_SHARD:05d})...")
    random.seed(99999)  # fixed seed for reproducible val
    val_docs = generate_shard(DOCS_PER_SHARD, anomaly_rate=0.06, cascade_rate=0.03)
    write_parquet_shard(VAL_SHARD, val_docs)

    # Summary
    total_docs = (NUM_TRAIN_SHARDS + 1) * DOCS_PER_SHARD
    total_bytes = sum(
        os.path.getsize(os.path.join(DATA_DIR, f))
        for f in os.listdir(DATA_DIR) if f.endswith(".parquet")
    )
    print(f"\nDone.")
    print(f"  Total documents : {total_docs:,}")
    print(f"  Total size      : {total_bytes/1e6:.1f} MB")
    print(f"\nNext steps:")
    print("  1. uv run python prepare.py --num-shards 20")
    print("  2. uv run train.py  (smoke test)")
    print("  3. Start the agent loop")
