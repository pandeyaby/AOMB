"""
Fetch public-real OTel Demo telemetry from Hugging Face.

Dataset: smithclay/otel-demo-telemetry (Apache-2.0)
https://huggingface.co/datasets/smithclay/otel-demo-telemetry

Downloads parquet trees into:
  ~/.cache/autoresearch/corpus-v1/public/{otlp_traces,otlp_logs}/...

Does not invent data. If download fails, prints manual steps.
"""

from __future__ import annotations

import argparse
import os
import sys

DEFAULT_OUT = os.path.join(
    os.path.expanduser("~"), ".cache", "autoresearch", "corpus-v1", "public"
)
REPO_ID = "smithclay/otel-demo-telemetry"


def _try_huggingface_hub(out_dir: str, signals: list[str]) -> bool:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return False

    allow: list[str] = ["README.md"]
    for sig in signals:
        if sig == "traces":
            allow.append("otlp_traces/*")
            allow.append("otlp_traces/**")
        elif sig == "logs":
            allow.append("otlp_logs/*")
            allow.append("otlp_logs/**")
        elif sig == "metrics":
            allow.append("otlp_metrics_*/*")
            allow.append("otlp_metrics_*/**")

    print(f"Downloading {REPO_ID} via huggingface_hub → {out_dir}")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=out_dir,
        allow_patterns=allow,
    )
    return True


def _try_hf_cli(out_dir: str, signals: list[str]) -> bool:
    import shutil
    import subprocess

    if not shutil.which("huggingface-cli") and not shutil.which("hf"):
        return False
    cmd = ["hf", "download", REPO_ID, "--repo-type", "dataset", "--local-dir", out_dir]
    # Prefer hf, fall back
    if shutil.which("hf") is None:
        cmd[0] = "huggingface-cli"
        cmd[1] = "download"
    print("Running:", " ".join(cmd))
    try:
        subprocess.check_call(cmd)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _manual_instructions(out_dir: str, signals: list[str]) -> None:
    print(
        "\nAutomatic download unavailable (install `huggingface_hub` or `hf` CLI).\n"
        "Manual steps:\n"
        f"  1. Open https://huggingface.co/datasets/{REPO_ID}\n"
        f"  2. Download these trees into {out_dir}/ :\n"
    )
    for sig in signals:
        if sig == "traces":
            print("       - otlp_traces/**/*.parquet")
        elif sig == "logs":
            print("       - otlp_logs/**/*.parquet")
        elif sig == "metrics":
            print("       - otlp_metrics_*/**/*.parquet")
    print(
        "\n  3. Then:\n"
        "       uv run python -m corpus.ingest.build_shards "
        f"--adapter otel_demo_hf --input {out_dir} "
        "--num-train-shards 8 --write-val-shard\n"
        "\nLicense: Apache-2.0 — cite smithclay/otel-demo-telemetry + "
        "open-telemetry/opentelemetry-demo.\n"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=f"Fetch {REPO_ID}")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument(
        "--signals",
        default="traces,logs",
        help="Comma-separated: traces,logs,metrics",
    )
    args = p.parse_args(argv)
    signals = [s.strip() for s in args.signals.split(",") if s.strip()]
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    # Prefer huggingface_hub; optional dep — do not hard-require for smoke.
    if _try_huggingface_hub(out_dir, signals):
        print("Done.")
        print(f"Next: uv run python -m corpus.ingest.build_shards --adapter otel_demo_hf --input {out_dir} --num-train-shards 8 --write-val-shard")
        return 0
    if _try_hf_cli(out_dir, signals):
        print("Done.")
        return 0

    _manual_instructions(out_dir, signals)
    # Honest non-zero: fetch not wired without optional dep / network auth
    return 2


if __name__ == "__main__":
    sys.exit(main())
