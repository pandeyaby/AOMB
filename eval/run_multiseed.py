"""
Multi-seed runner for public accuracy eval.

Runs eval.run_eval for seeds 0..N-1 (or an explicit list) and writes
aggregate mean±std JSON + markdown. Does not invent metrics or publish claims.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.report import aggregate_seed_reports, render_aggregate_markdown


def parse_seeds(spec: str) -> list[int]:
    spec = spec.strip()
    if not spec:
        raise ValueError("empty seeds")
    if ".." in spec and "," not in spec:
        # 0..4 inclusive
        a, b = spec.split("..", 1)
        start, end = int(a), int(b)
        if end < start:
            raise ValueError("seed range end < start")
        return list(range(start, end + 1))
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Multi-seed public accuracy eval (protocol: 3–5 seeds)"
    )
    p.add_argument(
        "--seeds",
        type=str,
        default="0..2",
        help="Comma list (0,1,2) or inclusive range 0..4 (default: 0..2)",
    )
    p.add_argument(
        "--capture",
        type=str,
        default=str(ROOT / "corpus" / "fixtures" / "lab_sample"),
    )
    p.add_argument(
        "--scores-from",
        type=str,
        default="length",
        choices=["length", "events", "precomputed", "model"],
    )
    p.add_argument("--precomputed-scores", type=str, default=None)
    p.add_argument("--train-seconds", type=float, default=0.0)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--random-draws", type=int, default=64)
    p.add_argument(
        "--skip-run",
        action="store_true",
        help="Only aggregate existing seed-*/report.json under out-dir",
    )
    args = p.parse_args(argv)

    seeds = parse_seeds(args.seeds)
    if len(seeds) < 3 and not args.skip_run:
        print(
            f"WARNING: protocol asks for 3–5 seeds; got {len(seeds)}. "
            "Aggregate still written; claim checklist will fail.",
            file=sys.stderr,
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_paths: list[Path] = []

    if not args.skip_run:
        for seed in seeds:
            seed_out = out_dir / f"seed-{seed}"
            cmd = [
                sys.executable,
                "-m",
                "eval.run_eval",
                "--capture",
                args.capture,
                "--seed",
                str(seed),
                "--scores-from",
                args.scores_from,
                "--out-dir",
                str(seed_out),
                "--random-draws",
                str(args.random_draws),
            ]
            if args.precomputed_scores:
                cmd += ["--precomputed-scores", args.precomputed_scores]
            if args.scores_from == "model":
                cmd += ["--train-seconds", str(args.train_seconds)]
            print("Running:", " ".join(cmd))
            subprocess.check_call(cmd, cwd=str(ROOT))
            seed_paths.append(seed_out / "report.json")
    else:
        seed_paths = sorted(out_dir.glob("seed-*/report.json"))
        if not seed_paths:
            print(f"ERROR: no seed-*/report.json under {out_dir}", file=sys.stderr)
            return 2

    agg = aggregate_seed_reports(seed_paths)
    agg_path = out_dir / "aggregate.json"
    md_path = out_dir / "aggregate.md"
    agg_path.write_text(json.dumps(agg, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_aggregate_markdown(agg), encoding="utf-8")
    print(f"Wrote {agg_path}")
    print(f"Wrote {md_path}")
    ms = agg["metrics_mean_std"]["auroc"]
    print(
        f"AUROC mean±std = {ms['mean']:.4f}±{ms['std']:.4f} "
        f"(n={ms['n']}, claim_status=not_published)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
