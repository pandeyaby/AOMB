#!/usr/bin/env bash
# Agent loop optimising injected-fault detection (not val_bpb), on LogHub HDFS.
#
# Question: if the agent keeps a train.py change only when the model gets better
# at telling corrupted normal sessions from clean ones, does detection of REAL
# anomalies (labelled HDFS blocks the agent never sees) improve?
#
#   ./scripts/agent_inject_experiment.sh prepare    # cache → HDFS shards (+ end marker), worktree + branch
#   ./scripts/agent_inject_experiment.sh baseline   # measure the base model 3× → bar + noise margin (~25 min)
#   ./scripts/agent_inject_experiment.sh start      # launch the loop (overnight)
#   ./scripts/agent_inject_experiment.sh status
#   ./scripts/agent_inject_experiment.sh stop
#   ./scripts/agent_inject_experiment.sh evaluate   # real-fault AUROC for every kept commit (~12 min each)
#   ./scripts/agent_inject_experiment.sh restore    # put the previous prepare.py cache back
#
# Safety: separate git worktree and branch, AOMB_NO_PUSH=1 (covers every push),
# the prepare.py cache is moved aside and restored by `restore`. The agent's
# objective is computed by eval/objective.py on held-out NORMAL blocks; the
# labelled eval blocks are only touched by `evaluate`, after the run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="$HOME/.cache/autoresearch"
DATASETS="$HOME/.cache/aomb-datasets/loghub"
STATE="$HOME/.cache/aomb-datasets/agent-inject"
WORKTREE="$(dirname "$ROOT")/AOMB-agent-inject"
SESSIONS="$DATASETS/hdfs_sessions.jsonl"
VAL="$DATASETS/hdfs_sessions.val.jsonl"

die() { echo "ERROR: $*" >&2; exit 1; }
branch() { cat "$STATE/branch"; }
objective_env() {
  export AOMB_OBJECTIVE=inject AOMB_OBJECTIVE_VAL="$VAL" AOMB_OBJECTIVE_TRAIN="$SESSIONS"
  export AOMB_NO_PUSH=1 AOMB_CORPUS=hdfs_v1_inject
}

add_corpus_note() {
  grep -q "agent-inject experiment" "$WORKTREE/program.md" && return 0
  cat >> "$WORKTREE/program.md" <<'EOF'

## Current corpus and objective (agent-inject experiment)

Training data is LogHub HDFS_v1: real Hadoop DataNode / NameNode log lines, one
document per HDFS block, each ending with an [end_of_session] line. Only normal
blocks are used. This run keeps a change only when inject_auroc rises (see the
objective note in the prompt); val_bpb is reported for information.
EOF
}

cmd_prepare() {
  [[ -f "$SESSIONS" && -f "$VAL" ]] || die "HDFS sessions missing. Build them with:
  uv run python -m corpus.ingest.loghub_hdfs --input $DATASETS/HDFS_v1 --out $SESSIONS --n-val 1000"
  [[ -e "$STATE/hold" ]] && die "already prepared (see $STATE). Run 'restore' first to start over."
  mkdir -p "$STATE"
  cd "$ROOT"
  [[ -z "$(git status --porcelain)" ]] || die "main checkout has uncommitted changes; commit or stash first."

  local hold="$CACHE/_hold_before_agent_inject_$(date +%Y%m%d%H%M%S)"
  mkdir -p "$hold"
  for d in data tokenizer; do
    [[ -e "$CACHE/$d" ]] && mv "$CACHE/$d" "$hold/$d"
  done
  echo "$hold" > "$STATE/hold"
  echo "Moved previous cache to $hold"

  uv run python -m corpus.ingest.sessions_to_shards --train "$SESSIONS" --val "$VAL" \
    --data-dir "$CACHE/data" --num-train-shards 8 --end-marker
  uv run python prepare.py --num-shards 8

  local br="experiment/agent-inject-$(date +%Y%m%d)"
  git worktree add "$WORKTREE" -b "$br" HEAD
  echo "$br" > "$STATE/branch"
  git -C "$WORKTREE" rev-parse HEAD > "$STATE/base_sha"
  (cd "$WORKTREE" && uv sync --quiet)
  add_corpus_note
  echo
  echo "Prepared. Branch: $br   Worktree: $WORKTREE   Base: $(cat "$STATE/base_sha")"
  echo "Next: $0 baseline"
}

cmd_baseline() {
  [[ -f "$STATE/branch" ]] || die "run 'prepare' first"
  cd "$WORKTREE"
  mkdir -p logs
  objective_env
  : > "$STATE/baseline_runs.txt"
  for i in 1 2 3; do
    echo "Base model run $i/3 (about 8 minutes)..."
    caffeinate -i uv run python -m eval.objective > "logs/baseline_$i.txt" 2>&1 \
      || die "objective run failed; see $WORKTREE/logs/baseline_$i.txt"
    grep -E "^(val_bpb|inject_auroc):" "logs/baseline_$i.txt" | tr '\n' ' ' | tee -a "$STATE/baseline_runs.txt"
    echo | tee -a "$STATE/baseline_runs.txt"
  done
  # bar = mean; margin = 2 × sd, with a floor so a lucky trio can't set it to ~0
  python3 - "$STATE" <<'PY'
import re, statistics as st, sys, pathlib
state = pathlib.Path(sys.argv[1])
vals = [float(x) for x in re.findall(r"inject_auroc:\s*([0-9.]+)", (state / "baseline_runs.txt").read_text())]
assert len(vals) == 3, vals
mean, sd = st.mean(vals), st.pstdev(vals)
margin = max(0.005, 2 * sd)
(state / "base_inject_auroc").write_text(f"{mean:.6f}\n")
(state / "min_delta").write_text(f"{margin:.6f}\n")
print(f"Base inject_auroc: {mean:.4f} (runs {vals}, sd {sd:.4f}). Margin to count as an improvement: {margin:.4f}")
PY
  echo "Next: $0 start"
}

cmd_start() {
  [[ -f "$STATE/branch" ]] || die "run 'prepare' first"
  [[ -s "$STATE/base_inject_auroc" && -s "$STATE/min_delta" ]] || die "run 'baseline' first (the loop needs a bar and a noise margin)"
  add_corpus_note
  cd "$WORKTREE"
  mkdir -p logs
  objective_env
  export AOMB_BEST_INJECT_AUROC="$(cat "$STATE/base_inject_auroc")" AOMB_MIN_DELTA="$(cat "$STATE/min_delta")"
  local run=(uv run)
  if [[ -n "${AOMB_ANTHROPIC_API_KEYS:-}${ANTHROPIC_API_KEY:-}" ]]; then
    run=(uv run --with anthropic)
    echo "Using the Anthropic SDK (API key found in the environment)."
  else
    echo "No API key in the environment: the loop will use the 'claude' CLI login (rate-limited)."
  fi
  echo "Bar: inject_auroc > $AOMB_BEST_INJECT_AUROC + $AOMB_MIN_DELTA"
  nohup caffeinate -i "${run[@]}" python agent_loop.py >> logs/agent_loop.log 2>&1 &
  sleep 3
  echo "Started (pid $(cat logs/agent_loop.pid 2>/dev/null || echo '?')). Log: $WORKTREE/logs/agent_loop.log"
  echo "Each experiment is ~9 min; overnight (~8 h) gives roughly 50 experiments."
}

cmd_status() {
  [[ -f "$STATE/branch" ]] || die "run 'prepare' first"
  local pid; pid="$(cat "$WORKTREE/logs/agent_loop.pid" 2>/dev/null || true)"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then echo "Loop running (pid $pid)"; else echo "Loop not running"; fi
  echo "Kept commits on $(branch):"
  git -C "$ROOT" log --oneline "$(cat "$STATE/base_sha")..$(branch)" -- train.py | cat
  echo "--- last log lines"
  tail -n 8 "$WORKTREE/logs/agent_loop.log" 2>/dev/null || true
}

cmd_stop() {
  local pid; pid="$(cat "$WORKTREE/logs/agent_loop.pid" 2>/dev/null || true)"
  [[ -n "$pid" ]] && kill "$pid" 2>/dev/null && echo "Stopped pid $pid" || echo "Loop not running"
}

cmd_evaluate() {
  [[ -f "$STATE/branch" ]] || die "run 'prepare' first"
  local out="reports/public-accuracy/agent-loop-inject-$(date +%Y%m%d)"
  cd "$ROOT"
  caffeinate -i uv run python -m eval.agent_commits --base "$(cat "$STATE/base_sha")" \
    --branch "$(branch)" --sessions "$SESSIONS" --out-dir "$out" \
    --end-marker --primary bpb_top10 "$@"
  echo "Results: $out/summary.md"
}

cmd_restore() {
  [[ -f "$STATE/hold" ]] || die "nothing to restore"
  local hold; hold="$(cat "$STATE/hold")"
  local done_dir="$CACHE/_agent_inject_cache_$(date +%Y%m%d%H%M%S)"
  mkdir -p "$done_dir"
  for d in data tokenizer; do
    [[ -e "$CACHE/$d" ]] && mv "$CACHE/$d" "$done_dir/$d"
    [[ -e "$hold/$d" ]] && mv "$hold/$d" "$CACHE/$d"
  done
  rmdir "$hold" 2>/dev/null || true
  rm -f "$STATE/hold"
  echo "Restored the previous cache. HDFS shards kept at $done_dir"
  echo "The worktree ($WORKTREE) and branch $(branch 2>/dev/null || echo '?') are left in place."
}

case "${1:-}" in
  prepare|baseline|start|status|stop|evaluate|restore) c="$1"; shift; "cmd_$c" "$@" ;;
  *) sed -n '2,22p' "$0"; exit 1 ;;
esac
