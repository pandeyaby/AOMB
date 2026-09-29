#!/usr/bin/env bash
# Agent loop ↔ detection experiment on LogHub HDFS.
#
# Question: when the overnight agent keeps a train.py change because val_bpb
# improved, does anomaly detection (AUROC on labelled HDFS blocks) improve too?
#
#   ./scripts/agent_hdfs_experiment.sh prepare    # cache → HDFS shards, worktree + branch
#   ./scripts/agent_hdfs_experiment.sh start      # launch the agent loop (overnight)
#   ./scripts/agent_hdfs_experiment.sh status     # progress
#   ./scripts/agent_hdfs_experiment.sh stop       # stop the loop
#   ./scripts/agent_hdfs_experiment.sh evaluate   # AUROC for every kept commit (~15 min each)
#   ./scripts/agent_hdfs_experiment.sh restore    # put the previous prepare.py cache back
#
# Safety: the loop runs in a separate git worktree on its own branch with
# AOMB_NO_PUSH=1, so it never touches main and never pushes. The prepare.py cache
# (~/.cache/autoresearch/{data,tokenizer}) is moved aside, not deleted, and
# `restore` puts it back. The agent only ever sees normal HDFS blocks: train
# shards = the first 5,000 normal blocks, val shard = 1,000 other normal blocks
# disjoint from the labelled eval set.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="$HOME/.cache/autoresearch"
DATASETS="$HOME/.cache/aomb-datasets/loghub"
STATE="$HOME/.cache/aomb-datasets/agent-hdfs"
WORKTREE="$(dirname "$ROOT")/AOMB-agent-hdfs"
SESSIONS="$DATASETS/hdfs_sessions.jsonl"
VAL="$DATASETS/hdfs_sessions.val.jsonl"

die() { echo "ERROR: $*" >&2; exit 1; }
branch() { cat "$STATE/branch"; }

cmd_prepare() {
  [[ -f "$SESSIONS" && -f "$VAL" ]] || die "HDFS sessions missing. Build them with:
  uv run python -m corpus.ingest.loghub_hdfs --input $DATASETS/HDFS_v1 --out $SESSIONS --n-val 1000"
  [[ -e "$STATE/hold" ]] && die "already prepared (see $STATE). Run 'restore' first to start over."
  mkdir -p "$STATE"
  cd "$ROOT"
  [[ -z "$(git status --porcelain)" ]] || die "main checkout has uncommitted changes; commit or stash first."

  # 1. Move the current prepare.py cache aside (reversible).
  local hold="$CACHE/_hold_before_agent_hdfs_$(date +%Y%m%d%H%M%S)"
  mkdir -p "$hold"
  for d in data tokenizer; do
    [[ -e "$CACHE/$d" ]] && mv "$CACHE/$d" "$hold/$d"
  done
  echo "$hold" > "$STATE/hold"
  echo "Moved previous cache to $hold"

  # 2. HDFS normal blocks → shards, then fit the tokenizer.
  uv run python -m corpus.ingest.sessions_to_shards --train "$SESSIONS" --val "$VAL" \
    --data-dir "$CACHE/data" --num-train-shards 8
  uv run python prepare.py --num-shards 8

  # 3. Worktree on its own branch for the agent's commits.
  local br="experiment/agent-hdfs-$(date +%Y%m%d)"
  git worktree add "$WORKTREE" -b "$br" HEAD
  echo "$br" > "$STATE/branch"
  git -C "$WORKTREE" rev-parse HEAD > "$STATE/base_sha"
  (cd "$WORKTREE" && uv sync --quiet)
  cat >> "$WORKTREE/program.md" <<'EOF'

## Current corpus (agent-hdfs experiment)

Training data is LogHub HDFS_v1: real Hadoop DataNode / NameNode log lines,
one document per HDFS block (block ids, IPs, job and task ids normalised).
Only normal blocks are used. Optimise val_bpb as usual.
EOF
  echo
  echo "Prepared. Branch: $br   Worktree: $WORKTREE   Base: $(cat "$STATE/base_sha")"
  echo "Next: export ANTHROPIC_API_KEY (or rely on the claude CLI login), then: $0 start"
}

cmd_start() {
  [[ -f "$STATE/branch" ]] || die "run 'prepare' first"
  cd "$WORKTREE"
  mkdir -p logs
  local run=(uv run)
  if [[ -n "${AOMB_ANTHROPIC_API_KEYS:-}${ANTHROPIC_API_KEY:-}" ]]; then
    run=(uv run --with anthropic)
    echo "Using the Anthropic SDK (API key found in the environment)."
  else
    echo "No API key in the environment: the loop will use the 'claude' CLI login (rate-limited)."
  fi
  AOMB_NO_PUSH=1 AOMB_CORPUS=hdfs_v1 nohup caffeinate -i "${run[@]}" python agent_loop.py \
    >> logs/agent_loop.log 2>&1 &
  sleep 3
  echo "Started (pid $(cat logs/agent_loop.pid 2>/dev/null || echo '?')). Log: $WORKTREE/logs/agent_loop.log"
  echo "Each experiment is ~7 min; overnight (~8 h) gives roughly 60–70 experiments."
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
  local out="reports/public-accuracy/agent-loop-hdfs-$(date +%Y%m%d)"
  cd "$ROOT"
  caffeinate -i uv run python -m eval.agent_commits --base "$(cat "$STATE/base_sha")" \
    --branch "$(branch)" --sessions "$SESSIONS" --out-dir "$out" "$@"
  echo "Results: $out/summary.md"
}

cmd_restore() {
  [[ -f "$STATE/hold" ]] || die "nothing to restore"
  local hold; hold="$(cat "$STATE/hold")"
  local done_dir="$CACHE/_agent_hdfs_cache_$(date +%Y%m%d%H%M%S)"
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
  prepare|start|status|stop|evaluate|restore) c="$1"; shift; "cmd_$c" "$@" ;;
  *) sed -n '2,20p' "$0"; exit 1 ;;
esac
