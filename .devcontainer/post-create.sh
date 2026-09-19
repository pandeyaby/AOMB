#!/usr/bin/env bash
# Codespaces post-create: deps for the no-MPS / no-key stranger verify path.
set -euo pipefail
cd "${containerWorkspaceFolder:-$(pwd)}"

chmod +x scripts/stranger_verify.sh scripts/run_diptych_full8.sh scripts/run_public_ranking_card_v1.sh \
  .devcontainer/post-create.sh 2>/dev/null || true

if command -v uv >/dev/null 2>&1; then
  echo "uv present — optional: uv sync (heavier; stranger verify uses pip deps from image)"
else
  echo "Using system/image pip deps (pyarrow numpy rustbpe tiktoken)."
fi

echo
echo "Cite without cloning (in this Codespace):"
echo "  STRANGER_FAST=1 ./scripts/stranger_verify.sh"
echo "Docs: docs/stranger-verify.md"
echo "Honest: full-8 + gate_axis_mutate + ranking-card baselines ε — NOT lab AUROC / MPS train."
