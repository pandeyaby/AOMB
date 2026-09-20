"""
Thin entrypoint: ``uv run python -m score_session ...``

Delegates to ``eval.score_cli`` (session BPB / surprise only).
Refuses ``--auroc`` / publish flags (exit 1). Missing dump → exit 2 (same as ``eval.score_cli``). ``prepare.py`` is sacred.

See ``python -m score_session --help``.
"""

from __future__ import annotations

from eval.score_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
