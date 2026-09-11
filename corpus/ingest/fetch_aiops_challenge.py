"""
Eval-only: AIOps Challenge 2020 — labeled faults.

Non-commercial use only. Do NOT redistribute the dataset from this repo.
This module prints citation + fetch URLs; it does not vendor the data.
"""

from __future__ import annotations

import argparse
import sys

REPO = "https://github.com/NetManAIOps/AIOps-Challenge-2020-Data"
TSINGHUA = "https://cloud.tsinghua.edu.cn/f/c1ea3426ce444bc9baae/"
GDRIVE = "https://drive.google.com/file/d/1nkEsD1g7THm_T58KwUQZ7o-b174fdx-n/view?usp=sharing"
MD5 = "fac7fe1b4e048c81ef88874334b73534"
CREDIT = "https://competition.aiops-challenge.com/home/competition/1484441527290765368"


def print_notice() -> None:
    print(
        f"""
AIOps Challenge 2020 — EVAL ONLY (do not redistribute)

  Source:  {REPO}
  Credit:  {CREDIT}
  License: Non-commercial (research / classroom teaching only).
           Users take full responsibility; organizers disclaim liability.
           See LICENSE section in the upstream README.

  Stage One downloads:
    - Tsinghua Cloud: {TSINGHUA}
    - Google Drive:   {GDRIVE}
    - md5sum:         {MD5}

  Contents (upstream):
    - 故障整理（预赛）.csv — labeled failures (time, type, localization)
    - Daily zips with business / infra metrics + call-chain traces

  AOMB policy:
    - Cite + fetch locally for evaluation / labeled-fault scoring.
    - Do NOT commit or publish redistributed copies in this repository.
    - Not part of the flagship training corpus product story.
"""
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Cite + fetch instructions for AIOps Challenge 2020 (eval-only)"
    )
    p.add_argument(
        "--check-only",
        action="store_true",
        help="Print notice and exit (default behavior)",
    )
    p.parse_args(argv)
    print_notice()
    return 0


if __name__ == "__main__":
    sys.exit(main())
