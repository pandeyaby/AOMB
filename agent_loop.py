"""
agent_loop.py — Autonomous Observability Model Breeder Orchestrator
════════════════════════════════════════════════════════════════════
Drives the autoresearch-macos loop autonomously:

  1. Read program.md + current train.py + git experiment history
  2. Call `claude -p` (Claude Code print mode) to get proposed train.py
  3. Validate proposed change (syntax check)
  4. Run `uv run train.py` for 5 minutes (TIME_BUDGET=300s)
  5. Parse val_bpb from output
  6. If improved → commit; else → roll back
  7. Repeat until MAX_EXPERIMENTS or stopped

Start with:
    nohup python agent_loop.py > logs/agent_loop.log 2>&1 &
    echo $! > logs/agent_loop.pid

Stop cleanly:
    kill $(cat logs/agent_loop.pid)
"""

import ast
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import anthropic as _anthropic_module
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

try:
    import openai as _openai_module
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

# ── Config ───────────────────────────────────────────────────────────────────

REPO_DIR       = Path(__file__).parent.resolve()
TRAIN_PY       = REPO_DIR / "train.py"
PROGRAM_MD     = REPO_DIR / "program.md"
LOGS_DIR       = REPO_DIR / "logs"
BACKUP_DIR     = LOGS_DIR / "train_backups"

MAX_EXPERIMENTS   = 120      # safety cap for overnight
MAX_RETRIES       = 3        # retries if claude call fails / bad output
SLEEP_BETWEEN     = 10       # seconds between experiments (let MPS cool)
CLAUDE_TIMEOUT    = 300      # seconds to wait for claude -p response (opus can take ~3 min)
TRAIN_TIMEOUT     = 660      # seconds (5 min budget + 6 min buffer for larger models + eval)
CLAUDE_MODEL      = "sonnet" # default; override with AOMB_CLAUDE_MODELS env var
                             # Anthropic aliases: opus, sonnet, haiku
                             # OpenAI models:     gpt-4o, gpt-4o-mini, gpt-4.1, gpt-4.1-mini
MAX_GIT_LOG_LINES = 30       # recent experiment lines to feed agent


def _parse_csv_env(name: str) -> list[str]:
    raw = os.getenv(name, "")
    return [v.strip() for v in raw.split(",") if v.strip()]


def _discover_api_keys() -> list[str]:
    """
    Discover API keys from env in priority order:
    1) AOMB_ANTHROPIC_API_KEYS=key1,key2,...
    2) ANTHROPIC_API_KEY_1, ANTHROPIC_API_KEY_2, ...
    3) ANTHROPIC_API_KEY
    """
    keys: list[str] = []
    keys.extend(_parse_csv_env("AOMB_ANTHROPIC_API_KEYS"))

    numbered: list[tuple[int, str]] = []
    for env_name, env_val in os.environ.items():
        m = re.fullmatch(r"ANTHROPIC_API_KEY_(\d+)", env_name)
        if m and env_val.strip():
            numbered.append((int(m.group(1)), env_val.strip()))
    numbered.sort(key=lambda t: t[0])
    keys.extend([v for _, v in numbered])

    default_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if default_key:
        keys.append(default_key)

    # Stable de-dup preserving order
    deduped: list[str] = []
    seen = set()
    for key in keys:
        if key not in seen:
            deduped.append(key)
            seen.add(key)
    return deduped


def _discover_openai_keys() -> list[str]:
    """Discover OpenAI keys: AOMB_OPENAI_API_KEYS, OPENAI_API_KEY_1..N, OPENAI_API_KEY."""
    keys: list[str] = []
    keys.extend(_parse_csv_env("AOMB_OPENAI_API_KEYS"))
    numbered: list[tuple[int, str]] = []
    for env_name, env_val in os.environ.items():
        m = re.fullmatch(r"OPENAI_API_KEY_(\d+)", env_name)
        if m and env_val.strip():
            numbered.append((int(m.group(1)), env_val.strip()))
    numbered.sort(key=lambda t: t[0])
    keys.extend([v for _, v in numbered])
    default_key = os.getenv("OPENAI_API_KEY", "").strip()
    if default_key:
        keys.append(default_key)
    deduped: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if key not in seen:
            deduped.append(key)
            seen.add(key)
    return deduped


CLAUDE_MODELS    = _parse_csv_env("AOMB_CLAUDE_MODELS") or [CLAUDE_MODEL]
API_KEY_RING     = _discover_api_keys()
OPENAI_KEY_RING  = _discover_openai_keys()

# Which models are OpenAI (routed to OpenAI SDK instead of Anthropic)
_OPENAI_MODEL_PREFIXES = ("gpt-", "o1", "o3", "o4")

# ── Logging setup ─────────────────────────────────────────────────────────────

LOGS_DIR.mkdir(exist_ok=True)
BACKUP_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("AOMB")

# ── Helpers ───────────────────────────────────────────────────────────────────

def read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_file(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def git(*args, check=True) -> str:
    result = subprocess.run(
        ["git"] + list(args),
        cwd=REPO_DIR, capture_output=True, text=True, check=False
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr}")
    return result.stdout.strip()


def get_recent_experiments(n: int = MAX_GIT_LOG_LINES) -> str:
    """Return recent git commits that contain val_bpb for agent context."""
    try:
        log_output = git("log", "--oneline", f"-{n}", "--format=%s")
        lines = [l for l in log_output.splitlines() if "val_bpb" in l]
        if not lines:
            return "No experiments yet — this is the first run."
        return "\n".join(lines)
    except Exception:
        return "Could not retrieve experiment history."


def get_best_val_bpb() -> float:
    """Parse git log to find the lowest val_bpb so far."""
    try:
        log_output = git("log", "--format=%s", "-100")
        bpbs = [
            float(m.group(1))
            for m in re.finditer(r"val_bpb=(\d+\.\d+)", log_output)
        ]
        return min(bpbs) if bpbs else float("inf")
    except Exception:
        return float("inf")


def parse_val_bpb(output: str) -> float | None:
    """Extract the FINAL val_bpb line from train.py output."""
    # train.py prints: "val_bpb:          X.XXXXXX"
    matches = re.findall(r"val_bpb:\s+(\d+\.\d+)", output)
    if matches:
        val = float(matches[-1])
        return None if (val != val or val > 50) else val   # NaN / exploded check
    return None


def validate_python(code: str) -> tuple[bool, str]:
    """Check code is valid Python. Returns (ok, error_message)."""
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, str(e)


def extract_code_block(text: str) -> str | None:
    """
    Extract the content of the LAST ```python ... ``` block.
    Falls back to first ``` block if no python-tagged block found.
    """
    # Try ```python ... ``` first
    matches = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    # Fallback: any ``` block
    matches = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    if matches:
        candidate = matches[-1].strip()
        if "import torch" in candidate or "def " in candidate:
            return candidate
    return None


def _build_prompt(experiment_num: int) -> str:
    program_md  = read_file(PROGRAM_MD)
    train_py    = read_file(TRAIN_PY)
    recent_exps = get_recent_experiments()
    best_bpb    = get_best_val_bpb()
    return f"""You are the research agent for the Autonomous Observability Model Breeder.
This is experiment #{experiment_num}.

=== RESEARCH CONSTITUTION (program.md) ===
{program_md}

=== CURRENT train.py ===
```python
{train_py}
```

=== EXPERIMENT HISTORY (recent val_bpb results, chronological) ===
Best val_bpb so far: {best_bpb:.4f}

{recent_exps}

=== YOUR TASK ===
Follow the mandatory Thinking Protocol from program.md (all 6 steps).
Then propose EXACTLY ONE focused change to train.py.

CRITICAL OUTPUT FORMAT:
- Output the COMPLETE new train.py as a single ```python ... ``` fenced code block.
- The code block must be the LAST thing in your response.
- Do NOT truncate or summarize any part of train.py.
- Do NOT use any tools — only output text.
- Your thinking and reasoning BEFORE the code block is encouraged and welcome.
"""


# Full API model names for the SDK (aliases not always accepted).
# Override at runtime via AOMB_CLAUDE_MODELS env var with full model IDs.
_SDK_MODEL_MAP = {
    "opus":   "claude-opus-4-5",
    "sonnet": "claude-sonnet-4-5",
    "haiku":  "claude-haiku-4-5",
}


def _call_via_sdk(prompt: str, model: str, api_key: str,
                  experiment_num: int, attempt_num: int) -> str | None:
    """Use the Anthropic Python SDK directly — bypasses Claude Code rate limits."""
    sdk_model = _SDK_MODEL_MAP.get(model, model)
    try:
        client = _anthropic_module.Anthropic(api_key=api_key)
        log.info(f"[Exp {experiment_num}] SDK call (model={sdk_model}, key=...{api_key[-6:]})...")
        t0 = time.time()
        message = client.messages.create(
            model=sdk_model,
            max_tokens=16000,
            messages=[{"role": "user", "content": prompt}],
        )
        elapsed = time.time() - t0
        log.info(f"[Exp {experiment_num}] SDK responded in {elapsed:.1f}s")
        return message.content[0].text
    except _anthropic_module.RateLimitError as e:
        log.warning(f"[Exp {experiment_num}] SDK rate limit (429): {e}. Sleeping 60s.")
        time.sleep(60)
        return None
    except _anthropic_module.APIStatusError as e:
        log.error(f"[Exp {experiment_num}] SDK API error ({e.status_code}): {e.message[:200]}")
        return None
    except Exception as e:
        log.error(f"[Exp {experiment_num}] SDK unexpected error: {e}")
        return None


def _call_via_openai(prompt: str, model: str, api_key: str,
                     experiment_num: int) -> str | None:
    """Use the OpenAI Python SDK — GPT-4o, gpt-4.1, o1, etc."""
    try:
        client = _openai_module.OpenAI(api_key=api_key)
        log.info(f"[Exp {experiment_num}] OpenAI call (model={model}, key=...{api_key[-6:]})...")
        t0 = time.time()
        response = client.chat.completions.create(
            model=model,
            max_tokens=16000,
            messages=[{"role": "user", "content": prompt}],
        )
        elapsed = time.time() - t0
        log.info(f"[Exp {experiment_num}] OpenAI responded in {elapsed:.1f}s")
        return response.choices[0].message.content
    except _openai_module.RateLimitError as e:
        log.warning(f"[Exp {experiment_num}] OpenAI rate limit (429): {e}. Sleeping 60s.")
        time.sleep(60)
        return None
    except _openai_module.APIStatusError as e:
        log.error(f"[Exp {experiment_num}] OpenAI API error ({e.status_code}): {str(e)[:200]}")
        return None
    except Exception as e:
        log.error(f"[Exp {experiment_num}] OpenAI unexpected error: {e}")
        return None


def _call_via_cli(prompt: str, model: str, experiment_num: int) -> str | None:
    """Fallback: use `claude --print`. Detects rate-limit fast-failures."""
    cmd = ["claude", "--print", "--model", model, "--output-format", "text", prompt]
    log.info(f"[Exp {experiment_num}] CLI fallback (model={model})...")
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, cwd=REPO_DIR, capture_output=True,
            text=True, timeout=CLAUDE_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        log.error(f"Claude CLI timed out after {CLAUDE_TIMEOUT}s")
        return None
    elapsed = time.time() - t0

    if result.returncode != 0:
        detail = (result.stderr or "").strip() or (result.stdout or "").strip()
        # Fast failure (<8s) with no output = Claude Code usage/rate limit
        if elapsed < 8 and not detail:
            log.warning(
                f"[Exp {experiment_num}] CLI rate-limited (rc=1, {elapsed:.1f}s, no output). "
                "Sleeping 3600s to let Claude Code quota reset."
            )
            time.sleep(3600)
        else:
            log.error(f"[Exp {experiment_num}] CLI error (rc={result.returncode}, {elapsed:.1f}s): {detail[:300]}")
        return None

    log.info(f"[Exp {experiment_num}] CLI responded in {elapsed:.1f}s")
    return result.stdout or None


def call_claude_agent(experiment_num: int, attempt_num: int) -> str | None:
    """
    Call the Claude API to get a proposed train.py.
    Priority:
      1. Anthropic Python SDK (if anthropic installed + API key available) — no Code rate limits
      2. `claude --print` CLI fallback with rate-limit detection
    Returns proposed train.py content, or None on failure.
    """
    prompt = _build_prompt(experiment_num)
    model  = CLAUDE_MODELS[(attempt_num - 1) % len(CLAUDE_MODELS)]

    is_openai = model.startswith(_OPENAI_MODEL_PREFIXES)

    # ── Path 1a: OpenAI SDK ───────────────────────────────────────────────────
    if is_openai and _OPENAI_AVAILABLE and OPENAI_KEY_RING:
        key_idx = (attempt_num - 1) % len(OPENAI_KEY_RING)
        response = _call_via_openai(prompt, model, OPENAI_KEY_RING[key_idx],
                                    experiment_num)
    # ── Path 1b: Anthropic SDK ────────────────────────────────────────────────
    elif not is_openai and _ANTHROPIC_AVAILABLE and API_KEY_RING:
        key_idx = (attempt_num - 1) % len(API_KEY_RING)
        response = _call_via_sdk(prompt, model, API_KEY_RING[key_idx],
                                 experiment_num, attempt_num)
    # ── Path 2: Claude CLI fallback (no API key) ──────────────────────────────
    else:
        if is_openai:
            log.warning(f"OpenAI model '{model}' requested but no OPENAI_API_KEY found — falling back to CLI")
        elif not _ANTHROPIC_AVAILABLE:
            log.warning("anthropic SDK not installed — using CLI fallback")
        else:
            log.warning("No Anthropic API key found — using CLI (subject to Code rate limits)")
        response = _call_via_cli(prompt, model, experiment_num)

    if not response or not response.strip():
        log.error(f"[Exp {experiment_num}] Empty response from Claude")
        return None

    # Save full response for debugging
    resp_path = LOGS_DIR / f"exp_{experiment_num:04d}_claude_response.txt"
    resp_path.write_text(response, encoding="utf-8")

    proposed = extract_code_block(response)
    if not proposed:
        log.error(f"[Exp {experiment_num}] No code block found in response")
        log.debug(f"Response preview: {response[:500]}")
        return None

    return proposed


def run_training(experiment_num: int) -> tuple[str, float | None]:
    """
    Run `uv run train.py` for up to TRAIN_TIMEOUT seconds.
    Returns (full_output, val_bpb or None).
    """
    log.info(f"[Exp {experiment_num}] Starting 5-minute training run...")
    t0 = time.time()

    out_path = LOGS_DIR / f"exp_{experiment_num:04d}_train_output.txt"

    try:
        result = subprocess.run(
            ["uv", "run", "train.py"],
            cwd=REPO_DIR,
            capture_output=True,
            text=True,
            timeout=TRAIN_TIMEOUT,
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout or b""
        stderr = e.stderr or b""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        output = stdout + stderr
        log.warning(f"Training timed out after {TRAIN_TIMEOUT}s")
    except Exception as ex:
        log.error(f"Training subprocess error: {ex}")
        return "", None

    elapsed = time.time() - t0
    out_path.write_text(output, encoding="utf-8")

    val_bpb = parse_val_bpb(output)
    if val_bpb is not None:
        log.info(f"[Exp {experiment_num}] Training done in {elapsed:.0f}s | val_bpb={val_bpb:.4f}")
    else:
        log.warning(f"[Exp {experiment_num}] Training done in {elapsed:.0f}s | val_bpb=NONE (crash/NaN)")
        # Log last 20 lines for diagnosis
        tail = "\n".join(output.splitlines()[-20:])
        log.debug(f"Training tail:\n{tail}")

    return output, val_bpb


def commit_experiment(experiment_num: int, val_bpb: float, prev_best: float,
                      change_summary: str) -> None:
    """Commit train.py with val_bpb in message."""
    delta = val_bpb - prev_best
    msg = (
        f"[val_bpb={val_bpb:.4f}] [Δ={delta:+.4f}] "
        f"[change: {change_summary[:60]}] "
        f"[exp: {experiment_num}]\n\n"
        f"Authored-By: Abhinav Pandey <pandey.aby@gmail.com>"
    )
    git("add", "train.py")
    git("commit", "-m", msg)
    log.info(f"[Exp {experiment_num}] Committed: val_bpb={val_bpb:.4f} (Δ={delta:+.4f})")


def extract_change_summary(claude_response_path: Path) -> str:
    """Try to extract a one-line summary of the change from Claude's reasoning."""
    try:
        text = claude_response_path.read_text(encoding="utf-8")
        # Look for common patterns Claude uses to describe the change
        for pattern in [
            r"(?:Change|Modification|Change I'm making|Proposed change)[:\s]+(.{20,100})",
            r"(?:I'm|I am|Will) (?:changing|modifying|setting|trying|increasing|decreasing)\s+(.{15,80})",
        ]:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return m.group(1).strip().rstrip(".")
        # Fallback: first non-blank line before the code block
        lines = text.split("```python")[0].strip().splitlines()
        for line in reversed(lines):
            line = line.strip()
            if len(line) > 20:
                return line[:80]
    except Exception:
        pass
    return "see claude response log"


# ── Push helper ───────────────────────────────────────────────────────────────

def push_to_remote(experiment_num: int) -> None:
    """Push every 10 experiments to keep GitHub up to date."""
    if experiment_num % 10 != 0:
        return
    try:
        git("push", "origin", "main", check=False)
        log.info(f"[Exp {experiment_num}] Pushed to GitHub")
    except Exception as ex:
        log.warning(f"Push failed (non-fatal): {ex}")


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("  AUTONOMOUS OBSERVABILITY MODEL BREEDER — AGENT LOOP")
    log.info(f"  Max experiments: {MAX_EXPERIMENTS}")
    log.info(f"  Models: {', '.join(CLAUDE_MODELS)}")
    log.info(f"  Anthropic key slots: {len(API_KEY_RING)}")
    log.info(f"  OpenAI key slots:    {len(OPENAI_KEY_RING)}")
    log.info(f"  Repo: {REPO_DIR}")
    log.info("=" * 60)

    # Write PID file
    pid_path = LOGS_DIR / "agent_loop.pid"
    pid_path.write_text(str(os.getpid()))
    log.info(f"PID {os.getpid()} written to {pid_path}")

    # Verify prerequisites
    for req in [TRAIN_PY, PROGRAM_MD]:
        if not req.exists():
            log.error(f"Missing required file: {req}")
            sys.exit(1)

    best_val_bpb = get_best_val_bpb()
    if best_val_bpb == float("inf"):
        log.info("No prior experiments found. Starting fresh.")
    else:
        log.info(f"Resuming from best val_bpb={best_val_bpb:.4f}")

    experiment_num = 0
    consecutive_failures = 0

    while experiment_num < MAX_EXPERIMENTS:
        experiment_num += 1
        log.info(f"\n{'─'*60}")
        log.info(f"  EXPERIMENT {experiment_num}/{MAX_EXPERIMENTS}  |  Best so far: {best_val_bpb:.4f}")
        log.info(f"{'─'*60}")

        # ── Step 1: Backup current train.py ─────────────────────────────────
        backup_path = BACKUP_DIR / f"train_exp_{experiment_num:04d}_before.py"
        shutil.copy2(TRAIN_PY, backup_path)

        # ── Step 2: Get proposed change from Claude ──────────────────────────
        proposed_code = None
        for attempt in range(1, MAX_RETRIES + 1):
            proposed_code = call_claude_agent(experiment_num, attempt)
            if proposed_code:
                ok, err = validate_python(proposed_code)
                if ok:
                    break
                else:
                    log.warning(f"Attempt {attempt}: syntax error in proposed code: {err}")
                    proposed_code = None
            else:
                log.warning(f"Attempt {attempt}: Claude returned no valid code")
            if attempt < MAX_RETRIES:
                time.sleep(60)  # 60s between retries — give rate limits time to clear

        if not proposed_code:
            log.error(f"[Exp {experiment_num}] All {MAX_RETRIES} Claude attempts failed. Skipping.")
            consecutive_failures += 1
            if consecutive_failures >= 10:
                log.critical("10 consecutive failures. Stopping loop.")
                break
            # Exponential backoff: 2min, 4min, 8min … capped at 15min
            # Handles Claude rate-limits which reset on a rolling window
            backoff = min(120 * (2 ** (consecutive_failures - 1)), 900)
            log.warning(f"Consecutive failures={consecutive_failures}. Cooling off {backoff}s before retry.")
            time.sleep(backoff)
            continue

        consecutive_failures = 0

        # ── Step 3: Apply proposed train.py ─────────────────────────────────
        write_file(TRAIN_PY, proposed_code)
        log.info(f"[Exp {experiment_num}] Applied proposed train.py ({len(proposed_code)} chars)")

        # ── Step 4: Run training ─────────────────────────────────────────────
        output, val_bpb = run_training(experiment_num)

        # ── Step 5: Evaluate result ──────────────────────────────────────────
        if val_bpb is None:
            log.warning(f"[Exp {experiment_num}] Training crashed or NaN — rolling back")
            shutil.copy2(backup_path, TRAIN_PY)
            time.sleep(SLEEP_BETWEEN)
            continue

        # ── Step 6: Commit if improved, else roll back ───────────────────────
        if val_bpb < best_val_bpb:
            resp_path = LOGS_DIR / f"exp_{experiment_num:04d}_claude_response.txt"
            summary = extract_change_summary(resp_path)
            commit_experiment(experiment_num, val_bpb, best_val_bpb, summary)
            best_val_bpb = val_bpb
            push_to_remote(experiment_num)
            log.info(f"[Exp {experiment_num}] ✓ IMPROVED → new best={best_val_bpb:.4f}")
        else:
            log.info(f"[Exp {experiment_num}] ✗ No improvement ({val_bpb:.4f} ≥ {best_val_bpb:.4f}) — rolling back")
            shutil.copy2(backup_path, TRAIN_PY)

        time.sleep(SLEEP_BETWEEN)

    # ── Final push and summary ───────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info(f"  LOOP COMPLETE — {experiment_num} experiments run")
    log.info(f"  Final best val_bpb: {best_val_bpb:.4f}")
    log.info("=" * 60)

    try:
        git("push", "origin", "main", check=False)
        log.info("Final push to GitHub complete.")
    except Exception:
        pass

    # ── Morning report + macOS notification ──────────────────────────────────
    try:
        report_log = LOGS_DIR / "morning_report.log"
        with open(report_log, "w") as f:
            subprocess.run(
                [sys.executable, str(REPO_DIR / "morning_report.py"), "--plot"],
                cwd=REPO_DIR, stdout=f, stderr=subprocess.STDOUT,
            )
        log.info(f"Morning report written to {report_log}")
    except Exception as e:
        log.warning(f"Morning report failed: {e}")

    try:
        improvement_pct = (1 - best_val_bpb / 0.4372) * 100
        subprocess.run([
            "osascript", "-e",
            f'display notification "Best val_bpb: {best_val_bpb:.4f} ({improvement_pct:.1f}% improvement) after {experiment_num} experiments." '
            f'with title "AOMB Complete 🧬" subtitle "Check morning_report.log for details" sound name "Glass"',
        ], check=False)
    except Exception:
        pass

    pid_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
