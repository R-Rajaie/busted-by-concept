"""LLM calls via the Claude Code CLI in headless mode (`claude -p`).

This runs on the user's Claude subscription, so no API key is needed. Each call is a
fresh, tool-less session with our own system prompt, which keeps overhead small.
"""

import json
import logging
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel

log = logging.getLogger(__name__)

# Aliases understood by `claude --model`. Bulk passes use the cheap model.
FAST_MODEL = "haiku"
SMART_MODEL = "sonnet"


class LLMError(RuntimeError):
    pass


class LLMStop(LLMError):
    """Every later call would fail the same way (logged out, usage limit): stop the run."""


# Substrings of claude's error text that no retry or next episode can get past.
FATAL = ("authenticate", "/login")
# A plan limit ("You've hit your session limit · resets 1am (America/Los_Angeles)"): wait for
# the reset and carry on, so an unattended run doesn't fail every episode until morning.
LIMIT = re.compile(r"(session|usage|weekly|rate) limit|limit reached|hit your", re.IGNORECASE)
RESET = re.compile(r"resets\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)", re.IGNORECASE)
MAX_WAIT_S = 6 * 3600  # longer than this (a weekly limit, say) stops the run instead


def _seconds_until_reset(err: str, now: datetime | None = None) -> float | None:
    """Seconds until the reset time in a limit message, or None if it names none. The time is
    read as local time; the machine and the plan's time zone are assumed to match."""
    m = RESET.search(err)
    if not m:
        return None
    hour = int(m.group(1)) % 12 + (12 if m.group(3).lower() == "pm" else 0)
    now = now or datetime.now()
    reset = now.replace(hour=hour, minute=int(m.group(2) or 0), second=0, microsecond=0)
    if reset <= now:
        reset += timedelta(days=1)
    return (reset - now).total_seconds() + 120  # a little slack past the reset


def _error_text(proc: subprocess.CompletedProcess) -> str:
    """claude -p --output-format json reports most errors on stdout, in `result`."""
    try:
        return str(json.loads(proc.stdout).get("result") or "")
    except (json.JSONDecodeError, AttributeError):
        return (proc.stderr or proc.stdout)[-500:]


def _claude_exe() -> str:
    exe = shutil.which("claude")
    if not exe:
        raise LLMError("`claude` CLI not found on PATH; install Claude Code")
    return exe


def complete(
    prompt: str,
    system: str,
    model: str = FAST_MODEL,
    schema: dict[str, Any] | None = None,
    retries: int = 3,
    timeout: float = 600,
) -> Any:
    """Run one prompt. Returns the parsed object when `schema` is given, else the text."""
    # The system prompt goes in a file: with the concept list it outgrows Windows' 32K
    # command line.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "system.txt"
        path.write_text(system, encoding="utf-8")
        return _complete(prompt, path, model, schema, retries, timeout)


def _complete(
    prompt: str,
    system_file: Path,
    model: str,
    schema: dict[str, Any] | None,
    retries: int,
    timeout: float,
) -> Any:
    cmd = [
        _claude_exe(),
        "-p",
        "--model",
        model,
        "--system-prompt-file",
        str(system_file),
        "--tools",
        "",
        "--strict-mcp-config",
        "--setting-sources",
        "",
        "--no-session-persistence",
        "--output-format",
        "json",
    ]
    if schema is not None:
        cmd += ["--json-schema", json.dumps(schema)]

    attempt = 0
    while True:
        try:
            proc = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=timeout,
            )
            out = json.loads(proc.stdout) if proc.returncode == 0 else {}
            if proc.returncode != 0 or out.get("is_error"):
                err = _error_text(proc)
                if any(f in err.lower() for f in FATAL):
                    raise LLMStop(f"claude: {err}")
                if LIMIT.search(err):
                    wait = _seconds_until_reset(err)
                    if wait is None or wait > MAX_WAIT_S:
                        raise LLMStop(f"claude: {err}")
                    log.warning("%s; waiting %.0f min for the reset", err, wait / 60)
                    time.sleep(wait)
                    continue  # waiting doesn't use up a retry
                raise LLMError(f"claude exited {proc.returncode}: {err}")
            if schema is None:
                return out["result"]
            if out.get("structured_output") is not None:
                return out["structured_output"]
            return json.loads(out["result"])
        except LLMStop:
            raise
        except (LLMError, json.JSONDecodeError, subprocess.TimeoutExpired, KeyError) as e:
            attempt += 1
            log.warning("llm call failed (attempt %d/%d): %s", attempt, retries, e)
            if attempt >= retries:
                raise LLMError(f"llm call failed after {retries} attempts") from e
            time.sleep(5 * attempt)


def complete_model[T: BaseModel](
    prompt: str, system: str, output: type[T], model: str = FAST_MODEL
) -> T:
    """Like `complete`, but validates the result against a Pydantic model."""
    data = complete(prompt, system, model=model, schema=output.model_json_schema())
    return output.model_validate(data)
