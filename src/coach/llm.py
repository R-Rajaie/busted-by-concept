"""LLM calls via the Claude Code CLI in headless mode (`claude -p`).

This runs on the user's Claude subscription, so no API key is needed. Each call is a
fresh, tool-less session with our own system prompt, which keeps overhead small.
"""

import json
import logging
import shutil
import subprocess
import time
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
FATAL = ("authenticate", "/login", "usage limit")


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
    cmd = [
        _claude_exe(),
        "-p",
        "--model",
        model,
        "--system-prompt",
        system,
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

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
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
                raise LLMError(f"claude exited {proc.returncode}: {err}")
            if schema is None:
                return out["result"]
            if out.get("structured_output") is not None:
                return out["structured_output"]
            return json.loads(out["result"])
        except LLMStop:
            raise
        except (LLMError, json.JSONDecodeError, subprocess.TimeoutExpired, KeyError) as e:
            last_err = e
            log.warning("llm call failed (attempt %d/%d): %s", attempt, retries, e)
            time.sleep(5 * attempt)
    raise LLMError(f"llm call failed after {retries} attempts") from last_err


def complete_model[T: BaseModel](
    prompt: str, system: str, output: type[T], model: str = FAST_MODEL
) -> T:
    """Like `complete`, but validates the result against a Pydantic model."""
    data = complete(prompt, system, model=model, schema=output.model_json_schema())
    return output.model_validate(data)
