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
            if proc.returncode != 0:
                raise LLMError(f"claude exited {proc.returncode}: {proc.stderr[-500:]}")
            out = json.loads(proc.stdout)
            if out.get("is_error"):
                raise LLMError(f"claude error: {out.get('result')}")
            if schema is None:
                return out["result"]
            if out.get("structured_output") is not None:
                return out["structured_output"]
            return json.loads(out["result"])
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
