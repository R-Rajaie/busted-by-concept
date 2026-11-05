"""A small local web app for the coach: a chat page on http://127.0.0.1:8765.

uv run python -m coach.serve.web [--port 8765]

Uses only the standard library's HTTP server. It listens on localhost alone, keeps one Coach
(with its conversation history) per browser tab, and answers one question at a time per tab.
"""

import argparse
import json
import logging
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from coach.serve.agent import Coach

log = logging.getLogger("coach.serve.web")
PAGE = Path(__file__).with_name("static") / "index.html"
MAX_QUESTION = 2000


class Sessions:
    """One Coach per browser tab, keyed by an id the page makes up."""

    def __init__(self) -> None:
        self._coaches: dict[str, Coach] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def get(self, sid: str, riot_id: str | None, platform: str) -> tuple[Coach, threading.Lock]:
        with self._guard:
            coach = self._coaches.get(sid)
            if coach is None:
                coach = self._coaches[sid] = Coach()
                self._locks[sid] = threading.Lock()
            coach.riot_id, coach.platform = riot_id, platform  # the page may change these
            return coach, self._locks[sid]

    def reset(self, sid: str) -> None:
        with self._guard:
            self._coaches.pop(sid, None)


SESSIONS = Sessions()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, kind: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, data: dict) -> None:
        self._send(code, json.dumps(data).encode("utf-8"), "application/json")

    def do_GET(self) -> None:  # noqa: N802 (http.server's naming)
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(min(n, 1_000_000)) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._json(400, {"error": "bad request"})
        sid = str(req.get("session") or "")
        try:
            uuid.UUID(sid)
        except ValueError:
            return self._json(400, {"error": "bad session id"})

        if self.path == "/api/reset":
            SESSIONS.reset(sid)
            return self._json(200, {"ok": True})
        if self.path != "/api/ask":
            return self._json(404, {"error": "not found"})

        question = str(req.get("question") or "").strip()[:MAX_QUESTION]
        if not question:
            return self._json(400, {"error": "empty question"})
        riot_id = str(req.get("riot_id") or "").strip() or None
        platform = str(req.get("platform") or "na1").strip() or "na1"
        coach, lock = SESSIONS.get(sid, riot_id, platform)
        if not lock.acquire(blocking=False):
            return self._json(409, {"error": "still answering your last question"})
        try:
            answer = coach.ask(question)
        except Exception as e:  # shown in the page instead of a dead request
            log.exception("ask failed")
            return self._json(500, {"error": f"{type(e).__name__}: {e}"})
        finally:
            lock.release()
        self._json(200, {"answer": answer})

    def log_message(self, fmt: str, *args) -> None:
        log.info("%s %s", self.address_string(), fmt % args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="coach.serve.web", description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("-v", "--verbose", action="store_true", help="log requests and tools")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Coach running at http://127.0.0.1:{args.port}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
