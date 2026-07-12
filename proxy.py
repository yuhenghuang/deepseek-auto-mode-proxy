#!/usr/bin/env python3
"""HTTP proxy that controls DeepSeek thinking on auto-mode classifier requests.

Claude Code's auto mode sends safety classification requests that take ~30 seconds
when DeepSeek V4 Pro's thinking mode is enabled, hitting the internal timeout.

**Root cause:** DeepSeek V4 Pro enables thinking by default on all requests. The
classifier receives a ~350-line security policy as system prompt plus a ~200K-char
transcript as the user message. With thinking enabled and a high max_tokens, the
model generates thousands of thinking tokens — taking 28–32s. DeepSeek ignores
``budget_tokens`` and ``output_config.effort`` doesn't constrain thinking, so the
only effective lever is ``thinking: { type: "disabled" }``.

**Detection (4 criteria, all must match):**
1. ``stream`` is not ``true`` — classifier is non-streaming
2. ``tools`` is absent/empty — classifier has no tool definitions
3. ``messages`` has exactly 1 entry — single transcript message
4. System prompt starts with the classifier's distinctive signature

**Patching:** Injects ``thinking`` and ``output_config`` (controlled by env vars
PROXY_THINKING and PROXY_EFFORT). Strips ``reasoning_effort`` for compatibility.
All other requests pass through unchanged — streaming, tool calls, conversation.

Usage::

    python3 proxy.py [--port PORT] [--upstream URL]

Environment::

    PROXY_THINKING=disabled|enabled  (default: enabled)
    PROXY_EFFORT=low|medium|high     (default: medium)

Then configure Claude Code::

    ANTHROPIC_BASE_URL=http://127.0.0.1:PORT

The proxy appends ``/anthropic`` to the upstream path automatically, so
ANTHROPIC_BASE_URL must NOT include ``/anthropic``.

Default port is 8799. Override with --port.
"""

from __future__ import annotations

import argparse
import http.client
import http.server
import json
import logging
import os
import signal
import ssl
import sys
from urllib.parse import urlparse

log = logging.getLogger("deepseek-proxy")

DEFAULT_UPSTREAM = "https://api.deepseek.com/anthropic"
PID_FILE = "/tmp/deepseek-proxy.pid"

# Hop-by-hop headers — must not be forwarded in either direction (RFC 2616 §13.5.1)
_HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate",
    "proxy-authorization", "te", "trailer", "transfer-encoding",
    "upgrade", "host",
})


# The classifier system prompt always opens with this exact sentence.
# Using it as a fingerprint is far more reliable than structural heuristics
# (message count, stream flag, tools) which could match edge cases.
_CLASSIFIER_SIGNATURE = (
    "You are a security monitor for autonomous AI coding agents."
)

# Default patching behaviour — can be overridden via env vars.
# PROXY_THINKING: "disabled" | "enabled" (default: "enabled")
# PROXY_EFFORT: "low" | "medium" | "high" (default: "medium")
_PROXY_THINKING = os.environ.get("PROXY_THINKING", "enabled")
_PROXY_EFFORT = os.environ.get("PROXY_EFFORT", "medium")


def _is_classifier(body: dict) -> bool:
    """Return True if *body* is an auto-mode safety classifier request.

    Uses two independent signals that must both match:
    1. Structural — non-streaming, no tools, exactly one message
       (the battle-tested heuristic from deepseek-claude-proxy)
    2. Content — system prompt opens with the classifier's distinctive
       signature (unique to the ~350-line security policy)
    """
    # Structural check
    if body.get("stream"):
        return False
    if body.get("tools"):
        return False
    if len(body.get("messages", [])) != 1:
        return False

    # Content check — system prompt fingerprint
    system = body.get("system", "")
    if isinstance(system, str) and system.startswith(_CLASSIFIER_SIGNATURE):
        return True
    if isinstance(system, list) and len(system) > 0:
        first_text = system[0].get("text", "") if isinstance(system[0], dict) else ""
        if isinstance(first_text, str) and first_text.startswith(_CLASSIFIER_SIGNATURE):
            return True

    return False


def _patch_classifier(body: dict) -> dict:
    """Configure thinking and effort for classifier requests.

    Controlled by env vars PROXY_THINKING and PROXY_EFFORT.
    Default: thinking=enabled + effort=medium.
    Set PROXY_THINKING=disabled to fully disable thinking.
    """
    body["thinking"] = {"type": _PROXY_THINKING}
    body["output_config"] = {"effort": _PROXY_EFFORT}
    # Strip legacy params that conflict with thinking
    body.pop("reasoning_effort", None)
    return body


def _forward_headers(source, sink_dict: dict[str, str]) -> None:
    """Copy headers from *source* to *sink_dict*, skipping hop-by-hop."""
    for key, value in source.items():
        if key.lower() not in _HOP_BY_HOP:
            sink_dict[key] = value


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    """Forward Anthropic Messages API requests → DeepSeek, patching classifiers."""

    # Set once at startup by main()
    upstream_host: str = "api.deepseek.com"
    upstream_port: int = 443
    upstream_base_path: str = ""
    upstream_use_tls: bool = True

    # ---------- HTTP methods -------------------------------------------------

    def do_POST(self) -> None:
        """Forward a POST request, patching classifier bodies on the way."""
        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length) if content_length > 0 else b""

        try:
            data = json.loads(raw_body)
            if _is_classifier(data):
                data = _patch_classifier(data)
                raw_body = json.dumps(data).encode("utf-8")
                log.info("[classifier] thinking=%s effort=%s %s", _PROXY_THINKING, _PROXY_EFFORT, self.path)
            else:
                # Check: structural criteria match but signature failed?
                # This may indicate a Claude Code update changed the prompt.
                if not data.get("stream") and not data.get("tools") and len(data.get("messages", [])) == 1:
                    log.warning(
                        "[structural match — possible prompt change] "
                        "classifier signature not detected, request passed through unpatched"
                    )
                log.info("[pass] %s", self.path)
        except json.JSONDecodeError:
            log.info("[non-json pass] %s", self.path)

        self._forward("POST", raw_body)

    def do_GET(self) -> None:
        """Health-check endpoint."""
        if self.path == "/health":
            self._json_response(200, {"ok": True})
        else:
            self._json_response(404, {"error": "not found"})

    # ---------- Internals ----------------------------------------------------

    def _forward(self, method: str, body: bytes) -> None:
        """Forward *method* request with *body* to upstream; stream response back."""
        # Build upstream connection
        conn: http.client.HTTPConnection
        if self.upstream_use_tls:
            ctx = ssl.create_default_context()
            conn = http.client.HTTPSConnection(
                self.upstream_host, self.upstream_port,
                context=ctx, timeout=120,
            )
        else:
            conn = http.client.HTTPConnection(
                self.upstream_host, self.upstream_port, timeout=120,
            )

        try:
            # Collect headers to forward (skip hop-by-hop)
            fwd_headers: dict[str, str] = {}
            _forward_headers(self.headers, fwd_headers)
            if body:
                fwd_headers["Content-Length"] = str(len(body))

            upstream_path = self.upstream_base_path + self.path
            conn.request(method, upstream_path, body=body, headers=fwd_headers)
            resp = conn.getresponse()

            # Send status line
            self.send_response(resp.status)

            # Copy response headers (skip hop-by-hop)
            for key, value in resp.headers.items():
                if key.lower() not in _HOP_BY_HOP:
                    self.send_header(key, value)

            # Ensure connection closes so the client sees EOF (important for
            # streaming responses where Content-Length is unknown).
            self.send_header("Connection", "close")
            self.close_connection = True

            self.end_headers()

            # Stream body in chunks — works for both chunked (SSE) and
            # Content-Length-delimited upstream responses because
            # http.client de-chunks internally.
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()

        except Exception as exc:
            log.error("Upstream unreachable: %s", exc)
            try:
                self._json_response(502, {"error": "upstream unreachable"})
            except Exception:
                pass  # best-effort when the response has already started
        finally:
            conn.close()

    def _json_response(self, status: int, data: dict) -> None:
        """Send a small JSON response (health checks, errors)."""
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        """Suppress default per-request log line (we emit our own)."""
        pass


class _ThreadedHTTPServer(http.server.ThreadingHTTPServer):
    """Handle requests concurrently so one slow response doesn't block others."""
    allow_reuse_address = True
    daemon_threads = True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _stop_existing(port: int) -> None:
    """Stop a running proxy instance on *port* by PID file or port match."""
    # Try PID file first
    try:
        with open(PID_FILE) as f:
            pid = int(f.read().strip())
        os.kill(pid, signal.SIGTERM)
        log.info("Stopped proxy (PID %d from %s)", pid, PID_FILE)
        os.remove(PID_FILE)
        return
    except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
        pass

    # Fallback: find by port
    import subprocess
    try:
        result = subprocess.run(
            ["ss", "-tlnp"], capture_output=True, text=True
        )
        for line in result.stdout.splitlines():
            if f":{port}" in line:
                # Extract PID from ss output: pid=12345
                for part in line.split():
                    if part.startswith("pid="):
                        pid = int(part.split("=")[1].rstrip(","))
                        os.kill(pid, signal.SIGTERM)
                        log.info("Stopped proxy (PID %d bound to port %d)", pid, port)
                        return
        log.warning("No running proxy found on port %d", port)
    except Exception as exc:
        log.error("Failed to stop existing proxy: %s", exc)
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeepSeek auto-mode classifier proxy"
    )
    parser.add_argument(
        "--port", type=int, default=8799,
        help="Listen port (default: 8799)",
    )
    parser.add_argument(
        "--upstream", type=str, default=DEFAULT_UPSTREAM,
        help="Upstream API base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--stop", action="store_true",
        help="Stop any running proxy on the given port and exit",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.stop:
        _stop_existing(args.port)
        return

    # Parse upstream into static handler attributes (avoids re-parsing per request)
    parsed = urlparse(args.upstream)
    if not parsed.hostname:
        parser.error(f"Invalid upstream URL: {args.upstream!r}")

    ProxyHandler.upstream_host = parsed.hostname
    ProxyHandler.upstream_port = parsed.port or (443 if parsed.scheme == "https" else 80)
    ProxyHandler.upstream_base_path = parsed.path or ""
    ProxyHandler.upstream_use_tls = parsed.scheme == "https"

    try:
        server = _ThreadedHTTPServer(("127.0.0.1", args.port), ProxyHandler)
    except OSError as exc:
        log.error("Cannot bind to port %d: %s", args.port, exc)
        log.error(
            "The proxy may already be running. Check with: "
            "curl http://127.0.0.1:%d/health",
            args.port,
        )
        raise SystemExit(1)

    # Write PID file for --stop support
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    log.info("DeepSeek auto-mode proxy listening on http://127.0.0.1:%d", args.port)
    log.info("Upstream: %s", args.upstream)
    log.info(
        "Configure Claude Code with:  ANTHROPIC_BASE_URL=http://127.0.0.1:%d",
        args.port,
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down")
        server.shutdown()
    finally:
        try:
            os.remove(PID_FILE)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
