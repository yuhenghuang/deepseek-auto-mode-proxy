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
3. ``messages`` has ≤2 entries — transcript + optional assistant pre-fill; regular conversations have dozens
4. System prompt starts with the classifier's distinctive signature

**Patching:** Injects ``thinking`` and ``output_config`` (controlled by env vars
PROXY_THINKING and PROXY_EFFORT). Strips ``reasoning_effort`` for compatibility.
All other requests pass through unchanged — streaming, tool calls, conversation.

Usage::

    python3 proxy.py [--port PORT] [--upstream URL] [-v|--verbose] [--stop] [--version]

Environment::

    PROXY_THINKING=disabled|enabled  (default: disabled)
    PROXY_EFFORT=low|medium|high     (default: low)
    PROXY_PID_FILE=<path>            (default: tempdir/deepseek-proxy.pid)

Then configure Claude Code::

    ANTHROPIC_BASE_URL=http://127.0.0.1:PORT

The proxy appends ``/anthropic`` to the upstream path automatically, so
ANTHROPIC_BASE_URL must NOT include ``/anthropic``.

Default port is 8799. Override with --port.

Tests: ``python3 -m unittest test_proxy -v`` (offline, stdlib only).
"""

from __future__ import annotations

import argparse
import http.client
import http.server
import json
import logging
import os
import platform
import re
import signal
import ssl
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlparse

log = logging.getLogger("deepseek-proxy")

__version__ = "1.1.0"

DEFAULT_UPSTREAM = "https://api.deepseek.com/anthropic"
DEFAULT_PORT = 8799
_IS_WINDOWS = platform.system() == "Windows"

# PID file holds "<pid> <port>" on one line. The path is overridable via
# PROXY_PID_FILE so tests (or multi-user setups) never touch a shared file.
PID_FILE = os.environ.get("PROXY_PID_FILE") or os.path.join(
    tempfile.gettempdir(), "deepseek-proxy.pid"
)

# Hop-by-hop headers — must not be forwarded in either direction (RFC 2616 §13.5.1)
_HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate",
    "proxy-authorization", "te", "trailer", "transfer-encoding",
    "upgrade", "host",
})

# Response headers the proxy synthesizes itself via send_response() —
# skip the upstream's copies so the client doesn't see duplicates.
_SKIP_RESPONSE_HEADERS = frozenset({"server", "date"})

_VALID_THINKING = frozenset({"disabled", "enabled"})
_VALID_EFFORT = frozenset({"low", "medium", "high"})


# The classifier system prompt always opens with this sentence.
# Verified on Claude Code v2.1.205 (Jul 2026).
# WARNING: may change in future versions — if detection stops working,
# update this string to match the new prompt opening.
_CLASSIFIER_SIGNATURE = (
    "You are a security monitor for autonomous AI coding agents."
)

_tls_context = None


def _get_tls_context() -> ssl.SSLContext:
    """Lazily-built default SSL context, reused across requests.

    Building one per request re-loads the CA store every time (tens of ms).
    """
    global _tls_context
    if _tls_context is None:
        _tls_context = ssl.create_default_context()
    return _tls_context


def _structural_match(body: dict) -> bool:
    """Criteria 1–3: non-streaming, no tools, ≤2 messages."""
    if body.get("stream"):
        return False
    if body.get("tools"):
        return False
    # Classifier has 1-2 messages (transcript + optional assistant pre-fill
    # to force <block> output).  Real conversations have dozens to hundreds.
    return len(body.get("messages", [])) <= 2


def _signature_match(body: dict) -> bool:
    """Criterion 4: system prompt opens with the classifier's signature."""
    system = body.get("system", "")
    if isinstance(system, str):
        return system.startswith(_CLASSIFIER_SIGNATURE)
    if isinstance(system, list) and len(system) > 0:
        first = system[0]
        if isinstance(first, dict):
            text = first.get("text", "")
            if isinstance(text, str):
                return text.startswith(_CLASSIFIER_SIGNATURE)
    return False


def _is_classifier(body) -> bool:
    """Return True if *body* is an auto-mode safety classifier request.

    Uses two independent signals that must both match:
    1. Structural — non-streaming, no tools, ≤2 messages
       (transcript + optional assistant pre-fill since v2.1.160)
    2. Content — system prompt opens with the classifier's distinctive
       signature (unique to the ~350-line security policy)
    """
    if not isinstance(body, dict):
        return False
    return _structural_match(body) and _signature_match(body)


def _patch_classifier(body: dict, thinking: str = "disabled", effort: str = "low") -> dict:
    """Configure thinking and effort for classifier requests.

    Controlled by env vars PROXY_THINKING and PROXY_EFFORT.
    Default: thinking=disabled + effort=low.
    """
    body["thinking"] = {"type": thinking}
    body["output_config"] = {"effort": effort}
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

    # NOTE: protocol_version stays "HTTP/1.0" on purpose — responses are
    # EOF-delimited via `Connection: close`, which is exactly what the
    # streaming path relies on when Content-Length is unknown. Do not switch
    # to 1.1 without re-testing SSE passthrough.

    # Set once at startup by build_server()
    upstream_host: str = "api.deepseek.com"
    upstream_port: int = 443
    upstream_base_path: str = ""
    upstream_use_tls: bool = True
    verbose: bool = False
    # Classifier patching config, set from env vars in main()
    thinking: str = "disabled"
    effort: str = "low"

    # ---------- HTTP methods -------------------------------------------------

    def do_POST(self) -> None:
        """Forward a POST request, patching classifier bodies on the way."""
        raw_header = self.headers.get("Content-Length", "0")
        try:
            content_length = int(raw_header)
        except ValueError:
            log.warning("Malformed Content-Length %r — rejecting", raw_header)
            self._json_response(400, {"error": "invalid content-length"})
            return
        if content_length < 0:
            self._json_response(400, {"error": "invalid content-length"})
            return
        raw_body = self.rfile.read(content_length) if content_length > 0 else b""

        try:
            data = json.loads(raw_body)
        except json.JSONDecodeError:
            log.info("[non-json pass] %s", self.path)
            self._forward("POST", raw_body)
            return

        if not isinstance(data, dict):
            log.info("[non-object pass] %s", self.path)
            self._forward("POST", raw_body)
            return

        if _is_classifier(data):
            # Capture original values before patching
            orig_th = data.get("thinking", "absent")
            orig_re = "yes" if data.get("reasoning_effort") else "no"
            orig_oc = "yes" if data.get("output_config") else "no"
            orig_mt = data.get("max_tokens", "?")
            patched = _patch_classifier(data, self.thinking, self.effort)
            raw_body = json.dumps(patched).encode("utf-8")
            t0 = time.monotonic()
            self._forward("POST", raw_body)
            elapsed = time.monotonic() - t0
            if self.verbose:
                log.info("[classifier] %.1fs re=%s oc=%s th=%s max_tok=%s → thinking=%s effort=%s %s",
                         elapsed, orig_re, orig_oc, orig_th, orig_mt,
                         self.thinking, self.effort, self.path)
            else:
                log.info("[classifier] %.1fs thinking=%s effort=%s %s",
                         elapsed, self.thinking, self.effort, self.path)
            return

        # Structural criteria matched but the signature didn't?
        # This may indicate a Claude Code update changed the prompt.
        if _structural_match(data):
            log.warning(
                "[structural match — possible prompt change] "
                "classifier signature not detected, request passed through unpatched"
            )
        if self.verbose:
            log.info("[pass] stream=%s tools=%d msgs=%d sys=%s re=%s oc=%s %s",
                     bool(data.get("stream")),
                     len(data.get("tools", [])),
                     len(data.get("messages", [])),
                     "yes" if data.get("system") else "no",
                     "yes" if data.get("reasoning_effort") else "no",
                     "yes" if data.get("output_config") else "no",
                     self.path)
        else:
            log.info("[pass] %s", self.path)
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
        headers_sent = False
        if self.upstream_use_tls:
            conn: http.client.HTTPConnection = http.client.HTTPSConnection(
                self.upstream_host, self.upstream_port,
                context=_get_tls_context(), timeout=120,
            )
        else:
            conn = http.client.HTTPConnection(
                self.upstream_host, self.upstream_port, timeout=120,
            )

        try:
            # Collect headers to forward (skip hop-by-hop)
            fwd_headers: dict[str, str] = {}
            _forward_headers(self.headers, fwd_headers)
            # Always set Content-Length ourselves so it matches the bytes we
            # actually send (drop any client copy, regardless of header case).
            fwd_headers.pop("Content-Length", None)
            fwd_headers.pop("content-length", None)
            fwd_headers["Content-Length"] = str(len(body))

            upstream_path = self.upstream_base_path + self.path
            conn.request(method, upstream_path, body=body, headers=fwd_headers)
            resp = conn.getresponse()

            # Send status line
            self.send_response(resp.status)

            # Copy response headers (skip hop-by-hop and self-synthesized ones)
            for key, value in resp.headers.items():
                if (key.lower() not in _HOP_BY_HOP
                        and key.lower() not in _SKIP_RESPONSE_HEADERS):
                    self.send_header(key, value)

            # Ensure connection closes so the client sees EOF (important for
            # streaming responses where Content-Length is unknown).
            self.send_header("Connection", "close")
            self.close_connection = True

            self.end_headers()
            headers_sent = True

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
            # Only a 502 if nothing has been sent yet — once the response
            # has started there is no way to change its status.
            if not headers_sent:
                try:
                    self._json_response(502, {"error": "upstream unreachable"})
                except Exception:
                    pass  # best-effort when the connection is already broken
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


def build_server(port: int, upstream_url: str = DEFAULT_UPSTREAM,
                 verbose: bool = False) -> _ThreadedHTTPServer:
    """Construct a proxy server without touching process-level state.

    port=0 binds an ephemeral port (used by tests). Raises ValueError for an
    invalid upstream URL and OSError if the port cannot be bound.
    """
    parsed = urlparse(upstream_url)
    if not parsed.hostname:
        raise ValueError(f"Invalid upstream URL: {upstream_url!r}")

    ProxyHandler.upstream_host = parsed.hostname
    ProxyHandler.upstream_port = parsed.port or (443 if parsed.scheme == "https" else 80)
    ProxyHandler.upstream_base_path = parsed.path or ""
    ProxyHandler.upstream_use_tls = parsed.scheme == "https"
    ProxyHandler.verbose = verbose

    return _ThreadedHTTPServer(("127.0.0.1", port), ProxyHandler)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _env_value(name: str, default: str, valid) -> str:
    """Read an env var; warn (and keep the default) if the value is invalid."""
    value = os.environ.get(name, default)
    if value not in valid:
        log.warning("Invalid %s=%r (expected one of %s) — using default %r",
                    name, value, ", ".join(sorted(valid)), default)
        return default
    return value


def _write_pid_file(pid: int, port: int) -> None:
    """Record "<pid> <port>" so --stop can find the instance later."""
    with open(PID_FILE, "w") as f:
        f.write(f"{pid} {port}\n")


def _read_pid_file() -> tuple[int, int] | None:
    """Return (pid, port) from the PID file, or None if absent/malformed."""
    try:
        with open(PID_FILE) as f:
            parts = f.read().strip().split()
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
    except (OSError, ValueError):
        pass
    return None


def _remove_pid_file_if_own(pid: int, port: int) -> None:
    """Remove the PID file only if it still describes *this* instance.

    Guards against racing another instance that rewrote the file.
    """
    if _read_pid_file() == (pid, port):
        try:
            os.remove(PID_FILE)
        except OSError:
            pass


def _process_alive(pid: int) -> bool:
    """Return True if a process with *pid* is running."""
    if _IS_WINDOWS:
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True
            )
            return str(pid) in result.stdout
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True  # exists, but owned by another user


def _kill_process(pid: int) -> bool:
    """Terminate a process by PID, cross-platform. Returns success."""
    try:
        if _IS_WINDOWS:
            result = subprocess.run(
                ["taskkill", "/PID", str(pid), "/F"],
                capture_output=True,
            )
            return result.returncode == 0
        os.kill(pid, signal.SIGTERM)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def _find_pid_by_port(port: int) -> int | None:
    """Return the PID of the process listening on *port*, or None."""
    # Match on ":<port> " (trailing space) so :8799 doesn't match :18799.
    marker = f":{port} "
    if _IS_WINDOWS:
        try:
            result = subprocess.run(
                ["netstat", "-ano"], capture_output=True, text=True
            )
            for line in result.stdout.splitlines():
                if marker in line and "LISTENING" in line:
                    # netstat -ano columns: Proto  Local Address  Foreign  State  PID
                    parts = line.split()
                    return int(parts[-1])
            return None
        except Exception:
            return None
    if platform.system() == "Darwin":
        # macOS lacks `ss`; lsof is the standard listener lookup.
        try:
            result = subprocess.run(
                ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
                capture_output=True, text=True,
            )
            first = result.stdout.splitlines()[0] if result.stdout.strip() else None
            return int(first) if first else None
        except Exception:
            return None
    try:
        result = subprocess.run(
            ["ss", "-tlnp"], capture_output=True, text=True
        )
        for line in result.stdout.splitlines():
            if marker in line:
                # ss -tlnp output: ... users:(("proc",pid=1234,fd=3))
                m = re.search(r"pid=(\d+)", line)
                if m:
                    return int(m.group(1))
        return None
    except Exception:
        return None


def _stop_existing(port: int) -> None:
    """Stop a running proxy instance on *port* — never an unrelated process.

    Primary: the PID bound to the port (authoritative — it owns the port, so
    it cannot be an unrelated process). Fallback: the PID file entry, only
    when its recorded port matches *and* the process is still alive.
    """
    pid = _find_pid_by_port(port)
    if pid is not None:
        if _kill_process(pid):
            log.info("Stopped proxy (PID %d bound to port %d)", pid, port)
        else:
            log.warning("Failed to stop PID %d bound to port %d", pid, port)
        return

    entry = _read_pid_file()
    if entry is not None:
        file_pid, file_port = entry
        if file_port == port:
            if _process_alive(file_pid):
                if _kill_process(file_pid):
                    log.info("Stopped proxy (PID %d from %s)", file_pid, PID_FILE)
                else:
                    log.warning("Failed to stop PID %d from %s", file_pid, PID_FILE)
                    return
            else:
                log.info("PID file stale (PID %d not running); removing", file_pid)
            _remove_pid_file_if_own(file_pid, file_port)
            return

    log.warning("No running proxy found on port %d", port)


def _install_signal_handlers(server) -> None:
    """Make SIGTERM shut the server down cleanly (PID file cleanup)."""
    if _IS_WINDOWS:
        return

    def _shutdown(signum, frame) -> None:
        log.info("Received SIGTERM, shutting down")
        # shutdown() must run outside the serve_forever thread.
        threading.Thread(target=server.shutdown, daemon=True).start()

    try:
        signal.signal(signal.SIGTERM, _shutdown)
    except (ValueError, OSError):
        pass  # e.g. not called from the main thread


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeepSeek auto-mode classifier proxy"
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help="Listen port (default: %(default)s)",
    )
    parser.add_argument(
        "--upstream", type=str, default=DEFAULT_UPSTREAM,
        help="Upstream API base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--stop", action="store_true",
        help="Stop any running proxy on the given port and exit",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Log request structure details for debugging",
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

    # Load classifier config from env; warn and keep defaults on bad values.
    ProxyHandler.thinking = _env_value("PROXY_THINKING", "disabled", _VALID_THINKING)
    ProxyHandler.effort = _env_value("PROXY_EFFORT", "low", _VALID_EFFORT)

    try:
        server = build_server(args.port, args.upstream, args.verbose)
    except ValueError as exc:
        parser.error(str(exc))
    except OSError as exc:
        log.error("Cannot bind to port %d: %s", args.port, exc)
        log.error(
            "The proxy may already be running. Check with: "
            "curl http://127.0.0.1:%d/health",
            args.port,
        )
        raise SystemExit(1)

    _write_pid_file(os.getpid(), args.port)
    _install_signal_handlers(server)

    log.info("DeepSeek auto-mode proxy %s listening on http://127.0.0.1:%d",
             __version__, args.port)
    log.info("Upstream: %s", args.upstream)
    log.info(
        "Configure Claude Code with:  ANTHROPIC_BASE_URL=http://127.0.0.1:%d",
        args.port,
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down")
    finally:
        server.server_close()
        _remove_pid_file_if_own(os.getpid(), args.port)


if __name__ == "__main__":
    main()
