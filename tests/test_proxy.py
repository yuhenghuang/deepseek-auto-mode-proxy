#!/usr/bin/env python3
"""Offline test suite for the DeepSeek auto-mode proxy.

Stdlib ``unittest`` only — no pytest, no requests. The suite never touches
the live PID file (``proxy.PID_FILE`` is repointed to a temp path) and never
hits the real DeepSeek API (a local fake upstream stands in).

Run from the repo root with::

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import http.client
import http.server
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
import proxy  # noqa: E402

_PROXY_PY = os.path.join(_REPO_ROOT, "proxy.py")
_TMPDIR = tempfile.mkdtemp(prefix="deepseek-proxy-test-")

_CLASSIFIER = proxy._CLASSIFIER_SIGNATURE


def setUpModule():
    """Never let tests touch the real PID file."""
    proxy.PID_FILE = os.path.join(_TMPDIR, "proxy-test.pid")


def tearDownModule():
    shutil.rmtree(_TMPDIR, ignore_errors=True)


def _free_port() -> int:
    """Return a currently-free port (racy, fine for tests)."""
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


# ---------------------------------------------------------------------------
# Fake upstream — records every request, replies with a canned response
# ---------------------------------------------------------------------------

class _FakeUpstreamHandler(http.server.BaseHTTPRequestHandler):
    """A minimal Anthropic-compatible upstream that records what it receives."""

    requests: list = []  # (method, path, dict(headers), raw body bytes)
    status = 200
    response_headers = [("Content-Type", "application/json")]
    response_body = b'{"id":"upstream"}'

    def _record_and_reply(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length > 0 else b""
        self.__class__.requests.append(
            (self.command, self.path, dict(self.headers), body)
        )
        self.send_response(self.status)
        for key, value in self.response_headers:
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(self.response_body)))
        self.end_headers()
        self.wfile.write(self.response_body)

    do_POST = _record_and_reply
    do_GET = _record_and_reply

    def log_message(self, *args) -> None:
        pass


# ---------------------------------------------------------------------------
# Integration harness — proxy + fake upstream on ephemeral ports
# ---------------------------------------------------------------------------

class ProxyTestCase(unittest.TestCase):
    """Base class: a real proxy in a thread in front of the fake upstream."""

    @classmethod
    def setUpClass(cls):
        cls.upstream = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), _FakeUpstreamHandler
        )
        cls.upstream_port = cls.upstream.server_address[1]
        cls.proxy_server = proxy.build_server(
            0, f"http://127.0.0.1:{cls.upstream_port}", verbose=False
        )
        cls.proxy_port = cls.proxy_server.server_address[1]
        cls.upstream_thread = threading.Thread(
            target=cls.upstream.serve_forever, daemon=True
        )
        cls.proxy_thread = threading.Thread(
            target=cls.proxy_server.serve_forever, daemon=True
        )
        cls.upstream_thread.start()
        cls.proxy_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.upstream.shutdown()
        cls.upstream.server_close()
        cls.proxy_server.shutdown()
        cls.proxy_server.server_close()

    def setUp(self):
        _FakeUpstreamHandler.requests.clear()

    def post(self, body: bytes, path: str = "/v1/messages"):
        """POST *body* through the proxy; return (status, headers, body)."""
        conn = http.client.HTTPConnection("127.0.0.1", self.proxy_port, timeout=10)
        try:
            conn.request("POST", path, body=body,
                         headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            return resp.status, dict(resp.headers), resp.read()
        finally:
            conn.close()

    def get(self, path: str):
        conn = http.client.HTTPConnection("127.0.0.1", self.proxy_port, timeout=10)
        try:
            conn.request("GET", path)
            resp = conn.getresponse()
            return resp.status, dict(resp.headers), resp.read()
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Unit tests — detection
# ---------------------------------------------------------------------------

def _classifier_body(**overrides) -> dict:
    body = {
        "model": "deepseek-v4-flash[1m]",
        "max_tokens": 2112,
        "stream": False,
        "messages": [{"role": "user", "content": "t" * 100}],
        "system": _CLASSIFIER + " 350 lines of security policy...",
    }
    body.update(overrides)
    return body


class TestClassifierDetection(unittest.TestCase):
    def test_stream_true(self):
        self.assertFalse(proxy._is_classifier(_classifier_body(stream=True)))

    def test_tools_present(self):
        self.assertFalse(proxy._is_classifier(
            _classifier_body(tools=[{"name": "Bash"}])
        ))

    def test_three_messages(self):
        self.assertFalse(proxy._is_classifier(
            _classifier_body(messages=[{}, {}, {}])
        ))

    def test_empty_messages(self):
        self.assertTrue(proxy._is_classifier(_classifier_body(messages=[])))

    def test_signature_str_detected(self):
        self.assertTrue(proxy._is_classifier(_classifier_body()))

    def test_signature_list_of_dicts(self):
        system = [{"type": "text", "text": _CLASSIFIER + " policy"}]
        self.assertTrue(proxy._is_classifier(_classifier_body(system=system)))

    def test_signature_not_first_position(self):
        # Signature must open the FIRST block — later blocks don't count.
        system = [
            {"type": "text", "text": "Something else entirely"},
            {"type": "text", "text": _CLASSIFIER},
        ]
        self.assertFalse(proxy._is_classifier(_classifier_body(system=system)))

    def test_no_system(self):
        self.assertFalse(proxy._is_classifier(_classifier_body(system="")))

    def test_non_dict_input(self):
        for bad in ([1, 2, 3], "text", None, 42):
            self.assertFalse(proxy._is_classifier(bad))


class TestPatchClassifier(unittest.TestCase):
    def test_defaults(self):
        out = proxy._patch_classifier({"messages": []})
        self.assertEqual(out["thinking"], {"type": "disabled"})
        self.assertEqual(out["output_config"], {"effort": "low"})

    def test_custom_thinking_and_effort(self):
        out = proxy._patch_classifier(
            {"messages": []}, thinking="enabled", effort="high"
        )
        self.assertEqual(out["thinking"], {"type": "enabled"})
        self.assertEqual(out["output_config"], {"effort": "high"})

    def test_strips_reasoning_effort(self):
        out = proxy._patch_classifier({"messages": [], "reasoning_effort": "high"})
        self.assertNotIn("reasoning_effort", out)

    def test_preserves_other_fields(self):
        body = {"model": "m", "messages": [], "custom": {"nested": 1}}
        out = proxy._patch_classifier(body)
        self.assertEqual(out["model"], "m")
        self.assertEqual(out["custom"], {"nested": 1})


# ---------------------------------------------------------------------------
# Integration tests — passthrough behaviour
# ---------------------------------------------------------------------------

class TestProxyPassthrough(ProxyTestCase):
    def test_non_classifier_body_byte_identical(self):
        body = json.dumps({
            "model": "deepseek-v4-flash[1m]",
            "stream": True,
            "messages": [{"role": "user", "content": "hello there"}],
        }).encode()
        status, _, resp_body = self.post(body)
        self.assertEqual(status, 200)
        self.assertEqual(resp_body, _FakeUpstreamHandler.response_body)
        _, _, _, up_body = _FakeUpstreamHandler.requests[0]
        self.assertEqual(up_body, body)  # byte-identical to what the client sent

    def test_non_object_body_passthrough(self):
        body = b"[1, 2, 3]"  # valid JSON, but not a dict — must not crash
        status, _, _ = self.post(body)
        self.assertEqual(status, 200)
        self.assertEqual(_FakeUpstreamHandler.requests[0][3], body)

    def test_empty_body_post(self):
        status, _, _ = self.post(b"")
        self.assertEqual(status, 200)

    def test_streaming_sse_passthrough(self):
        _FakeUpstreamHandler.response_headers = [("Content-Type", "text/event-stream")]
        _FakeUpstreamHandler.response_body = b'data: {"a":1}\n\ndata: {"b":2}\n\n'
        try:
            status, headers, resp_body = self.post(b'{"stream": true, "messages": []}')
            self.assertEqual(status, 200)
            self.assertEqual(headers.get("Content-Type"), "text/event-stream")
            self.assertEqual(resp_body, _FakeUpstreamHandler.response_body)
        finally:
            _FakeUpstreamHandler.response_headers = [("Content-Type", "application/json")]
            _FakeUpstreamHandler.response_body = b'{"id":"upstream"}'

    def test_health_ok(self):
        status, _, body = self.get("/health")
        self.assertEqual(status, 200)
        self.assertEqual(body, b'{"ok": true}')

    def test_unknown_get_404(self):
        status, _, _ = self.get("/nope")
        self.assertEqual(status, 404)

    def test_502_when_upstream_down(self):
        # Save/restore handler attrs: build_server mutates them globally.
        saved = (proxy.ProxyHandler.upstream_host, proxy.ProxyHandler.upstream_port,
                 proxy.ProxyHandler.upstream_base_path,
                 proxy.ProxyHandler.upstream_use_tls)
        down_server = None
        try:
            down_server = proxy.build_server(
                0, f"http://127.0.0.1:{_free_port()}", verbose=False
            )
            threading.Thread(target=down_server.serve_forever, daemon=True).start()
            conn = http.client.HTTPConnection(
                "127.0.0.1", down_server.server_address[1], timeout=10
            )
            try:
                conn.request("POST", "/v1/messages", body=b'{"stream": true, "messages": []}',
                             headers={"Content-Type": "application/json"})
                resp = conn.getresponse()
                self.assertEqual(resp.status, 502)
                self.assertIn(b"upstream unreachable", resp.read())
            finally:
                conn.close()
        finally:
            if down_server is not None:
                down_server.shutdown()
                down_server.server_close()
            (proxy.ProxyHandler.upstream_host, proxy.ProxyHandler.upstream_port,
             proxy.ProxyHandler.upstream_base_path,
             proxy.ProxyHandler.upstream_use_tls) = saved


class TestClassifierThroughProxy(ProxyTestCase):
    def test_patched_body_reaches_upstream(self):
        body = json.dumps(_classifier_body(reasoning_effort="high")).encode()
        status, _, resp_body = self.post(body)
        self.assertEqual(status, 200)
        self.assertEqual(resp_body, _FakeUpstreamHandler.response_body)

        _, _, _, up_body = _FakeUpstreamHandler.requests[0]
        parsed = json.loads(up_body)
        self.assertEqual(parsed["thinking"], {"type": "disabled"})
        self.assertEqual(parsed["output_config"], {"effort": "low"})
        self.assertNotIn("reasoning_effort", parsed)
        # Unrelated fields must survive the patch untouched.
        self.assertEqual(parsed["max_tokens"], 2112)
        self.assertEqual(parsed["system"], _CLASSIFIER + " 350 lines of security policy...")


# ---------------------------------------------------------------------------
# Process-level tests — PID file, --stop, SIGTERM
# ---------------------------------------------------------------------------

class TestStopAndShutdown(unittest.TestCase):
    def _pid_file(self, name: str) -> str:
        return os.path.join(_TMPDIR, name)

    def test_pid_file_roundtrip(self):
        proxy.PID_FILE = self._pid_file("roundtrip.pid")
        proxy._write_pid_file(1234, 8799)
        self.assertEqual(proxy._read_pid_file(), (1234, 8799))
        proxy._remove_pid_file_if_own(1234, 8799)
        self.assertIsNone(proxy._read_pid_file())
        # Wrong pid/port must NOT remove the file.
        proxy._write_pid_file(1234, 8799)
        proxy._remove_pid_file_if_own(999, 8799)
        self.assertEqual(proxy._read_pid_file(), (1234, 8799))
        proxy._remove_pid_file_if_own(1234, 8799)

    def test_process_alive(self):
        self.assertTrue(proxy._process_alive(os.getpid()))
        self.assertFalse(proxy._process_alive(99999999))

    def test_stale_pid_file_never_kills_unrelated_process(self):
        marker = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"]
        )

        def _kill_marker():
            marker.kill()
            marker.wait()  # reap, so no ResourceWarning at shutdown

        self.addCleanup(_kill_marker)
        proxy.PID_FILE = self._pid_file("stale.pid")

        # Port that is guaranteed free — _stop_existing must not kill
        # anything that owns a different port.
        other_port = _free_port()
        proxy._write_pid_file(marker.pid, other_port)
        proxy._stop_existing(_free_port())  # request a *different* port
        self.assertTrue(proxy._process_alive(marker.pid), "marker must survive")

        # Stale entry (dead PID) on the requested port → removed, no crash.
        proxy._write_pid_file(99999999, other_port)
        proxy._stop_existing(other_port)
        self.assertIsNone(proxy._read_pid_file())

    def _spawn_proxy(self, port: int, pid_file: str):
        proc = subprocess.Popen(
            [sys.executable, _PROXY_PY, "--port", str(port),
             "--upstream", f"http://127.0.0.1:{_free_port()}"],
            env=dict(os.environ, PROXY_PID_FILE=pid_file),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self.addCleanup(proc.kill)
        return proc

    def _wait_health(self, port: int, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
                try:
                    conn.request("GET", "/health")
                    if conn.getresponse().status == 200:
                        return
                finally:
                    conn.close()
            except OSError:
                time.sleep(0.1)
        raise AssertionError(f"proxy on port {port} never became healthy")

    @unittest.skipIf(proxy._IS_WINDOWS, "SIGTERM semantics differ on Windows")
    def test_sigterm_clean_shutdown(self):
        port = _free_port()
        pid_file = self._pid_file("sigterm.pid")
        proc = self._spawn_proxy(port, pid_file)
        self._wait_health(port)
        self.assertTrue(os.path.exists(pid_file))

        os.kill(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
        self.assertEqual(proc.returncode, 0, "SIGTERM should exit cleanly")
        self.assertFalse(os.path.exists(pid_file), "PID file must be cleaned up")

    @unittest.skipIf(proxy._IS_WINDOWS, "taskkill semantics differ on Windows")
    def test_stop_by_port_owner(self):
        port = _free_port()
        pid_file = self._pid_file("stop.pid")
        proc = self._spawn_proxy(port, pid_file)
        self._wait_health(port)

        result = subprocess.run(
            [sys.executable, _PROXY_PY, "--port", str(port), "--stop"],
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        proc.wait(timeout=10)
        self.assertEqual(proc.returncode, 0)
        self.assertFalse(os.path.exists(pid_file), "PID file must be cleaned up")


if __name__ == "__main__":
    unittest.main()
