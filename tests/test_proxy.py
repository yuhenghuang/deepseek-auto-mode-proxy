#!/usr/bin/env python3
"""Offline test suite for the DeepSeek auto-mode proxy.

Stdlib ``unittest`` only — no pytest, no requests. The suite never touches
the live PID file (``proxy.PID_FILE`` is repointed to a temp path) and never
hits the real DeepSeek API (a local fake upstream stands in).

Run from the repo root with::

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import gzip
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

    def log_message(self, format: str, *args) -> None:
        del format, args  # silence the default access log


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
        self.assertEqual(out["output_config"], {"effort": "high"})

    def test_custom_thinking_and_effort(self):
        out = proxy._patch_classifier(
            {"messages": []}, thinking="enabled", effort="high"
        )
        self.assertEqual(out["thinking"], {"type": "enabled"})
        self.assertEqual(out["output_config"], {"effort": "high"})

    def test_max_effort(self):
        # "max" is the third distinct tier on deepseek-v4-pro (Aug 2026 docs).
        out = proxy._patch_classifier({"messages": []}, effort="max")
        self.assertEqual(out["output_config"], {"effort": "max"})

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
        # Server-scoped config (v1.3.0): this server's dead upstream must not
        # affect the shared ProxyTestCase server, even while both run at once.
        down_server = proxy.build_server(
            0, f"http://127.0.0.1:{_free_port()}", verbose=False
        )
        self.addCleanup(down_server.server_close)
        threading.Thread(target=down_server.serve_forever, daemon=True).start()
        self.addCleanup(down_server.shutdown)
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


class TestClassifierThroughProxy(ProxyTestCase):
    def test_patched_body_reaches_upstream(self):
        body = json.dumps(_classifier_body(reasoning_effort="high")).encode()
        status, _, resp_body = self.post(body)
        self.assertEqual(status, 200)
        self.assertEqual(resp_body, _FakeUpstreamHandler.response_body)

        _, _, _, up_body = _FakeUpstreamHandler.requests[0]
        parsed = json.loads(up_body)
        self.assertEqual(parsed["thinking"], {"type": "disabled"})
        self.assertEqual(parsed["output_config"], {"effort": "high"})
        self.assertNotIn("reasoning_effort", parsed)
        # Unrelated fields must survive the patch untouched.
        self.assertEqual(parsed["max_tokens"], 2112)
        self.assertEqual(parsed["system"], _CLASSIFIER + " 350 lines of security policy...")


# ---------------------------------------------------------------------------
# Usage reporting (v1.4.0) — unit level
# ---------------------------------------------------------------------------

class TestTelemetryUrlResolution(unittest.TestCase):
    """PROXY_TELEMETRY_PATH is resolved against OTEL_EXPORTER_OTLP_ENDPOINT."""

    def test_path_joins_endpoint(self):
        self.assertEqual(
            proxy.resolve_telemetry_url("/classifier", "http://localhost:4318"),
            "http://localhost:4318/classifier",
        )

    def test_slashes_are_normalised(self):
        for path, endpoint in (("classifier", "http://h:1/"),
                               ("/classifier", "http://h:1"),
                               ("//classifier", "http://h:1//")):
            self.assertEqual(
                proxy.resolve_telemetry_url(path, endpoint),
                "http://h:1/classifier", f"{endpoint!r} + {path!r}",
            )

    def test_unset_path_disables_reporting(self):
        self.assertEqual(proxy.resolve_telemetry_url("", "http://localhost:4318"), "")

    def test_path_without_endpoint_disables_reporting(self):
        # Honours the path but cannot place it: must degrade, not crash.
        self.assertEqual(proxy.resolve_telemetry_url("/classifier", ""), "")

    def test_whitespace_only_values_are_treated_as_unset(self):
        self.assertEqual(proxy.resolve_telemetry_url("  ", "http://h:1"), "")
        self.assertEqual(proxy.resolve_telemetry_url("/classifier", "  "), "")


class TestCaptureHelpers(unittest.TestCase):
    def test_gzip_requested_when_client_accepts_it(self):
        for value in ("gzip", "gzip, deflate, br, zstd", "br;q=0.8, gzip;q=1.0",
                      "GZIP"):
            self.assertEqual(proxy._capture_accept_encoding(value), "gzip", value)

    def test_identity_when_client_cannot_take_gzip(self):
        # br/zstd alone are undecodable by the stdlib, so ask for plaintext.
        for value in ("", "br", "zstd, br", "identity"):
            self.assertEqual(proxy._capture_accept_encoding(value), "identity", value)

    def test_decode_identity_and_gzip(self):
        raw = b'{"usage":{"output_tokens":7}}'
        self.assertEqual(proxy._decode_response_body(raw, ""), raw)
        self.assertEqual(proxy._decode_response_body(raw, "identity"), raw)
        self.assertEqual(
            proxy._decode_response_body(gzip.compress(raw), "gzip"), raw
        )

    def test_decode_rejects_undecodable_encoding(self):
        with self.assertRaises(ValueError):
            proxy._decode_response_body(b"x", "br")

    def test_timestamp_is_iso_with_z(self):
        ts = proxy._utc_now_iso()
        self.assertTrue(ts.endswith("Z"))
        self.assertRegex(ts, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")


class _SinkHandler(http.server.BaseHTTPRequestHandler):
    """Stand-in telemetry sink — records what the proxy POSTs to it."""

    records: list = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length > 0 else b""
        _SinkHandler.records.append((self.path, dict(self.headers), body))
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, format: str, *args) -> None:
        del format, args


class TestUsageReportingIntegration(unittest.TestCase):
    """End-to-end: classifier responses reach the sink, others don't."""

    @classmethod
    def setUpClass(cls):
        cls.upstream = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), _FakeUpstreamHandler
        )
        cls.upstream_port = cls.upstream.server_address[1]
        cls.sink = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _SinkHandler)
        cls.sink_port = cls.sink.server_address[1]
        cls.proxy_server = proxy.build_server(
            0, f"http://127.0.0.1:{cls.upstream_port}", verbose=False,
            telemetry_url=f"http://127.0.0.1:{cls.sink_port}/classifier",
        )
        cls.proxy_port = cls.proxy_server.server_address[1]
        cls._threads = [
            threading.Thread(target=srv.serve_forever, daemon=True)
            for srv in (cls.upstream, cls.sink, cls.proxy_server)
        ]
        for t in cls._threads:
            t.start()

    @classmethod
    def tearDownClass(cls):
        for srv in (cls.upstream, cls.sink, cls.proxy_server):
            srv.shutdown()
            srv.server_close()

    def setUp(self):
        _FakeUpstreamHandler.requests.clear()
        _SinkHandler.records.clear()
        _FakeUpstreamHandler.response_headers = [("Content-Type", "application/json")]
        _FakeUpstreamHandler.response_body = b'{"id":"upstream"}'

    def _post(self, body: bytes, accept_encoding: str | None = None):
        headers = {"Content-Type": "application/json"}
        if accept_encoding is not None:
            headers["Accept-Encoding"] = accept_encoding
        headers[proxy._SESSION_HEADER] = "sess-1234"
        conn = http.client.HTTPConnection("127.0.0.1", self.proxy_port, timeout=10)
        try:
            conn.request("POST", "/v1/messages", body=body, headers=headers)
            resp = conn.getresponse()
            return resp.status, dict(resp.headers), resp.read()
        finally:
            conn.close()

    def _await_sink(self, count: int = 1, timeout: float = 5.0) -> list:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if len(_SinkHandler.records) >= count:
                return list(_SinkHandler.records)
            time.sleep(0.05)
        return list(_SinkHandler.records)

    def test_classifier_usage_is_reported(self):
        payload = {
            "id": "msg_1", "model": "deepseek-flash",
            "usage": {
                "input_tokens": 34815, "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0, "output_tokens": 138,
                "server_tool_use": {"web_search_requests": 0},
            },
        }
        _FakeUpstreamHandler.response_body = json.dumps(payload).encode()

        status, _, resp_body = self._post(
            json.dumps(_classifier_body()).encode(), accept_encoding="gzip, br, zstd"
        )
        self.assertEqual(status, 200)
        self.assertEqual(resp_body, _FakeUpstreamHandler.response_body)

        records = self._await_sink()
        self.assertEqual(len(records), 1, "classifier usage must be reported once")
        path, _, body = records[0]
        self.assertEqual(path, "/classifier")
        record = json.loads(body)
        self.assertEqual(record["session_id"], "sess-1234")
        self.assertEqual(record["model"], "deepseek-flash")
        self.assertEqual(record["usage"], {
            "input_tokens": 34815,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "output_tokens": 138,
        })
        # Nested/unknown keys must not leak into the report.
        self.assertNotIn("server_tool_use", record["usage"])

    def test_gzipped_response_is_decoded_for_the_sink_only(self):
        payload = {"model": "deepseek-flash",
                   "usage": {"input_tokens": 5, "output_tokens": 6}}
        raw = json.dumps(payload).encode()
        compressed = gzip.compress(raw)
        _FakeUpstreamHandler.response_headers = [
            ("Content-Type", "application/json"),
            ("Content-Encoding", "gzip"),
        ]
        try:
            _FakeUpstreamHandler.response_body = compressed
            status, headers, resp_body = self._post(
                json.dumps(_classifier_body()).encode(), accept_encoding="gzip, br"
            )
            # The client's bytes are the upstream's, untouched...
            self.assertEqual(status, 200)
            self.assertEqual(resp_body, compressed)
            self.assertEqual(headers.get("Content-Encoding"), "gzip")
        finally:
            _FakeUpstreamHandler.response_headers = [
                ("Content-Type", "application/json")
            ]
            _FakeUpstreamHandler.response_body = b'{"id":"upstream"}'

        records = self._await_sink()
        self.assertEqual(len(records), 1)
        self.assertEqual(json.loads(records[0][2])["usage"],
                         {"input_tokens": 5, "output_tokens": 6})

    def test_accept_encoding_is_narrowed_only_when_capturing(self):
        self._post(json.dumps(_classifier_body()).encode(),
                   accept_encoding="gzip, deflate, br, zstd")
        _, _, headers, _ = _FakeUpstreamHandler.requests[0]
        self.assertEqual(headers.get("Accept-Encoding"), "gzip")

    def test_conversation_request_is_not_captured(self):
        body = json.dumps({
            "model": "deepseek-flash[1m]",
            "stream": True,
            "tools": [{"name": "Bash"}],
            "messages": [{"role": "user", "content": "hi"}],
        }).encode()
        self._post(body, accept_encoding="gzip, deflate, br, zstd")

        # Client's own header stands — the proxy must not rewrite encodings
        # for traffic it isn't measuring.
        _, _, headers, _ = _FakeUpstreamHandler.requests[0]
        self.assertEqual(headers.get("Accept-Encoding"),
                         "gzip, deflate, br, zstd")

        time.sleep(0.5)  # give a stray reporter thread time to appear
        self.assertEqual(_SinkHandler.records, [],
                         "conversation calls are telemetry's job, not the proxy's")

    def test_structurally_matching_drift_case_is_still_captured(self):
        # Signature check failed (prompt drift) but structurally classifier-
        # shaped — the population that would otherwise vanish silently.
        _FakeUpstreamHandler.response_body = json.dumps({
            "model": "deepseek-flash",
            "usage": {"input_tokens": 11, "output_tokens": 22},
        }).encode()
        body = json.dumps({
            "model": "deepseek-flash[1m]", "stream": False, "messages": [],
            "system": "A NEW security monitor prompt we have never seen",
        }).encode()
        self._post(body, accept_encoding="gzip")

        records = self._await_sink()
        self.assertEqual(len(records), 1)
        self.assertEqual(json.loads(records[0][2])["usage"],
                         {"input_tokens": 11, "output_tokens": 22})

    def test_response_without_usage_reports_nothing(self):
        _FakeUpstreamHandler.response_body = b'{"id":"msg","model":"deepseek-flash"}'
        self._post(json.dumps(_classifier_body()).encode(), accept_encoding="gzip")
        time.sleep(0.5)
        self.assertEqual(_SinkHandler.records, [])


class TestUsageReportingDisabled(unittest.TestCase):
    """With usage reporting unconfigured the proxy must behave exactly as before."""

    @classmethod
    def setUpClass(cls):
        cls.upstream = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), _FakeUpstreamHandler
        )
        cls.proxy_server = proxy.build_server(
            0, f"http://127.0.0.1:{cls.upstream.server_address[1]}"
        )
        cls.proxy_port = cls.proxy_server.server_address[1]
        for srv in (cls.upstream, cls.proxy_server):
            threading.Thread(target=srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        for srv in (cls.upstream, cls.proxy_server):
            srv.shutdown()
            srv.server_close()

    def test_default_config_has_no_telemetry_url(self):
        self.assertEqual(self.proxy_server.config.telemetry_url, "")

    def test_classifier_accept_encoding_untouched(self):
        _FakeUpstreamHandler.requests.clear()
        conn = http.client.HTTPConnection("127.0.0.1", self.proxy_port, timeout=10)
        try:
            conn.request("POST", "/v1/messages",
                         body=json.dumps(_classifier_body()).encode(),
                         headers={"Content-Type": "application/json",
                                  "Accept-Encoding": "gzip, br, zstd"})
            self.assertEqual(conn.getresponse().status, 200)
        finally:
            conn.close()
        _, _, headers, _ = _FakeUpstreamHandler.requests[0]
        self.assertEqual(headers.get("Accept-Encoding"), "gzip, br, zstd")


class TestEnvValidation(unittest.TestCase):
    """PROXY_EFFORT validation: only DeepSeek's real values are accepted."""

    def setUp(self):
        self._saved = os.environ.get("PROXY_EFFORT")

        def _restore():
            if self._saved is None:
                os.environ.pop("PROXY_EFFORT", None)
            else:
                os.environ["PROXY_EFFORT"] = self._saved

        self.addCleanup(_restore)

    def test_effort_max_accepted(self):
        os.environ["PROXY_EFFORT"] = "max"
        self.assertEqual(
            proxy._env_value("PROXY_EFFORT", "high", proxy._VALID_EFFORT), "max"
        )

    def test_effort_low_accepted(self):
        os.environ["PROXY_EFFORT"] = "low"
        self.assertEqual(
            proxy._env_value("PROXY_EFFORT", "high", proxy._VALID_EFFORT), "low"
        )

    def test_effort_medium_rejected(self):
        # "medium" maps to "high" at the model level but is not accepted by
        # output_config.effort — must fall back to the default.
        os.environ["PROXY_EFFORT"] = "medium"
        self.assertEqual(
            proxy._env_value("PROXY_EFFORT", "high", proxy._VALID_EFFORT), "high"
        )


# ---------------------------------------------------------------------------
# build_server — server-scoped config (v1.3.0)
# ---------------------------------------------------------------------------

class TestBuildServer(unittest.TestCase):
    def test_invalid_upstream_raises(self):
        with self.assertRaises(ValueError):
            proxy.build_server(0, "not a url")

    def test_parses_upstream_parts(self):
        server = proxy.build_server(
            0, "http://example.com:8080/base", verbose=True,
            thinking="enabled", effort="max",
        )
        self.addCleanup(server.server_close)
        cfg = server.config
        self.assertEqual(cfg.upstream_host, "example.com")
        self.assertEqual(cfg.upstream_port, 8080)
        self.assertEqual(cfg.upstream_base_path, "/base")
        self.assertFalse(cfg.upstream_use_tls)
        self.assertTrue(cfg.verbose)
        self.assertEqual(cfg.thinking, "enabled")
        self.assertEqual(cfg.effort, "max")

    def test_https_defaults(self):
        server = proxy.build_server(0)  # default upstream is https
        self.addCleanup(server.server_close)
        cfg = server.config
        self.assertTrue(cfg.upstream_use_tls)
        self.assertEqual(cfg.upstream_port, 443)
        self.assertEqual(cfg.upstream_base_path, "/anthropic")
        self.assertFalse(cfg.verbose)
        self.assertEqual((cfg.thinking, cfg.effort), ("disabled", "high"))

    def test_servers_do_not_share_config(self):
        # Regression guard: before v1.3.0, build_server wrote upstream config
        # into ProxyHandler class attributes — two servers could not coexist.
        a = proxy.build_server(0, "http://a.example:1")
        b = proxy.build_server(0, "http://b.example:2")
        self.addCleanup(a.server_close)
        self.addCleanup(b.server_close)
        self.assertNotEqual(a.config.upstream_host, b.config.upstream_host)


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
