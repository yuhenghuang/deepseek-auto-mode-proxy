# DeepSeek Auto-Mode Proxy

Eliminates Claude Code auto-mode classifier timeouts when using DeepSeek V4 Pro.

## Problem

Claude Code's auto mode sends a safety classification request before each tool execution. DeepSeek V4 Pro's default thinking mode makes these take 28–32s, hitting the ~30s internal timeout.

## Solution

A local HTTP proxy that intercepts classifier requests and controls DeepSeek's thinking behavior — dropping response time from 30s to 2s.

```
Claude Code → HTTP → proxy (127.0.0.1:8799) → HTTPS → api.deepseek.com/anthropic
```

Only classifier requests are patched. All other traffic passes through unchanged.

### Why not just lower the thinking budget?

DeepSeek's Claude-compatible API **ignores** `budget_tokens`, and thinking effort has only **three distinct tiers on V4 Pro** (`low`, `high` — the default, and `max`; `medium` and `xhigh` are silently aliased to `high`). The proxy still disables thinking by default because that is the only mode measured at 1–3s on classifier-sized inputs; the new `low` tier is available via `PROXY_EFFORT=low` for anyone who wants to benchmark it. A local proxy is the minimum viable interception point.

## Quick start

```bash
python3 proxy.py &
export ANTHROPIC_BASE_URL=http://127.0.0.1:8799
claude
```

To re-enable thinking on classifier requests:

```bash
PROXY_THINKING=enabled python3 proxy.py &
```

Stop the proxy:

```bash
python3 proxy.py --stop
```

## Configuration

| Variable / flag  | Values                  | Default                              |
| ---------------- | ----------------------- | ------------------------------------ |
| `PROXY_THINKING` | `disabled`, `enabled`   | `disabled`                          |
| `PROXY_EFFORT`   | `low`, `high`, `max`    | `high`                              |
| `PROXY_PID_FILE` | any path                | `<tempdir>/deepseek-proxy.pid`      |
| `PROXY_TELEMETRY_PATH` | any path         | unset — usage reporting off          |
| `--port`         | any port                | `8799`                               |
| `--upstream`     | any URL                 | `https://api.deepseek.com/anthropic` |
| `--stop`         | —                       | Stop running instance and exit       |
| `-v`, `--verbose` | —                     | Log request details for debugging    |
| `--version`      | —                       | Print version and exit               |

`PROXY_EFFORT` accepts the values DeepSeek's Anthropic-compatible API accepts for `output_config.effort` (`low`, `high`, `max`). Since the August 2026 docs update, all three are distinct tiers on DeepSeek V4 Pro: `low` → low, `high` → high (the default), `max` → max. The docs' requested-level table also lists `medium` and `xhigh`, but both map to `high` and are rejected by `output_config.effort`. Source: [DeepSeek Thinking Mode docs](https://api-docs.deepseek.com/guides/thinking_mode).

## How detection works

Classifier requests are identified by 4 criteria (all must match):

1. `stream` is not `true` — classifier is non-streaming
2. `tools` is absent/empty — no tool definitions
3. `messages` has ≤2 entries — v2.1.160 (Jun 2025) added an assistant pre-fill to force `<block>` output, increasing the count from 1 to 2; real conversations have dozens to hundreds
4. System prompt starts with `"You are a security monitor for autonomous AI coding agents."` — verified on Claude Code v2.1.205. **This signature may change in future versions.**

Criteria 1–3 are the battle-tested heuristic from [deepseek-claude-proxy](https://github.com/dashxio/deepseek-claude-proxy). Criterion 4 adds content-level certainty. False positives are impossible in practice.

> **Warning:** Criterion 4 is version-dependent. If Anthropic changes the classifier system prompt in a future Claude Code release, detection will silently stop working — the proxy will pass classifier requests through unpatched, and timeouts will resume. After upgrading Claude Code, check the proxy log for `[classifier]` entries. If missing, update `_CLASSIFIER_SIGNATURE` in proxy.py to match the new prompt opening.

## How patching works

Claude Code v2.1.205 sends classifier requests with `max_tokens=2112` and no `thinking`, `reasoning_effort`, or `output_config` — verified via verbose proxy logging. With no `thinking` param, DeepSeek seems to default to full thinking on the classifier's large input (~300K chars total), causing 28–32s response times. Claude Code appears to enforce a ~30s internal deadline on classifier responses (hypothesis — not documented; `API_TIMEOUT_MS` defaults to 10 min and [auto-mode config](https://code.claude.com/docs/en/auto-mode-config) lists no timeout setting). With `thinking: disabled`, responses complete in 1–3s, well within any plausible deadline. The proxy injects `thinking` and `output_config.effort` to regain control.

## Usage reporting (optional)

Classifier requests are invisible to Claude Code's own OpenTelemetry export — the auto-mode classifier is instrumented nowhere — so token accounting for them has to happen at the seam where the proxy already sees the response. Set `PROXY_TELEMETRY_PATH` to report it:

```bash
PROXY_TELEMETRY_PATH=/classifier python3 proxy.py &
```

The path is resolved against `OTEL_EXPORTER_OTLP_ENDPOINT` — the endpoint Claude Code already exports its own telemetry to — so the host and port are configured once for both rather than repeated here. With the endpoint `http://localhost:4318`, that example posts to `http://localhost:4318/classifier`. If the path is set but the endpoint is not, reporting is disabled with a warning rather than guessed at.

With it set, every structurally-classifier-shaped request — the same `_structural_match` test used for detection, so requests whose signature check failed (prompt drift) are measured too, rather than silently vanishing — has its response teed and POSTed as:

```json
{"ts": "<ISO 8601 UTC>", "model": "deepseek-flash",
 "session_id": "6db675ec-…", "usage": {"input_tokens": 34815, "output_tokens": 138}}
```

Properties worth knowing:

- **Off by default.** With the variable unset the proxy is byte-for-byte the same as before, including not touching `Accept-Encoding`.
- **Never in the request path.** The response is teed while being streamed to the client, and reporting happens on a daemon thread *after* the client has been served. A slow or dead sink cannot add latency or fail a request; failures are logged as `[usage] report failed:` and swallowed.
- **`Accept-Encoding` is narrowed to `gzip`** for captured requests only, because Python's stdlib cannot decode the `br`/`zstd` the client advertises. The bytes forwarded to the client are unchanged — the proxy decodes only its own copy. If the client does not accept gzip, `identity` is requested instead.
- **Nothing but usage leaves the process.** The request body is never parsed for content, logged, or transmitted; only the response's `usage` block is read.

## Health check

```bash
curl http://127.0.0.1:8799/health
# {"ok": true}
```

## Testing

The repo ships an offline test suite (stdlib `unittest` only — no dependencies, no network):

```bash
python3 -m unittest discover -s tests -v
```

It covers classifier detection (all 4 criteria), patching, byte-identical passthrough, SSE streaming, health/error responses, and `--stop`/SIGTERM process handling. Tests use ephemeral ports and their own PID file, so they never touch a running proxy.

## Requirements

Python 3.7+. Zero dependencies (stdlib only). Works on Linux, macOS, and Windows.

On Windows, use `python proxy.py` instead of `python3`.

## Related

- [deepseek-claude-proxy](https://github.com/dashxio/deepseek-claude-proxy) — Node.js version

## License

MIT
