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

DeepSeek's Claude-compatible API **ignores** `budget_tokens`. The thinking mode is binary: `"disabled"` or full. `output_config.effort` controls output generation but doesn't constrain the thinking phase. A local proxy is the minimum viable interception point.

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
| `PROXY_EFFORT`   | `low`, `medium`, `high` | `low`                               |
| `PROXY_PID_FILE` | any path                | `<tempdir>/deepseek-proxy.pid`      |
| `--port`         | any port                | `8799`                               |
| `--upstream`     | any URL                 | `https://api.deepseek.com/anthropic` |
| `--stop`         | —                       | Stop running instance and exit       |
| `-v`, `--verbose` | —                     | Log request details for debugging    |
| `--version`      | —                       | Print version and exit               |

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
