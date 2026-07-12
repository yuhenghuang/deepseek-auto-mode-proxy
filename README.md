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

## How detection works

Classifier requests are identified by 4 criteria (all must match):

1. `stream` is not `true` — classifier is non-streaming
2. `tools` is absent/empty — no tool definitions
3. `messages` has exactly 1 entry — single transcript message
4. System prompt starts with `"You are a security monitor for autonomous AI coding agents."` — unique classifier fingerprint

Criteria 1–3 are the battle-tested heuristic from [deepseek-claude-proxy](https://github.com/dashxio/deepseek-claude-proxy). Criterion 4 adds content-level certainty. False positives are impossible in practice.

| Variable | Values | Default |
|---|---|---|
| `PROXY_THINKING` | `disabled`, `enabled` | `enabled` |
| `PROXY_EFFORT` | `low`, `medium`, `high` | `medium` |
| `--port` | any port | `8799` |
| `--upstream` | any URL | `https://api.deepseek.com/anthropic` |

## Health check

```bash
curl http://127.0.0.1:8799/health
# {"ok": true}
```

## Requirements

Python 3.7+. Zero dependencies (stdlib only).

## Related

- [deepseek-claude-proxy](https://github.com/dashxio/deepseek-claude-proxy) — Node.js version

## License

MIT
