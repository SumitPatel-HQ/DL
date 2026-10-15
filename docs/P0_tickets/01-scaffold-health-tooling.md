# 01: Scaffold, health, and tooling

Parent: #1 (P0 Spec — LSTM Keyword Spotting)

**What to build:** A runnable empty system: repo layout with backend app plus ML, frontend shell, artifacts, and scripts; CPU-only dependency set; FastAPI health reporting model-unavailable until a checkpoint is installed; test and lint commands; loopback-only binding; baseline system info capture.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [x] Fresh clone launches API, test suite, and minimal web page
- [x] `/api/health` reports model-unavailable with explicit setup action
- [x] Ruff/format, pytest, and TypeScript lint pass via one documented command
- [x] Backend binds loopback only; data dirs gitignored; sysinfo recorded

Implementation evidence, 15 October 2026:

- `npm run setup` passed in a separate local clone. Its API and web page launched
  on `127.0.0.1`; the browser displayed model-unavailable and its setup action.
- `npm run check` passed Ruff/format, strict mypy, ESLint, TypeScript and all 3
  pytest tests, including the CPU-only runtime gate and non-loopback host rejection.
- The browser displayed a distinct backend-disconnected state when the API stopped.
- `npm run build --prefix frontend` passed. Baseline environment information is in
  `artifacts/system/baseline.json`; setup and the known development dependency
  advisory are documented in `README.md`.
- Standards and spec reviews found no remaining actionable findings.
