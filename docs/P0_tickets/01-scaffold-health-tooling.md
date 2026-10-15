# 01: Scaffold, health, and tooling

Parent: #1 (P0 Spec — LSTM Keyword Spotting)

**What to build:** A runnable empty system: repo layout with backend app plus ML, frontend shell, artifacts, and scripts; CPU-only dependency set; FastAPI health reporting model-unavailable until a checkpoint is installed; test and lint commands; loopback-only binding; baseline system info capture.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] Fresh clone launches API, test suite, and minimal web page
- [ ] `/api/health` reports model-unavailable with explicit setup action
- [ ] Ruff/format, pytest, and TypeScript lint pass via one documented command
- [ ] Backend binds loopback only; data dirs gitignored; sysinfo recorded
