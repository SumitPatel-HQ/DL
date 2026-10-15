# 07: Validation, report, and demo

Parent: #1 (P0 Spec — LSTM Keyword Spotting)

**What to build:** The done proof: full acceptance run over corrupt audio, Silence, Unknown words, and reconnects; clean plus noisy labeled conditions; CPU, memory, and utterance-to-action latency measured on the target laptop; scripted mic test of each Keyword plus negatives with false-action counts; one rerun from saved checkpoint and config reproducing reported metrics; README with reproduction, limits, attribution, and demo steps; faculty demo of upload to Detections to live actions to dashboard.

**Blocked by:** 06 (analysis, live, and dashboard UI).

**Status:** ready-for-agent

- [ ] All P0 stories pass or carry an explicit failed-acceptance report with evidence
- [ ] Rerun from saved artifacts reproduces metrics within stated tolerance
- [ ] Limits and unmet targets are documented, not redefined
- [ ] Demo runs end to end on CPU-only target hardware from a clean install
