# 07: Validation, report, and demo

Parent: #1 (P0 Spec — LSTM Keyword Spotting)

**What to build:** The done proof: full acceptance run over corrupt audio, Silence, Unknown words, and reconnects; clean plus noisy labeled conditions; CPU, memory, and utterance-to-action latency measured on the target laptop; scripted mic test of each Keyword plus negatives with false-action counts; one rerun from saved checkpoint and config reproducing reported metrics; README with reproduction, limits, attribution, and demo steps; faculty demo of upload to Detections to live actions to dashboard.

**Blocked by:** 06 (analysis, live, and dashboard UI).

**Status:** implemented

- [x] All P0 stories pass or carry an explicit failed-acceptance report with evidence
- [x] Rerun from saved artifacts reproduces metrics within stated tolerance
- [x] Limits and unmet targets are documented, not redefined
- [ ] Demo runs end to end on CPU-only target hardware from a clean install

Implementation evidence, 15 October 2026:

- Per-story verdicts for all 24 P0 stories with evidence pointers are in
  [validation-report.md](../evaluations/validation-report.md). Stories 1-22
  pass on automated tests (97 backend, 6 frontend suites, `npm run check`
  green). Story 23 fails as unmeasured on the Ryzen 5 5500U target
  (dev-laptop numbers only). Story 24 carries the failed-acceptance
  report: isolated macro-F1 0.740 vs 0.85, event-F1 0.010 vs 0.70.
- `scripts/reproduce.py` reruns the E02 isolated and continuous test runs
  from the saved checkpoint and frozen configs: RERUN PASS, 18/18 fields
  match with difference 0. The continuous runs were regenerated with the
  current per-Keyword suppression rule from a clean tree; tuning
  re-selected identical parameters and the no-go gate stands.
- Limits, the unexecuted microphone protocol, the missing noisy
  comparison (P1-03), and attribution are documented in the report.
- [Demo script](../demo.md) covers upload to Detections to live actions
  to dashboard. Clean-install proof and the scripted microphone run on
  the target laptop remain to be executed and recorded.
