# P1-04: Temporal prototype for continuous localization (ex-PR-19, promoted)

Parent: P1+P2 spec ([P1&P2 Spec](P1&P2%20Spec.md)), #1 (P0 Spec — LSTM Keyword Spotting)

**What to build:** The ADR-0006 remediation: one bounded temporal-model
prototype (LSTM+CTC or equivalent sequence Detector) behind the existing
Detection interface (Keyword, start/end, Confidence, threshold metadata),
trained/scored through the same frozen protocol — validation-only tuning,
±0.5 s center tolerance, per-Keyword duplicate suppression, held-out AMI
IN1002 test split — and reported as a separate experimental Evaluation Run
that never replaces the isolated-classifier core until it passes the gate.

**Blocked by:** P0-07 (acceptance report). Uses frozen detector file + test split from `docs/evaluations/continuous-v1/`.

**Status:** done. Verdict: keep experimental ([ADR-0007](../adr/0007-temporal-prototype-keep-experimental.md), [runs](../evaluations/temporal-v1/README.md)). Test event-F1 0.057 vs the 0.010 baseline; the 0.70 target is unmet.

- [x] Prototype emits Detections in the existing shape through the existing Detector interface seam (no new serving contract)
- [x] Tuning uses validation only; test split, tolerance, and match rule are unchanged and frozen before test
- [x] Run reports event precision/recall/F1 + false alarms/hour with error review, directly comparable to the 0.010 baseline
- [x] Gate verdict is recorded (adopt / bounded retune / keep experimental); the 0.70 target is reported as met or unmet, never redefined
- [x] UI (if any) labels the prototype experimental with Confidence-only language and links the run
