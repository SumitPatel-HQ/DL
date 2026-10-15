# 04: Window detector and feasibility gate

Parent: #1 (P0 Spec — LSTM Keyword Spotting)

**What to build:** The continuous-spotting experiment plus its verdict: overlapping-window scoring (~1.0s window, ~0.2–0.25s stride) reusing the isolated model, validation-tuned threshold with adjacent-window consistency, merging and duplicate suppression into Detections with approximate times, event-level evaluation with keyword-plus-tolerance matching and false alarms per hour on negatives, error review with examples, and a recorded go/no-go decision.

**Blocked by:** 02 (data, features, and annotations), 03 (train and isolated evaluation).

**Status:** implemented; gate decision no-go (event-F1 target not met)

- [x] Detector returns Detections with Keyword, start/end, Confidence, and threshold metadata
- [x] Thresholds tuned on validation only; test set frozen; tolerance frozen before test
- [x] Event precision/recall/F1 plus false alarms per hour reported on natural disjoint set
- [x] Gate recorded: continue, one bounded retune, or separate temporal-model prototype — no silent redefinition

`ml.spotting` scores 1 s windows with the isolated E02 model and merges consecutive
votes into Detections. Each Detection carries Confidence and threshold metadata.
`ml.evaluate_continuous` tunes on validation only and freezes the ±0.5 s tolerance.
It refuses detector files that could have seen test data. On held-out AMI IN1002,
event-F1 is 0.007 and false alarms are 1,559/hour, against the 0.70 target.
Ticket 07 later regenerated these runs with the current per-Keyword
suppression rule (event-F1 0.010, 2,216 false alarms/hour); the no-go gate
stands. [ADR-0006](../adr/0006-sliding-window-gate-no-go.md) records the gate: no-go for
the sliding-window detector as a working feature, with a temporal model as a
separately scoped prototype. See [the Evaluation Runs](../evaluations/continuous-v1/README.md).
