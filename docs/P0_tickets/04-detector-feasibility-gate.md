# 04: Window detector and feasibility gate

Parent: #1 (P0 Spec — LSTM Keyword Spotting)

**What to build:** The continuous-spotting experiment plus its verdict: overlapping-window scoring (~1.0s window, ~0.2–0.25s stride) reusing the isolated model, validation-tuned threshold with adjacent-window consistency, merging and duplicate suppression into Detections with approximate times, event-level evaluation with keyword-plus-tolerance matching and false alarms per hour on negatives, error review with examples, and a recorded go/no-go decision.

**Blocked by:** 02 (data, features, and annotations), 03 (train and isolated evaluation).

**Status:** ready-for-agent

- [ ] Detector returns Detections with Keyword, start/end, Confidence, and threshold metadata
- [ ] Thresholds tuned on validation only; test set frozen; tolerance frozen before test
- [ ] Event precision/recall/F1 plus false alarms per hour reported on natural disjoint set
- [ ] Gate recorded: continue, one bounded retune, or separate temporal-model prototype — no silent redefinition
