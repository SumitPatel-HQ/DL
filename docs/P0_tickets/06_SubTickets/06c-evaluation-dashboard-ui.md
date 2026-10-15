# 06c: Evaluation dashboard

Parent: #6 (06 Analysis, live, and dashboard UI)

**What to build:** The student and evaluator dashboard where every chart traces to a saved Evaluation Run, isolated classification stays separate from continuous-event detection, and missing evidence reads as Not evaluated rather than a guessed number.

**Blocked by:** 06a (Backend Evaluation Runs API).

**Status:** ready-for-agent

- [ ] Training curves come from real saved logs with best-epoch marker; confusion matrix shows per-class precision, recall, and F1 including Unknown and Silence
- [ ] Isolated metrics render separately from event precision, recall, F1, and false alarms per hour; experimental continuous tab carries its frozen detector reference and Confidence-only framing
- [ ] Every chart traces to checkpoint hash, dataset/split version, config, and timestamp; switching models or deleting runs updates the view without mixing runs
- [ ] Not evaluated, target-unmet, and run-deleted states are explicit with denominators shown; offline-backend failure is distinct and recoverable
