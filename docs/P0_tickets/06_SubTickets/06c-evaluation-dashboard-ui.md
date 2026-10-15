# 06c: Evaluation dashboard

Parent: #6 (06 Analysis, live, and dashboard UI)

**What to build:** The student and evaluator dashboard where every chart traces to a saved Evaluation Run, isolated classification stays separate from continuous-event detection, and missing evidence reads as Not evaluated rather than a guessed number.

**Blocked by:** 06a (Backend Evaluation Runs API).

**Status:** implemented

- [x] Training curves come from real saved logs with best-epoch marker; confusion matrix shows per-class precision, recall, and F1 including Unknown and Silence
- [x] Isolated metrics render separately from event precision, recall, F1, and false alarms per hour; experimental continuous tab carries its frozen detector reference and Confidence-only framing
- [x] Every chart traces to checkpoint hash, dataset/split version, config, and timestamp; switching models or deleting runs updates the view without mixing runs
- [x] Not evaluated, target-unmet, and run-deleted states are explicit with denominators shown; offline-backend failure is distinct and recoverable

Implementation: `/dashboard` route, `frontend/components/dashboard/`. Run selection,
kind separation, and target views are covered by `frontend/tests/evaluations.test.mjs`.
