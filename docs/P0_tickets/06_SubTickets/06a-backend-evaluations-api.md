# 06a: Backend Evaluation Runs API

Parent: #6 (06 Analysis, live, and dashboard UI)

**What to build:** The persisted evidence behind the dashboard: stored Evaluation Runs that any chart can trace to checkpoint, dataset split, and config, with explicit empty states when no valid run exists.

**Blocked by:** None (can start immediately; ticket 05 persistence seam already provides the tables).

**Status:** implemented

- [x] Evaluation Runs list and detail read from stored runs; every entry carries checkpoint hash, dataset/split version, config, and timestamp
- [x] Importing the committed isolated and continuous runs makes them readable; switching runs reloads all dependent views from that run alone
- [x] Deleting a run clears dependent views to an explicit empty state, never silent fallback or mixed-run numbers
- [x] Unlabeled-audio rule holds: Confidence only, no fabricated accuracy; isolated classification and continuous-event numbers stay separate

Implementation and API contract: [backend-api.md](../../backend-api.md#evaluation-runs).
Automated API coverage: `backend/tests/test_evaluations.py`.
