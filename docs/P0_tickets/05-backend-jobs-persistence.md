# 05: Backend jobs and persistence

Parent: #1 (P0 Spec — LSTM Keyword Spotting)

**What to build:** The local service behind the app: health plus active model, recording upload with WAV validation, analysis jobs on one bounded worker with queued/running/completed/failed states, stored Detections, playback with range support, delete cascade of audio bytes and child rows, SQLite metadata for models, recordings, Detections, Evaluation Runs, and live events, with stable machine-readable errors and no arbitrary command execution.

**Blocked by:** 04 (window detector and feasibility gate).

**Status:** implemented

- [x] Upload labeled test WAV, analyze, poll, retrieve authentic Detection intervals, delete everything via API test
- [x] Bad WAV, too-long, unsupported Keyword, missing checkpoint, and zero-Detection cases return stable errors
- [x] One bounded worker; disconnects and failures clean up jobs and buffers
- [x] Injection of executable paths or commands via uploads or events is rejected

Implementation and API contract: [backend-api.md](../backend-api.md).
Automated API coverage: `backend/tests/test_recordings.py` and `test_health.py`.
Generated test checkpoints are plumbing fixtures, not model-quality evidence.

Local smoke validation also used the installed E02 checkpoint and the first held-out
Speech Commands WAV per Keyword from `data/isolated/manifest.json`. The API returned
authentic 0-1 s Detections for left, right, and down; up, go, and stop returned zero
Detections. Every job completed and every Recording was deleted through the API.
This was an API smoke check, not a new Evaluation Run or a quality claim.
The ADR-0006 experimental limitation remains explicit in model and Detection responses.

Deferred to ticket 06 (Analysis, live, and dashboard UI): Evaluation Runs
import/read APIs (`GET /api/evaluations`, `GET /api/evaluations/{id}`), the
command map (`GET /api/commands`), and live events WebSocket/actions. The
`evaluation_runs` and `live_events` tables are reserved schema only, with no
API behavior in this ticket.
