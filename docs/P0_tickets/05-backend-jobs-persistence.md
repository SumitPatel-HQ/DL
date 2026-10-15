# 05: Backend jobs and persistence

Parent: #1 (P0 Spec — LSTM Keyword Spotting)

**What to build:** The local service behind the app: health plus active model, recording upload with WAV validation, analysis jobs on one bounded worker with queued/running/completed/failed states, stored Detections, playback with range support, delete cascade of audio bytes and child rows, SQLite metadata for models, recordings, Detections, Evaluation Runs, and live events, with stable machine-readable errors and no arbitrary command execution.

**Blocked by:** 04 (window detector and feasibility gate).

**Status:** ready-for-agent

- [ ] Upload labeled test WAV, analyze, poll, retrieve authentic Detection intervals, delete everything via API test
- [ ] Bad WAV, too-long, unsupported Keyword, missing checkpoint, and zero-Detection cases return stable errors
- [ ] One bounded worker; disconnects and failures clean up jobs and buffers
- [ ] Injection of executable paths or commands via uploads or events is rejected
