# 06: Analysis, live, and dashboard UI

Parent: #1 (P0 Spec — LSTM Keyword Spotting)

**What to build:** The three user-facing slices on real backend data: Audio Analysis with upload, Keyword chips, progress, waveform regions with click-to-seek/play, sortable Detection table, and delete; Voice Control with explicit Start/Stop, permission handling, live status, in-app slide and player actions for the six Keywords with cooldown and accepted/rejected history; Evaluation dashboard with run-based training curves, confusion matrix, isolated versus event metrics, and `Not evaluated` empty states. Confidence shown as score; no mock accuracy.

**Voice "stop" versus session Stop:** saying the Keyword "stop" triggers the pause
action only and never ends the session. Only the Stop button (which sends
`{"type":"stop"}`), tab-hide, page exit, close, or disconnect ends the session
immediately and releases the mic.

**Blocked by:** 05 (backend jobs and persistence).

**Status:** ready-for-agent

- [ ] Uploaded WAV shows real Detections as waveform regions; click seeks and plays the segment
- [ ] Mic demo maps the six trained Keywords to in-app actions with dedup, cooldown, and immediate stop
- [ ] Every chart traces to a saved Evaluation Run; switching models or deleting runs updates the view
- [ ] Corrupt-file, empty-result, mic-denied, and offline-backend states are explicit and recoverable
