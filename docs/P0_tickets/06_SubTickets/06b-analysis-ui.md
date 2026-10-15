# 06b: Audio Analysis review workbench

Parent: #6 (06 Analysis, live, and dashboard UI)

**What to build:** The audio reviewer workbench on real backend data: upload a Recording, pick Keywords, run analysis, inspect Detections as linked waveform regions and rows with seek and play, then delete the Recording with its audio bytes and child Detections.

**Blocked by:** None (can start immediately; ticket 05 recording jobs and persistence already provide upload, analyze, status, Detections, playback, and delete).

**Status:** implemented

- [x] Uploaded WAV shows real Detections as waveform regions; clicking a region or row seeks and plays that segment with region, row, and clock linked both ways plus keyboard parity
- [x] Keyword selection offers only trained Keywords for the active model; analysis progress distinguishes uploading, queued, running, completed, failed, and zero-Detection empty state
- [x] Detection table shows Keyword, start, end, Confidence as score with threshold metadata, sortable and filterable by Keyword with Confidence never labeled as accuracy
- [x] Corrupt-file, oversize, queue-full, worker-unavailable, and offline-backend states are explicit and recoverable; delete removes audio bytes and child Detections with confirm and updates the view

Implementation: `/analyze` route, `frontend/components/analysis/`. Review logic
(sorting, lanes, clock linking, prechecks, job phases) is covered by
`frontend/tests/detections.test.mjs` and `frontend/tests/wav.test.mjs`.
