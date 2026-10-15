# Audio Analysis UI for Keyword Spotting Review

Repo constraints: local-first Next.js at `127.0.0.1:3000` + FastAPI at `127.0.0.1:8000`; vocabulary per `CONTEXT.md` is Detection (predicted Keyword occurrence with approximate start/end + score), Confidence (per-Detection model score, not correctness), Keyword (trained word); continuous detector is experimental per ADR-0006 and `docs/backend-api.md` (Confidence is classifier score, times are approximate window spans). Backend job states are `uploaded → queued → running → completed|failed`; `GET /api/recordings/{id}/detections` polled until terminal; completed zero-Detection returns `code: no_detections` + `detections: []`; `DELETE /api/recordings/{id}` deletes audio + metadata + child Detections with cascade.

## How best-in-class tools design waveform + region + transcript/detection table interaction

### Takeaway
Review-first tools keep one invariant: waveform region, list row, and audio clock stay bidirectionally linked — click/seek/play from any of the three — with zoom for precise boundary inspection and color-coded labels per Keyword/class.

### Cited Findings
- Label Studio `Audio` tag "plays audio and shows its waveform" for labeling regions of audio, seeing waveform, manipulating audio during annotation — [Source](https://labelstud.io/tags/audio.html)
- Label Studio ASR-segments template pairs `Labels` + `Audio` + per-region editable `TextArea` (`perRegion="true"`) so each audio segment carries its own transcript/classification — [Source](https://labelstud.io/templates/automatic_speech_recognition_segments)
- Label Studio workflow: spacebar toggles play, click/drag playhead or click timeline/waveform to navigate, zoom X (time) and Y (amplitude scale) to inspect peaks/valleys before marking label boundaries — [Source](https://labelstud.io/blog/labeling-audio-data-with-label-studio)
- Label Studio region model stores `value.start`/`value.end` in seconds + channel + labels, e.g. `start: 3.1, end: 8.2` — [Source](https://labelstud.io/tags/audio.html)
- Encord audio ontology equivalent: select Audio Region class, click-drag along waveform to apply label between start/end, snap boundary to playhead via "Fill to playhead", merge/copy-paste ranges — [Source](https://docs.encord.com/platform-documentation/Annotate/annotate-label-editor/annotate-audio)
- Descript wordbar "tells you exactly what words a waveform represents"; auto-aligns transcript to script audio; Cmd/Ctrl+drag word boundary above waveform to correct start/end after zooming until words ungroup — [Source](https://help.descript.com/hc/en-us/articles/10249346632717-The-wordbar)
- Descript waveform element supports Style (Lines/Circle/Ring/Wave), per-speaker assignment, fill/size/fidelity controls in Scene Editor properties pane — [Source](https://help.descript.com/hc/en-us/articles/10256425252749-Waveforms)
- Otter flow is Record-or-upload → instant transcript with speaker tags + timestamps → interactive editor (correct terms, assign speakers, highlight) → export TXT/DOCX/PDF/SRT or share link; supports AAC/MP3/M4A/WAV/WMA + video — [Source](https://otter.ai/audio-to-text)
- Otter generates "meeting summary with clickable time-stamped highlights so you can jump to key moments" and keyword search to navigate transcript — [Source](https://otter.ai/speech-to-text)

### Inferences
- For this repo: render each Detection as a non-editable region (Keyword label + Confidence in region content) + mirrored row in table; region-click and row-click both seek (`start_s - small pre-roll e.g. 0.2s`) and play; active region follows `timeupdate`/`region-in/out` to highlight row.
- Adopt Label Studio/Encord affordances: timeline ruler + zoom controls, playhead-drag + click-to-seek, per-Keyword colors with legend, `visibleWhen=region-selected`-style detail panel showing start/end/Confidence/threshold/windows.
- Adopt Descript/Otter affordance: clickable timestamp chips in table (`m:ss.mmm`) that seek; keyword filter chips that dim non-matching regions.

### Gaps
- No reliable primary source found on Audino current UX or ultrasonic/speech dashboard specifics; Descript/Otter docs fetched describe transcript-first editing, not confidence-scored Detection review.
- No citable source on optimal pre-roll duration or overlap-region stacking for ~1.0s windows with 0.2–0.25s stride; needs prototype testing.

## Proven patterns for async audio job UX: validation, queued/running/completed/failed, polling vs websocket, empty and corrupt-file states

### Takeaway
Upload-then-analyze jobs need immediate client validation, durable server states with distinct queued/running visuals, poll-until-terminal (correct for this repo's single-worker backend), explicit empty `no_detections` state, and actionable corrupt-file errors with retry.

### Cited Findings
- This backend accepts only non-empty integer PCM WAV 8–96 kHz mono/stereo 8/16/24/32-bit, validates headers + actual sample-byte count, enforces 40 MiB and 600 s; filenames are display metadata with `.wav` suffix, paths/command chars rejected — [Source](file://docs/backend-api.md) — local repo doc, see `docs/backend-api.md`
- This backend job lifecycle: new uploads `uploaded`; analysis `queued → running → completed|failed`; one worker + two waiting jobs; duplicate analysis 409, full queue 429; polling/playback disconnects do not cancel durable job; shutdown marks unfinished `service_interrupted` — [Source](file://docs/backend-api.md)
- Completed zero-Detection job returns HTTP 200 `code: no_detections`, `detections: []`, "never invents a label"; pre-completion code is `analysis_not_completed` — [Source](file://docs/backend-api.md)
- Error contract uses `{"error":{"code":"...","message":"..."}}` with `invalid_wav/unsupported_wav` 400, `recording_too_long/upload_too_large` 413, `queue_full/rate_limited/uploads_busy` 429, `analysis_in_progress/delete_failed` 409, `model_incompatible/worker_unavailable` 503 — [Source](file://docs/backend-api.md)
- Upload UI guidance stresses distinct components/states and error handling strategies to improve completion rates — [Source](https://blog.filestack.com/upload-file-ui-design-components-states-and-errors)
- Uppy StatusBar state machine distinguishes waiting/preprocessing/uploading/postprocessing/complete/error from per-file `uploadStarted/uploadComplete/preprocess/postprocess` flags — [Source](https://github.com/transloadit/uppy/blob/main/packages/%40uppy/status-bar/src/StatusBar.tsx)
- Exemplar async upload modal pattern: `pending → processing → success|failed` with progress %, queued message ("task queued"), polling every 2 s via `setInterval`, terminal success auto-close after 2 s, inline `error_message` + retry — [Source](https://raw.githubusercontent.com/zhouxiaoka/autoclip/main/frontend/src/components/UploadModal.tsx)
- Accessible upload primitive tracks aggregate `EMPTY → PROCESSED → UPLOADING → UPLOADED` announced via hidden `role="status"` live region, per-file `processed|uploading|uploaded|error|abort` + 0–100 progress + inline short success/error feedback in reading order — [Source](https://www.ariaui.dev/docs/components/upload)

### Inferences
- Polling (not websocket) matches ticket 06 scope and backend doc ("poll Recording metadata until terminal status"); poll `GET /api/recordings/{id}` + `.../detections` at ~1–2 s with backoff, stop on `completed|failed`, surface `Retry` on `failed/analysis_failed` and `Queue full (429) — try again shortly`.
- Status chip mapping: `uploaded` (gray, "Uploaded — ready to analyze"), `queued` (amber pulsing, position if known), `running` (blue animated bar, indeterminate + "does not cancel on disconnect"), `completed` (green + count or `no_detections` empty art), `failed` (red + `error_code` + recover action).
- Client pre-checks before POST: extension `.wav`, non-empty, size ≤40 MiB, duration ≤600 s via `AudioContext.decodeAudioData` header peek; show filename/duration/channels/rate + supported Keywords from `GET /api/models/active` before allowing Analyze.

### Gaps
- No citable benchmark for polling interval vs websocket for 10-min WAV single-worker jobs; backend doc mandates polling so websocket comparison is out of scope.
- No primary source on best copy for `service_interrupted` / `persistence_error` recovery; propose "Restart-safe: job marked failed, re-run analysis" and verify against backend tests.

## Waveform rendering approaches that work in React/Next.js: canvas, wavesurfer.js, peaks.js, Web Audio API

### Takeaway
For click-to-seek + zoom + region highlight in Next.js, wavesurfer.js v7 (canvas, Regions+Timeline+Zoom+Hover plugins) is the lowest-friction choice; peaks.js (BBC, zoomview+overview) fits overview+detail layouts but has heavier integration; raw Web Audio API canvas is fallback only.

### Cited Findings
- wavesurfer.js plugins are optional ES modules (`dist/plugins/<name>.esm.js`) registered via `WaveSurfer.create({ plugins: [regions, timeline] })`; official set includes Regions, Timeline, Spectrogram, Minimap, Hover, Zoom, Record, Envelope — [Source](https://wavesurfer.xyz/docs/plugins)
- Regions are "visual overlays on the waveform that mark segments of audio. They can be dragged, resized, clicked, and played back independently"; `regions.addRegion({start, end, color, content})` after `ready`; side-channel `Map` keyed on region `id` holds metadata in v7 (no built-in `data` field) — [Source](https://wavesurfer.xyz/docs/plugins/regions)
- Region interaction pattern: `regions.on('region-clicked', (region, e) => { e.stopPropagation(); region.play(true); })`; `region.play(stopAtEnd?)` seeks to start and plays to end; drag-to-create via `enableDragSelection()` — [Source](https://wavesurfer.xyz/docs/plugins/regions)
- Regions plugin emits `region-created/region-updated/region-removed/region-clicked/region-double-clicked/region-in/region-out/region-content-changed` for row↔waveform sync — [Source](https://github.com/katspaugh/wavesurfer.js/blob/main/_autodocs/api-reference/plugin-regions.md)
- Timeline plugin renders virtualized time ruler with `timeInterval/primaryLabelInterval/formatTimeCallback`, e.g. `m:ss` formatter — [Source](https://github.com/katspaugh/wavesurfer.js/blob/main/_autodocs/api-reference/plugin-timeline.md)
- Zoom plugin enables scroll/pinch zoom via `Zoom.create({ scale, maxZoom, deltaThreshold })` + `wavesurfer.on('zoom')` — [Source](https://github.com/katspaugh/wavesurfer.js/blob/main/_autodocs/api-reference/plugin-zoom.md)
- `wavesurfer-react` wrapper provides `<WaveSurfer plugins={...}>`, `<Region onClick/onIn/onOut/onUpdateEnd>` and `useWavesurfer/useWaveSurferContext` hooks; known issue: regions desynchronization — [Source](https://www.npmjs.com/package/wavesurfer-react)
- peaks.js config uses `zoomview` + `overview` containers + `mediaElement` + `webAudio: { audioContext, scale, multiChannel }`; `view.setZoom({seconds})`, `peaksInstance.player.seek(seconds)`, `keyboard: true` option — [Source](https://github.com/bbc/peaks.js/blob/master/demo/zoomable-waveform.html)
- peaks.js `view.setSegmentDragMode('overlap'|'no-overlap'|'compress')` and `view.enableSegmentDragging(bool)` control segment overlap on zoomview — [Source](https://github.com/bbc/peaks.js/blob/master/doc/API.md)
- Label Studio `Audio` tag supports `decoder: webaudio|ffmpeg|none`, `player: html5|webaudio`, `splitchannels`, `defaultzoom 1–1500`, `spectrogram` flag — [Source](https://labelstud.io/tags/audio.html)

### Inferences
- Recommended Next.js stack: client-only (`'use client'` + `dynamic(..., {ssr:false})`) wavesurfer v7 with `url: /api/recordings/{id}/audio` (supports byte ranges per backend doc), `interact: true`, `Regions (drag:false, resize:false for read-only Detections) + Timeline + Hover + Zoom`; store Detection→region-id map for highlight; use `region-in/out` to set active row.
- Keyboard-accessible alternative is mandatory because canvas/waveform is pointer-centric: provide Prev/Next Detection buttons, seek ±1 s/±5 s buttons, space play/pause, and a listbox of Detections that seeks identically; do not rely on drag-to-create (disable it).
- For 10-min ceiling: lazy-decode via `url` + peaks-style overview only if perf suffers; test decode time/RAM on Ryzen 5 5500U/16 GB target per KWS_SPEC before adding Minimap/Spectrogram.

### Gaps
- No head-to-head React/Next.js render-perf numbers for wavesurfer vs peaks on 10-min 16 kHz mono WAV found; needs local measurement.
- Web Audio API docs via Context7 not queried (budget); raw `AudioContext.decodeAudioData` + canvas remains un-sourced fallback.

## Table design for Detections (keyword, start, end, confidence, threshold): sorting, filtering, linking, delete cascade

### Takeaway
Detection table should be a sortable, filterable, dense data table where every row is a seek control linked to its waveform region, with Confidence shown as score + threshold context and delete scoped to Recording-level cascade (no per-Detection delete in P0 API).

### Cited Findings
- Each Detection contains `id, keyword, start_s, end_s, confidence, threshold, windows`; Detection list response also returns `status, code, detector` + experimental metadata — [Source](file://docs/backend-api.md)
- Backend has no per-Detection delete; `DELETE /api/recordings/{id}` deletes "audio, metadata, and child Detections" (204), staged-rename safe, `delete_failed` retryable with same request, in-progress delete conflicts 409 — [Source](file://docs/backend-api.md)
- Otter pattern for navigating long results: keyword search + clickable timestamped summary to jump to key moments — [Source](https://otter.ai/speech-to-text)
- Accessible async-status pattern: aggregate state in `role="status"` live region + short inline per-row feedback in reading order so screen readers encounter it naturally — [Source](https://www.ariaui.dev/docs/components/upload)
- W3C draft support material expects audio players to document functional support for media alternatives and expose player features (transcript/captions where applicable) — [Source](https://www.w3.org/WAI/WCAG3/informative/images-and-media/accessible-media-player/accessible-audio-player-selected)

### Inferences
- Columns: `Keyword (filter chips multi-select, default = analyzed set) | Start (button m:ss.mmm, sorts numeric) | End (same) | Duration (derived, helps judge ~1 s windows) | Confidence (bar 0–1 + numeric, sorts numeric; label "score, not correctness") | Threshold (numeric, tooltip "frozen detector config") | Play (row action)`. Default sort Start ascending; allow sort by Confidence desc to triage strong Detections first.
- Row↔waveform: hover row highlights region (outline), `aria-current` on active row during playback, click row seeks+plays with pre-roll; region click sets table selection + scrolls row into view (`scrollIntoView({block:'nearest'})`).
- Delete UX: no per-row delete (API lacks it); Recording-level "Delete recording" with confirm modal stating "Deletes audio bytes + N Detections irreversibly", disable during `queued/running` (409 `analysis_in_progress`), optimistic removal from list + toast with backend `code` on failure; empty table after delete returns to upload state.
- Honesty copy: header banner "Experimental detector — times approximate (± window stride), Confidence is not accuracy. See Evaluation: E02 feasibility run."; zero-Detection panel: "No Detections — no Keyword met threshold. Try other Keywords or check volume/Silence." with `code: no_detections` shown subtly for debuggability.

### Gaps
- No citable NN/g or W3C table-sorting pattern retrieved (search returned generic accessibility report listings, no media-table guideline); sorting/filtering guidance above is inferred from Otter keyword search + common data-table practice, needs a dedicated accessibility pass (keyboard sort headers, `aria-sort`, focus-visible regions).
- No source on best Confidence visualization (bar vs dot vs numeric) for classifier scores that must not imply probability; propose bar + "score" label and validate with course evaluator.
