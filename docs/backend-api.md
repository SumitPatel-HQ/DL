# Local recording service

Start one API process with `npm run api`. The launcher binds to `127.0.0.1:8000`.
Run one service process per storage directory. The default directory is ignored
`data/service/`, containing `metadata.sqlite3` and `audio/<generated-id>.wav`.
Requests cannot select filesystem paths, models, Python functions, or executables.

The default active model is `artifacts/models/e02/model.pt`, paired with
`docs/evaluations/continuous-v1/detector-e02.json`. Startup checks the checkpoint
format, labels, features, and detector/checkpoint hash agreement. Missing or
incompatible artifacts leave the API running with explicit model-unavailable status.
Operator environment variables `KWS_STORAGE`, `KWS_CHECKPOINT`, `KWS_DETECTOR`, and
`KWS_EVALUATIONS` (default `docs/evaluations/`) override these paths before startup.
Restart to change the active model.

Continuous analysis is experimental per [ADR-0006](adr/0006-sliding-window-gate-no-go.md).
Active-model and Detection responses include the limitation and a pointer to the
committed E02 feasibility Evaluation Run. That evidence describes the E02 gate,
not a new Evaluation Run for an operator-installed alternative checkpoint.
Confidence is a classifier score. Times are approximate window spans.

## Recording workflow

| Method | Route | Response |
| --- | --- | --- |
| GET | `/api/health` | API status and model readiness with stable `code` |
| GET | `/api/models/active` | Checkpoint hash as `id`, version, ordered labels, Keywords, features, frozen detector config, experimental warning |
| POST | `/api/recordings` | Multipart field `file`, returns metadata with `id`, HTTP 201 |
| GET | `/api/recordings` | Local Recording metadata, ordered by creation time and ID |
| GET | `/api/recordings/{id}` | Metadata, analysis `status`, `error_code`, `model_id`, selected `keywords`, detector config |
| POST | `/api/recordings/{id}/analyze` | JSON `{"keywords":["go","stop"]}`, returns queued metadata, HTTP 202 |
| GET | `/api/recordings/{id}/detections` | Stored Detections and analysis status; poll Recording metadata until terminal status |
| GET | `/api/recordings/{id}/audio` | Original WAV; HTTP byte ranges, suffix ranges, and unsatisfiable-range responses |
| DELETE | `/api/recordings/{id}` | Delete audio, metadata, and child Detections, HTTP 204 |
| GET | `/api/evaluations` | Stored Evaluation Run summaries with provenance and target; `code` is `runs_available` or `not_evaluated` |
| GET | `/api/evaluations/{id}` | One Evaluation Run with kind-specific metrics, from that run alone |
| POST | `/api/evaluations/import` | Import the committed run files, returns `imported` IDs and `rejected` sources with codes |
| DELETE | `/api/evaluations/{id}` | Delete one stored Evaluation Run, HTTP 204 |

Upload accepts non-empty integer PCM WAV at 8-96 kHz, mono or stereo,
8/16/24/32-bit. The service inspects headers, validates actual sample-byte count,
and enforces 40 MiB and 600 seconds. Both declared and streamed request sizes are
bounded; multipart framing gets a separate 64 KiB allowance. Filenames are display
metadata with a `.wav` suffix; paths and command characters are rejected.
The existing feature pipeline converts inference input to 16 kHz mono float32.

New uploads have status `uploaded`. Analysis proceeds through
`queued`, `running`, and `completed` or `failed`. One worker runs one Recording
at a time, with two waiting jobs by default. A duplicate analysis returns 409;
a full queue returns 429 without changing the Recording. Reanalysis replaces
previous Detections and saves its selected Keywords, checkpoint ID, and detector
config. Polling or playback disconnects do not cancel a durable job. Upload
disconnects remove partial data. Shutdown stops work between batches of at most
32 windows, joins the worker, and marks unfinished jobs `service_interrupted`.
Startup also marks interrupted jobs failed and removes orphan audio files.

Detection responses contain `recording_id`, `model_id`, `status`, `code`,
`detections`, `detector`, and experimental metadata. Each Detection contains
`id`, `keyword`, `start_s`, `end_s`, `confidence`, `threshold`, and `windows`.
A completed job with no Detections returns HTTP 200, `code: "no_detections"`,
and `detections: []`. It never invents a label. Before completion the code is
`analysis_not_completed`; failed jobs expose their stored error code.

## Evaluation Runs

Every dashboard number comes from a stored Evaluation Run ([ADR-0005](adr/0005-run-based-evaluation.md)).
`POST /api/evaluations/import` reads `*/*.json` under the operator-configured
directory; requests never name files. Files without `run_id` (detector configs,
training logs, comparisons) are skipped. Each run is validated before storage:
a 64-hex checkpoint SHA-256, dataset and split versions, a config hash, and a
timestamp with UTC offset are required. Counts must reconcile: confusion-matrix
rows sum to per-class support and to the example count; Keyword hits plus misses
equal the scored Keyword events; hits plus false Detections equal the Detections.
Accuracy and macro-F1 must follow from the confusion matrix and per-class F1, and
event precision, recall, and F1 from the hit, false, and miss counts, so no run
can carry a number its own counts contradict. The stored `met` flag must agree
with the achieved metric and its goal. Unreadable files are rejected one by one.
Rejections use `invalid_evaluation_run`, `missing_ground_truth` (no labeled
examples, Keyword events, or negative exposure: unlabeled audio has no metrics),
or `duplicate_run_id`. Import replaces stored runs with the same ID, so a deleted
committed run returns only after an explicit import. Startup never imports.

Each summary and detail carries `id`, `kind`, `experiment_id`, `created_at`,
`source` (path relative to the run directory), and `provenance`:
`checkpoint_sha256`, `dataset_version`, `split_version`, `split`, `config_sha256`,
`config_role`, and `code_revision` with its `dirty` flag. For `isolated` runs the
config is the training config (`config_role: "training"`). For `continuous` runs
it is the frozen detector config, identified by the detector file SHA-256
(`config_role: "detector"`). `target` holds `metric` (`macro_f1` or `event_f1`),
`goal`, `achieved`, `met`, and the never-redefined note.

Detail adds `provenance.config` and `experimental`. Isolated classification and
continuous-event numbers never share a response:

- `isolated`: `metrics` has `examples`, `accuracy`, `macro_f1`, `per_class`
  (label order, with `support`), and `confusion_matrix` (`labels`, rows `true`,
  columns `detected`, counts). `training` has epochs, `best_epoch`, and per-epoch
  `curves` of train/validation loss and accuracy plus validation macro-F1.
- `continuous` (`experimental: true`): `metrics` has `detections`, hit/false/miss
  counts, event `precision`, `recall`, `f1`, `false_alarms` (`count`,
  `negative_seconds`, `per_hour`, `definition`, and `by_region` for Unknown and
  Silence), `ignored_matches`, and `per_keyword`. `detector` names tolerance,
  match rule, and the tuning split; `composition` gives Keyword event counts,
  annotated and unlabeled seconds, speaker IDs, and annotation quality.
  `limitations` and `confidence_note` state that Confidence is a score only.
  Continuous runs never report accuracy.

A missing or deleted run returns 404 `evaluation_not_found`. Clients clear every
view that depended on it; there is no fallback to another run. The list `code`
covers all runs, so a dashboard tab with no run of its `kind` shows Not evaluated
on its own. A run describes the active model when `provenance.checkpoint_sha256`
equals the `id` from `GET /api/models/active`.

## Errors and local access

Application errors use `{"error":{"code":"...","message":"..."}}`.

| Code | HTTP status |
| --- | --- |
| `invalid_wav`, `unsupported_wav` | 400 |
| `upload_disconnected` | 400 when a response can still be delivered |
| `invalid_filename`, `invalid_upload`, `invalid_request`, `unsupported_keyword` | 422 |
| `upload_too_large`, `recording_too_long` | 413 |
| `recording_not_found`, `evaluation_not_found` | 404 |
| `analysis_in_progress`, `delete_failed` | 409 |
| `audio_unavailable` | 410 |
| `queue_full`, `rate_limited`, `uploads_busy` | 429 |
| `checkpoint_not_installed`, `model_incompatible`, `vocabulary_not_reviewed`, `worker_unavailable`, `storage_error`, `persistence_error`, `evaluations_unavailable` | 503 |

`vocabulary_not_reviewed` means an expanded offline checkpoint was configured for
serving before its data and Evaluation Runs were reviewed. Health reports the
model as unavailable; active-model and live requests are blocked. Restore the P0
checkpoint and its matching detector. See [the vocabulary gate](data/vocabulary-expansion.md).

Worker inference failures use Recording status `failed` and `error_code: "analysis_failed"`.
Metadata write failures use `persistence_error`. If even saving failure status is
unavailable, readiness reports `worker_unavailable`, new analyses return 503, and
the worker retries that status write every 250 ms before accepting more work.
Shutdown still joins the worker if metadata remains unwritable; startup recovers
unfinished states when storage is writable again.

Public errors omit local paths and speech content. Deletion stages audio with a
reversible rename before deleting metadata. A failed database commit restores
playable audio; failed removal of committed staged bytes returns `delete_failed`
and can be retried with the same DELETE request. Startup restores staged audio
for surviving metadata and removes orphan bytes after committed deletions.
A deleted Recording cannot receive late worker results.

Host headers must name `127.0.0.1`. Browser origins are limited to
`http://127.0.0.1:3000` and `http://localhost:3000`. CORS allows the recording
methods and playback range headers. There are two concurrent upload slots and
a shared limit of 60 POST/DELETE requests per minute. These limits and smaller
duration/byte/queue limits can be set through `Settings` when embedding the app.
The ten-minute ceiling is a validation limit, not a target-hardware performance claim.

SQLite enables foreign keys and stores model identities, Recordings, Detections,
Evaluation Runs, and live-event metadata. `PRAGMA user_version` tracks the schema;
version 1 replaces the earlier reserved, never-written `evaluation_runs` shape.
The `live_events` table stores live decision metadata only (see below). No raw mic
audio is stored.

## Live voice session

| Method | Route | Response |
| --- | --- | --- |
| GET | `/api/actions` | Keyword-to-action allowlist, rejected labels, live policy, protocol limits |
| WebSocket | `/api/live` | One explicit mic session; versioned JSON control plus binary PCM frames |

`GET /api/commands` remains as a deprecated alias for `GET /api/actions`.

The action map is a strict allowlist of in-app demo actions
([ADR-0004](adr/0004-in-app-actions-only.md)): `right` `next_slide`, `left`
`previous_slide`, `go` `play`, `stop` `pause`, `up` `volume_up`, `down`
`volume_down`. Unknown and Silence never map to an action. By default these affect
only the in-app demo. The separately consented experimental Windows integration
is documented in [OS actions](os-external-actions.md); its permission and execution
path is separate from this map and from inference.

The browser opens the socket only after the user presses Start
([ADR-0003](adr/0003-local-websocket-mic.md)). Origins follow the HTTP allowlist;
other browser origins are refused before the handshake. Within 5 seconds the client
sends `{"type":"start","version":1,"sample_rate":48000,"channels":1,"encoding":"pcm_s16le"}`.
The server replies `ready` with a session ID, Keywords, actions, and policy, or an
`error` message followed by close: `invalid_message` and `unsupported_version` (1008),
`unsupported_audio` (1003, mono PCM_s16le at 8-96 kHz only), the model-unavailable
codes from health (1011), or `live_busy` (1013; one session at a time).

After `ready` the client streams binary mono PCM_s16le frames of at most 32 KiB at
its native rate. The server keeps about one window of mic audio in memory, resamples
each 1 s window to 16 kHz through the shared feature pipeline, and scores a window
every 0.25 s. Each window yields a `frame` message (`t_s`, top `label`, `score`).
Decisions follow the initial live policy, which is not yet tuned on validation data:
consecutive windows with the same top label form one utterance; a Keyword is
`accepted` when 2 windows in a row reach Confidence 0.7 and 1.2 s have passed since
the last accepted action; otherwise it is rejected as `cooldown` or, when the run
ends, `low_confidence`. Unknown and Silence runs are rejected once each as `unknown`
or `silence`. One utterance causes at most one action. A `decision` message carries
`id`, `t_s`, `label`, `score`, `accepted`, `reason`, `action` (only when accepted),
and `processing_ms` from the last received frame to the decision.

When unscored audio exceeds 1 s, the server drops the stale audio instead of
queueing it, ends the current utterance without a decision (cooldown is kept), and
sends `backpressure` with `dropped_s`. A malformed or oversized frame closes the
session with `invalid_frame` (1009). The client reports each in-app result with
`{"type":"action_result","decision_id":N,"status":"applied"|"no_change"}`.
`{"type":"stop"}` returns `stopped` and closes with 1000. Pressing the Stop
button, tab-hide, page exit, close, or disconnect ends the session immediately
and frees its in-memory audio. Saying the Keyword "stop" triggers the pause
action only; it never ends the session.

Each decision is stored in `live_events` as session ID, label, Confidence, accepted
flag, action, and action status (`dispatched`, then the reported result). Raw mic
audio is never written. Uvicorn needs the `websockets` package for this route; it is
a locked backend dependency.

## Validation

`backend/tests/test_live.py` covers the decision policy (smoothing, cooldown,
one action per utterance, Unknown/Silence rejection), stale-audio dropping, and the
WebSocket contract with a biased fixture checkpoint, including the stored metadata
and the absence of stored audio.
`backend/tests/test_recordings.py` exercises the API with a generated PCM fixture
and a real, deliberately biased test checkpoint. These are plumbing fixtures,
not evidence of speech accuracy. It covers persistence across restart, ranged
playback (including suffix ranges and unsatisfiable-range 416 responses),
deletion, validation errors, queue saturation, shutdown, source loss,
worker recovery, and upload disconnects. `backend/tests/test_evaluations.py`
imports the committed Evaluation Runs and covers provenance, kind separation,
deletion across restart, rejected documents, and the schema upgrade.
`npm run check` runs the repository checks.
