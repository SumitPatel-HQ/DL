# Local recording service

Start one API process with `npm run api`. The launcher binds to `127.0.0.1:8000`.
Run one service process per storage directory. The default directory is ignored
`data/service/`, containing `metadata.sqlite3` and `audio/<generated-id>.wav`.
Requests cannot select filesystem paths, models, Python functions, or executables.

The default active model is `artifacts/models/e02/model.pt`, paired with
`docs/evaluations/continuous-v1/detector-e02.json`. Startup checks the checkpoint
format, labels, features, and detector/checkpoint hash agreement. Missing or
incompatible artifacts leave the API running with explicit model-unavailable status.
Operator environment variables `KWS_STORAGE`, `KWS_CHECKPOINT`, and `KWS_DETECTOR`
override these paths before startup. Restart to change the active model.

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

## Errors and local access

Application errors use `{"error":{"code":"...","message":"..."}}`.

| Code | HTTP status |
| --- | --- |
| `invalid_wav`, `unsupported_wav` | 400 |
| `upload_disconnected` | 400 when a response can still be delivered |
| `invalid_filename`, `invalid_upload`, `invalid_request`, `unsupported_keyword` | 422 |
| `upload_too_large`, `recording_too_long` | 413 |
| `recording_not_found` | 404 |
| `analysis_in_progress`, `delete_failed` | 409 |
| `audio_unavailable` | 410 |
| `queue_full`, `rate_limited`, `uploads_busy` | 429 |
| `checkpoint_not_installed`, `model_incompatible`, `storage_error`, `persistence_error` | 503 |

Worker failures use Recording status `failed` and `error_code: "analysis_failed"`.
Public errors omit local paths and speech content. Failed audio deletion retains
metadata for retry; a deleted Recording cannot receive late worker results.

Host headers must name `127.0.0.1`. Browser origins are limited to
`http://127.0.0.1:3000` and `http://localhost:3000`. CORS allows the recording
methods and playback range headers. There are two concurrent upload slots and
a shared limit of 60 POST/DELETE requests per minute. These limits and smaller
duration/byte/queue limits can be set through `Settings` when embedding the app.
The ten-minute ceiling is a validation limit, not a target-hardware performance claim.

SQLite enables foreign keys and stores model identities, Recordings, Detections,
Evaluation Runs, and live-event metadata. Evaluation import/read APIs and the live
WebSocket/action implementation belong to later tickets. No raw mic audio is stored.

## Validation

`backend/tests/test_recordings.py` exercises the API with a generated PCM fixture
and a real, deliberately biased test checkpoint. These are plumbing fixtures,
not evidence of speech accuracy. It covers persistence across restart, ranged
playback, deletion, validation errors, queue saturation, shutdown, source loss,
worker recovery, and upload disconnects. `npm run check` runs the repository checks.
