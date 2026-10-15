# System Requirements Document (SRD)

**System:** LSTM Keyword Spotting / Audio Analysis / Voice Control / ML Dashboard  
**Version:** 1.0 (proposed) · **Date:** 15 October 2026 · **Status:** Architecture baseline, subject to feasibility validation  
**Related:** [Overview](KWS_PROJECT_OVERVIEW.md) · [PRD](KWS_PRD.md) · [Implementation Plan](KWS_IMPLEMENTATION_PLAN.md)

## 1. Purpose and boundaries

This document turns the PRD into testable software, data and ML requirements. The system is **single-user and local-first**, developed for **Windows** on **AMD Ryzen 5 5500U + 16 GB RAM** without relying on GPU acceleration. The user-facing application runs in a browser; a Python backend on the **same machine** carries out inference, persistence and approved actions.

The model identifies **trained keywords**, not arbitrary spoken instructions. A detection must pass postprocessing before it can become an application action. Continuous-audio event localization is initially an **experimental sliding-window approach** and must be evaluated separately from isolated-word classification.

## 2. Logical architecture

```text
Next.js (TypeScript)
  ├── Audio Analysis: upload, waveform, detections
  ├── Voice Control: permission, streaming mic, in-app player/slides
  └── Evaluation: model metadata, metrics, charts
        |
        |  REST HTTP + localhost WebSocket
        v
FastAPI application
  ├── Request validation / file lifecycle / health checks
  ├── Upload analysis jobs (bounded worker execution)
  ├── Live audio session controller + ring buffer
  ├── Keyword detection service (shared model)
  │    ├── WAV/PCM normalization -> 16 kHz mono
  │    ├── frames + MFCC/Log-Mel extraction
  │    ├── PyTorch LSTM inference
  │    └── threshold, smoothing, merging, cooldown
  ├── Allowlisted action dispatcher
  └── Experiment / model metadata + evaluation endpoints
        |
        +-- SQLite: models, recordings, detections, evaluation runs, actions
        +-- Local files: audio, checkpoints, metrics, artifacts
```

**Deployment rule:** Next.js and FastAPI are separate development processes initially; packaging into an installable Windows application is not P0. If later remotely deployed, OS-action execution must be redesigned rather than assuming that browser requests control a user's PC.

## 3. Component responsibilities

| Component | Responsibilities | Must not do |
|---|---|---|
| Frontend | Mic permissions, upload UI, waveform playback, charts, status/error display | Calculate authoritative metrics or execute arbitrary OS commands |
| FastAPI | Validate requests, manage jobs, stream detection events, return stored runs | Train the model in request handlers or block the event loop with CPU inference |
| Preprocessor | Decode PCM, resample, normalize, feature extraction | Fit normalization statistics on test data |
| Model service | Load compatible checkpoint, infer class scores, expose model version | Claim confidence is actual accuracy |
| Event detector | Turn overlapping scores into intervals, suppress duplicates | Trigger actions directly without validation |
| Action service | Map accepted labels to a limited set of actions | Shell execution, free-form commands, implicit elevated privileges |
| Evaluator | Match events to ground truth, compute metrics, store provenance | Evaluate on training examples as if they were unseen |
| Persistence | Keep metadata and local file references, support deletion | Store raw microphone data by default |

## 4. Functional software requirements

- **SR-F01:** On startup, verify the checkpoint, feature configuration, SQLite schema and storage folders; return explicit readiness status.
- **SR-F02:** Accept a WAV recording with bounded file size/duration, inspect codec/channels/sample rate and reject unsupported or corrupted files.
- **SR-F03:** Convert accepted audio to 16 kHz mono float32 PCM for model inference. Validate NaN/infinite samples and handle silence safely.
- **SR-F04:** Apply the **same versioned feature extraction** during training, testing, batch inference and live inference.
- **SR-F05:** Load one active model with ordered class labels including `unknown` and silence/noise handling; do not advertise untrained words.
- **SR-F06:** Run overlapping-window inference on uploaded recordings and return candidate detections with start/end timestamps and scores.
- **SR-F07:** Apply class-specific or global thresholds, temporal smoothing and non-maximum/duplicate suppression, with all parameters recorded in run metadata.
- **SR-F08:** Persist a recording job's state as `queued`, `running`, `completed` or `failed`; surface errors through REST/UI.
- **SR-F09:** Stream microphone PCM in an explicit live session; stop processing immediately after explicit session termination/disconnect.
- **SR-F10:** Apply a cooldown/action deduplication rule to live accepted detections.
- **SR-F11:** Dispatch `right/left/go/stop/up/down` only to predefined application functions, and report action result or rejection.
- **SR-F12:** Display real event locations in WaveSurfer, seek/play regions and expose detections in a sortable list.
- **SR-F13:** Produce classification and continuous-event evaluation reports from labeled independent sets, with per-run provenance.
- **SR-F14:** Delete recording data, file bytes and dependent detection rows upon explicit user deletion; record the outcome.
- **SR-F15:** Prevent uploads or API events from specifying arbitrary Python functions, executable paths, commands or scripts.

## 5. Audio and ML technical specification

### 5.1 Input contract

| Property | Initial decision | Notes |
|---|---|---|
| Recording format | `.wav` containing PCM audio | Reject others initially; FFmpeg conversion later |
| Uploaded duration | Up to 10 minutes (configurable) | Validate practical limit on hardware; fail early otherwise |
| Sample-rate canonicalization | 16,000 Hz | Resample other valid WAV rates |
| Channels | Mono | Average/downmix with documented method |
| Samples | Float32 internally | Clip/constrain and check finite values |
| Frame analysis | 25 ms frame, 10 ms hop | Configurable, stored in feature metadata |
| Features | 40 MFCC coefficients per frame initially | Baseline choice, compare against Log-Mel later |
| Class vocabulary | `left/right/up/down/go/stop/unknown/silence` | Exact ordered labels stored with checkpoint |

### 5.2 Proposed model baseline

```text
1 second audio window (16000 samples)
    -> MFCC tensor [time_frames, 40]
    -> batched tensor [B, T, F] (batch_first=True)
    -> LSTM (hidden=64 or 128, num_layers=1 or 2)
    -> final hidden representation / masked pooling
    -> optional dropout + Linear(128 -> number_of_classes)
    -> logits
    -> cross-entropy during training / softmax for display
```

- Optimize with Adam; train with early stopping and a fixed-seed experiment configuration.
- Set dropout on a single-layer model **outside** the LSTM; PyTorch's `nn.LSTM(dropout=...)` applies dropout between stacked layers, not after a sole layer.
- Cache extracted feature tensors where feasible to reduce repeated CPU-heavy work.
- Choose minibatch size empirically (e.g. 16 or 32) to avoid RAM pressure; do not assume specific epoch times.
- Preprocessing metadata must include version, sample rate, window/hop, feature count, normalization, class mapping and checkpoint hash.

### 5.3 Continuous spotting mechanism — initial research implementation

1. Convert recording to canonical PCM.
2. Create overlapping inference windows (initially 1.0 s wide, 0.2–0.25 s step).
3. Run same isolated-word model on each window.
4. Apply validated score threshold and a requirement for consistent adjacent-window evidence.
5. Merge overlapping positives of the same label, suppress repeated detections, return estimated intervals.
6. Store raw window scores optionally for debugging; do not show them as ground-truth probabilities.

**Limitation:** an isolated-word model may misclassify natural connected speech and the window boundaries only approximate a word's true start/end. A model can score well on Speech Commands yet fail continuous-event evaluation. A **go/no-go evaluation gate** in the implementation plan decides whether a dedicated temporal detector (such as LSTM+CTC) is necessary.

### 5.4 Live session behavior

- Microphone data comes from browser `getUserMedia` and AudioWorklet, after explicit user action. Localhost counts as a permitted secure context in supported modern browsers; still handle permission denial.
- Browser captures available device sampling rate, sends mono 16-bit PCM chunks with a session handshake; backend resamples if needed.
- Backend uses a bounded ring buffer, builds same inference windows and pushes result events.
- Duplicate predictions across windows **must not** create repeated actions; use threshold, consecutive evidence and cooldown (initially ~1.2 s; tune on validation).
- Session ends on stop, permission revocation, navigation or WebSocket disconnect. Do not retain unrequested live PCM by default.

## 6. HTTP API proposal

All JSON fields are illustrative contracts; freeze them once the first integration test is written.

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/health` | Backend and model readiness |
| GET | `/api/models/active` | Model ID, labels, feature spec, version |
| POST | `/api/recordings` | Multipart WAV upload; returns recording/job ID |
| GET | `/api/recordings/{id}` | Metadata and processing status |
| POST | `/api/recordings/{id}/analyze` | Start analysis (selected known labels) |
| GET | `/api/recordings/{id}/detections` | Stored detection events |
| GET | `/api/recordings/{id}/audio` | Audio playback; optionally support HTTP range |
| DELETE | `/api/recordings/{id}` | Delete recording, audio and child records |
| GET | `/api/evaluations` | List labeled evaluation runs |
| GET | `/api/evaluations/{id}` | Metrics, confusion matrix, curves and provenance |
| GET | `/api/commands` | Authorized keyword-to-action map |

**Example recording detection JSON:**

```json
{
  "recording_id": "rec_01",
  "model_version": "lstm-baseline-001",
  "detections": [
    {
      "id": "det_01",
      "keyword": "stop",
      "start_s": 21.8,
      "end_s": 22.6,
      "score": 0.93,
      "decision_threshold": 0.85
    }
  ],
  "timestamp_quality": "approximate_sliding_window"
}
```

A zero-detection recording returns `detections: []` rather than a guessed label.

## 7. WebSocket contract

**Route:** `ws://127.0.0.1:<port>/ws/live` during local development (or the equivalent configured host).

1. **Client JSON handshake:** `{"type":"start","sample_rate":48000,"channels":1,"encoding":"pcm_s16le","mode":"command"}`.
2. **Client audio chunks:** binary little-endian signed-16 PCM at declared rate. Audio bytes are not JSON/base64.
3. **Server event:** `{"type":"detection","keyword":"right","score":0.92,"accepted":true,"start_s":2.0,"end_s":2.8,"action":"next_slide","action_status":"completed"}`.
4. **Server status/error:** typed JSON messages with stable machine-readable error codes.
5. **Client stop:** `{"type":"stop"}`, then close. Server closes on backpressure violations or idle/disconnect.

Add protocol version, chunk sequencing and bounded queue safeguards before relying on live sessions. If the host uses another sample rate, the backend owns canonical resampling.

## 8. Data model (SQLite)

| Table | Important fields |
|---|---|
| `models` | `id`, `version`, `checkpoint_path`, `sha256`, `labels_json`, `feature_config_json`, `created_at` |
| `recordings` | `id`, `filename`, `duration_s`, `sample_rate`, `audio_path`, `status`, `error_code`, `created_at` |
| `detections` | `id`, `recording_id`, `model_id`, `keyword`, `start_s`, `end_s`, `score`, `threshold`, `created_at` |
| `evaluation_runs` | `id`, `model_id`, `dataset_id`, `split_id`, `config_json`, `metrics_json`, `artifacts_path`, `created_at` |
| `live_events` | `id`, `session_id`, `keyword`, `score`, `accepted`, `action`, `action_status`, `created_at` |

Only metadata/events are stored for microphone sessions by default. Use foreign keys and cascading deletions for recording-owned detections; delete source bytes in a controlled service transaction with explicit failure handling.

## 9. Evaluation specifications

### Isolated-word classification

- Accuracy, macro precision, macro recall, macro-F1, per-class precision/recall/F1 and confusion matrix.
- Use held-out test split after all threshold/hyperparameter tuning has been completed on validation data.
- Display count of each ground-truth class and confusion over `unknown` and `silence`, not just the six target words.

### Continuous-audio keyword events

- Event label + ground-truth onset/offset timestamps from manually checked annotations.
- A one-to-one matched prediction requires the **same keyword** and detection-center timestamp within a prespecified tolerance (start with ±0.5 s; freeze before evaluating test data). Each ground-truth event is matched at most once.
- `precision = TP / (TP + FP)`, `recall = TP / (TP + FN)`, `F1 = 2PR/(P+R)`; define behavior for empty denominators explicitly.
- Track **false alarms/hour** on negative audio (predicted false events ÷ evaluated negative audio hours), mean/median processing/inference time and time-to-detection in live sessions.
- Report evaluation set size and composition, silence/noise conditions and speaker separation. Distinguish this from per-window accuracy.

### Experiment reproducibility

Every evaluation references one immutable checkpoint hash, configuration hash, dataset/split version, timestamp and source commit hash (when repository exists). Keep training curves as artifacts for the dashboard.

## 10. Non-functional requirements

| Category | Technical requirement |
|---|---|
| Execution | CPU-only installation and inference must function without CUDA/ROCm |
| Memory | Avoid OOM on 16 GB system through bounded uploads, chunked inference, bounded live queues |
| Latency | Instrument end-of-speech to accepted event/action, report actual numbers, target median ≤1 s |
| Reliability | Stop/cleanup microphone sessions and background analysis jobs on failures/disconnect |
| Compatibility | Windows 10/11 development, documented modern Chromium-based browser |
| Data safety | Uploaded audio stays local; delete works; no raw mic recording without opt-in |
| API security | Bind backend to loopback by default; validate origin/host, WebSocket origin, size and rate limits |
| OS actions | Explicit allowlist, no command-line execution or arbitrary app control; enabled only for local authorized user |
| Observability | Structured error codes and status transitions; no raw speech contents in logs |
| Testability | Pure ML preprocessing module, deterministic tests and fixtures, mockable action engine |
| Accessibility | Button labels, keyboard operations, distinct recording status and clear permission denial state |

## 11. Failure-mode requirements

- **Missing model/checkpoint:** API reports unhealthy model, frontend displays setup action; do not emit fake detections.
- **Corrupt/unsupported file:** HTTP 4xx, preserve a useful error, no stored partial job after cleanup.
- **No detections:** empty result with explanatory status; no action.
- **Microphone unavailable/permission denied:** controls disabled with specific prompt; no background retry loop.
- **WebSocket disconnect:** session and audio buffers disposed; no lingering actions.
- **CPU overloaded:** drop/backpressure with visible warning instead of growing queue unbounded.
- **Low-confidence keyword:** log rejected event if the user enabled diagnostic logs; never execute action.
- **Data deletion failure:** report partial failure rather than pretending all local bytes are gone.

## 12. Verification matrix

| Requirement | Verification method |
|---|---|
| SR-F02/03 | WAV sample rates, stereo, silence, corrupt/oversize fixture tests |
| SR-F04/05 | Train/inference feature hash and class-order compatibility tests |
| SR-F06/07 | Synthetic timestamp fixtures + real connected-speech annotated test |
| SR-F08/14 | Integration test for job state and cascade deletion |
| SR-F09/10 | Browser mic smoke test + duplicate/cooldown simulated stream |
| SR-F11/15 | Allowed/disallowed commands tests; injection attempts rejected |
| SR-F12 | Playwright marker-seek and empty-results tests |
| SR-F13 | Known confusion-matrix fixture, event matching and FAR math test |
| CPU feasibility | Measured local training, inference, load, peak memory and UI responsiveness |

## 13. Architecture decision records to create

- `ADR-0001`: PyTorch LSTM baseline; CPU-first, no cloud inference.
- `ADR-0002`: Sliding-window continuous spotting is experimental; define validation gate and alternatives.
- `ADR-0003`: Browser mic to local FastAPI over WebSocket; local-only security model.
- `ADR-0004`: In-app allowlisted actions initially; OS integrations deferred.
- `ADR-0005`: Evaluation metrics are run-based and ground-truth-dependent.

## 14. Technical references

- [PyTorch `torch.nn.LSTM` API](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [Speech Commands dataset](https://www.tensorflow.org/datasets/catalog/speech_commands)
- [FastAPI WebSockets](https://fastapi.tiangolo.com/advanced/websockets/)
- [WaveSurfer.js Regions](https://wavesurfer.xyz/docs/)
- [Windows Core Audio support through Pycaw (optional)](https://github.com/AndreMiras/pycaw)
