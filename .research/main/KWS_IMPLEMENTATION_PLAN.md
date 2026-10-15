# Implementation Plan — LSTM Keyword Spotting Application

**Version:** 1.0 (proposed) · **Date:** 15 October 2026  
**Target:** Local Windows + Ryzen 5 5500U / 16 GB RAM / AMD iGPU (CPU-first)  
**Related:** [Overview](KWS_PROJECT_OVERVIEW.md) · [PRD](KWS_PRD.md) · [SRD](KWS_SRD.md)

## 1. Delivery strategy

Build **one small validated DL pipeline first**, then reuse it for (a) recorded keyword detection and (b) short live commands. Add UI last enough to demonstrate progress, but do not create a polished dashboard before the ML results are real.

This project has a crucial uncertainty: a model trained with isolated Speech Commands clips may not reliably find words inside natural sentences. Therefore a **continuous-speech feasibility gate** occurs **before** substantial front-end integration.

### Guiding rules

1. **No placeholder inference:** real LSTM weights and preprocessing in every end-to-end demo.
2. **No fabricated metrics:** all dashboard numbers originate from labeled evaluation runs.
3. **CPU-first:** optimize for repeatable laptop experiments, not a hidden CUDA dependency.
4. **Safe actions:** begin with in-app controls; external Windows control is optional.
5. **Measured scope:** support a known word list, not arbitrary spoken instructions.
6. **Small vertical slices:** each phase ends with an executable test/demo and a documented acceptance gate.

## 2. Suggested repository structure

```text
keyword-spotting/
├── docs/
│   ├── KWS_PROJECT_OVERVIEW.md
│   ├── KWS_PRD.md
│   ├── KWS_SRD.md
│   ├── KWS_IMPLEMENTATION_PLAN.md
│   └── adr/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI entry
│   │   ├── api/                     # REST and WebSocket endpoints
│   │   ├── services/                # recording jobs, inference, actions
│   │   ├── models/                  # Pydantic and persistence entities
│   │   └── db/                      # SQLite session / migrations
│   ├── ml/
│   │   ├── configs/                 # training + inference configuration
│   │   ├── data/                    # dataset loading/splits
│   │   ├── features/                # waveform preprocessing + MFCC
│   │   ├── networks/                # LSTM implementations
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── infer.py
│   │   └── spotting.py              # windowing + events
│   ├── tests/
│   └── pyproject.toml
├── frontend/
│   ├── app/                         # Next.js routes
│   ├── components/                  # waveform, charts, status
│   ├── lib/                         # typed API client / WS session
│   └── tests/
├── artifacts/
│   ├── models/                      # checkpoint + metadata
│   └── evaluations/                # immutable run summaries/plots
├── scripts/                         # reproducible experiment commands
├── data/                            # ignored: downloaded or private raw audio
├── .gitignore
└── README.md
```

Commit lightweight model metadata/configs and example evaluation summaries as appropriate. Do not commit private speech recordings or giant datasets; store local raw data under `.gitignore` and give download/attribution instructions.

## 3. Phased implementation plan

Time estimates are **planning ranges**, not guarantees. Allow approximately **5–8 weeks part-time** for a defensible full version; the ML feasibility gate can change the plan.

### Phase 0 — Specification, reproducibility and local environment (2–3 days)

**Tasks**

- [ ] Initialize repo, Python virtual environment, Next.js workspace and locked dependency manifests.
- [ ] Add Ruff/formatter, pytest, TypeScript linting and an automated test command.
- [ ] Establish CPU-only PyTorch and local FastAPI health endpoint.
- [ ] Define model labels, inference thresholds, audio format, feature config and reproducibility conventions.
- [ ] Create dataset licensing/consent notes and `.gitignore` protections.
- [ ] Capture baseline system info: OS, CPU, available RAM, package versions.

**Exit criteria:** clean setup launches API, test suite and a minimal web page; `/api/health` clearly reports model-unavailable until a checkpoint is installed.

### Phase 1 — Data pipeline and baseline LSTM (5–8 days)

**Tasks**

- [ ] Obtain a permitted Speech Commands dataset/subset, track provenance and speaker IDs.
- [ ] Build train/validation/test splits with no speaker leakage.
- [ ] Use `left/right/up/down/go/stop`; add proper `unknown` and silence/noise examples.
- [ ] Implement mono conversion, 16 kHz resampling, frame extraction and MFCC features.
- [ ] Test train/inference preprocessing parity on the same WAV fixture.
- [ ] Implement compact 1–2 layer PyTorch LSTM classifier; cross-entropy + Adam + early stopping.
- [ ] Save checkpoint **with** class order and full feature settings.
- [ ] Export learning curves, confusion matrix and held-out test metrics.
- [ ] Benchmark CPU training epoch time, peak RAM and inference time.

**Exit criteria:** `python -m ml.train ...` (or documented equivalent) creates a reloadable checkpoint; `python -m ml.evaluate ...` reproducibly reports macro-F1 and a confusion matrix from held-out data. Desired starting target: isolated-word macro-F1 ≥ 0.85, subject to measured baseline.

**Learning checkpoint:** explain LSTM input dimensions, gates, backpropagation through time, loss, overfitting, dropout and why validation data is not test data.

### Phase 2 — Continuous-speech feasibility gate (5–9 days)

**Why now:** this phase determines whether the planned audio-analysis feature has a valid technical foundation.

**Tasks**

- [ ] Create **natural spoken sentences** that contain selected keywords, plus unrelated/hard-negative speech; obtain consent for recorded samples.
- [ ] Annotate keyword and start/end times in a simple `recording_id,keyword,start_s,end_s` CSV.
- [ ] Build an independent continuous-speech test partition from different recordings/speakers.
- [ ] Implement overlapping-window LSTM scoring (~1 s window, ~0.2–0.25 s stride).
- [ ] Add threshold calibration using validation clips, smoothing, event merging and duplicate suppression.
- [ ] Report precision/recall/F1 on *events* and false alarms/hour on negative recordings.
- [ ] Manually review missed matches, false matches and timestamp errors; save representative error examples.

**Decision gate:**

| Outcome | Decision |
|---|---|
| Continuous-event F1 reaches preliminary ≥ 0.70 on held-out in-domain clips, with acceptable false-alarm behavior | Continue with the sliding-window detection service; document limits |
| Performance is below target but there is a credible, small improvement path | One bounded tuning cycle; rerun same held-out protocol with validation-only tuning |
| Natural-speech detection is consistently unreliable | **Do not market the feature as working.** Keep isolated keyword recognition as validated core; prototype LSTM+CTC/temporal model as a separately scoped extension, or renegotiate the continuous-audio acceptance goal transparently |

Do not substitute synthetically concatenated isolated utterances for natural connected speech at this gate.

### Phase 3 — Backend and local file-analysis workflow (4–6 days)

**Tasks**

- [ ] Add FastAPI routes for health, active model, recording upload, analysis, job status, detections and deletion.
- [ ] Persist recordings, detections and model metadata in SQLite.
- [ ] Implement constrained job worker with queue bounds and status transitions.
- [ ] Run the evaluated windowing/detection pipeline behind the API.
- [ ] Add stable errors (bad WAV, too long, unsupported keyword, missing checkpoint, no detections).
- [ ] Implement playback endpoint (and HTTP range support if the waveform player needs it).
- [ ] Test deletion of source audio and associated database records.

**Exit criteria:** API test can upload a labeled test WAV, request analysis, poll completion, retrieve authentic detection intervals, and delete all local artifacts.

### Phase 4 — Audio Analysis UI + waveform (4–6 days)

**Tasks**

- [ ] Build Next.js layout with Audio Analysis / Voice Control / Evaluation navigation.
- [ ] Add upload/select keywords/progress/errors and completed-job details.
- [ ] Render WaveSurfer waveform/timeline with Regions for actual model events.
- [ ] Make clicking detection seek and play the relevant segment.
- [ ] Add no-match state, recording deletion and optional CSV export.
- [ ] Add a Playwright happy-path plus corrupt-file/empty-result coverage.

**Exit criteria:** a user uploads a WAV through the UI and can visibly inspect and replay actual detections. The UI never displays invented confidence or accuracy.

### Phase 5 — Live microphone + voice-controlled demo (5–7 days)

**Tasks**

- [ ] Implement user-activated browser microphone capture with `getUserMedia`/AudioWorklet.
- [ ] Stream mono PCM frames over a versioned FastAPI WebSocket protocol; resample to 16 kHz server-side.
- [ ] Add a bounded ring buffer and live inference with configurable window step.
- [ ] Implement threshold, short temporal smoothing, cooldown and diagnostics for rejected events.
- [ ] Map the trained words to **in-app** demonstration actions:
  - `right` → next slide;
  - `left` → previous slide;
  - `go` → play;
  - `stop` → pause;
  - `up` / `down` → player volume changes.
- [ ] Add start/stop controls, permission handling, listening indicator and visible action history.
- [ ] Measure median/95th percentile time from utterance end to action, and count false actions in scripted negative sessions.

**Exit criteria:** the real trained model causes bounded UI actions with demonstrable rejection/duplicate suppression and immediate microphone stop. No unrestricted Windows automation.

**Note:** if the model recognizes `go` but not `play`, the UI must advertise `go`, not claim that it understands the untrained synonym.

### Phase 6 — Real evaluation dashboard (3–5 days)

**Tasks**

- [ ] Store run IDs with checkpoint hash, dataset/split identifier, thresholds, features, timestamp and code version.
- [ ] Build `/api/evaluations` endpoints returning actual stored evaluation results.
- [ ] Render training vs validation loss, accuracy, per-class metrics and confusion matrix.
- [ ] Clearly distinguish isolated-word classification metrics from continuous-event metrics.
- [ ] Show false alarms/hour and latency only when such experiments have actually been performed.
- [ ] Label non-ground-truth audio predictions with **score/confidence**, not accuracy.
- [ ] Add meaningful empty state: `No evaluation run available`.

**Exit criteria:** every chart can be traced to a saved, reproducible experiment; deleting test results or switching models updates the dashboard appropriately.

### Phase 7 — Validation, report, and final demo (4–6 days)

**Tasks**

- [ ] Run full acceptance suite, including corrupted audio, silence, unknown words and app reconnects.
- [ ] Test at least one clean and one noisy labeled evaluation condition.
- [ ] Profile processing time, memory and CPU on the actual Ryzen laptop.
- [ ] Conduct a small, scripted mic test covering each supported command + negative phrases.
- [ ] Repeat main evaluation once from saved settings/checkpoint to verify reported results.
- [ ] Write results, failure cases, sources, limitations and reproducibility instructions.
- [ ] Record a faculty demo showing audio upload → detections/timestamps → live actions → real dashboard.

**Exit criteria:** all P0 product scenarios are demonstrated or explicitly marked as not meeting acceptance, with evidence.

## 4. Dependency order / critical path

```text
Phase 0: Local setup
      ↓
Phase 1: LSTM + labeled evaluation
      ↓
Phase 2: Real continuous-audio feasibility gate
      ↓
Phase 3: FastAPI recording analysis service
      ↓
Phase 4: Upload + waveform UI
      ↓
Phase 5: Live mic + safe command actions
      ↓
Phase 6: Real evaluation dashboard
      ↓
Phase 7: Final tests / report / demo
```

**Parallelizable:** while one part of the team prepares continuous-speech annotations, another may scaffold the frontend shell/API contracts. Do **not** merge a fake inference path into the main branch simply to unblock UI work; use clearly identified fixture data in tests only.

## 5. Testing strategy

| Test class | Essential test cases | When |
|---|---|---|
| Unit — preprocessing | Resampling, mono mix, silence, NaNs, MFCC dimensions, parity | Phases 1–2 |
| Unit — ML | Checkpoint config compatibility; expected logits shape; deterministic fixture | Phase 1 |
| Unit — event detection | Window overlap merging, tolerance matching, duplicate suppression | Phase 2 |
| Unit — metrics | Known TP/FP/FN, F1, confusion matrix, false alarms/hour | Phases 1–2 |
| Integration — backend | Upload → analyze → retrieve → delete; status failures | Phase 3 |
| Integration — streaming | PCM handshake, wrong rate/channels, disconnect/backpressure | Phase 5 |
| Integration — actions | Valid words only, cooldown, no arbitrary command injection | Phase 5 |
| End-to-end — frontend | WAV upload, marker seek, start/stop mic, empty/error state | Phases 4–7 |
| Performance | Local CPU/memory and end-of-utterance to action latency | Phases 5–7 |
| Model validity | Held-out natural-speech event evaluation plus controlled negatives | Phases 2 and 7 |

## 6. Work packages / ticket-ready backlog

| Ticket | Deliverable | Depends on |
|---|---|---|
| T01 | Repo scaffold, tooling, env sample, CI test command | — |
| T02 | Dataset manifest, labels, consent/license notes, clean splits | T01 |
| T03 | Canonical audio conversion and MFCC features | T02 |
| T04 | LSTM training pipeline, checkpoint and training logs | T03 |
| T05 | Isolated-word evaluator, confusion matrix, macro-F1 | T04 |
| T06 | Annotated natural continuous-speech set | T02 |
| T07 | Sliding-window event detector, calibration and event evaluator | T04,T06 |
| T08 | Feasibility review/ADR and go/no-go decision | T07 |
| T09 | FastAPI model/audio job API, SQLite schema | T04,T08 |
| T10 | Waveform Audio Analysis UI and detections list | T09 |
| T11 | Browser mic PCM WebSocket session | T09 |
| T12 | Command decision layer + in-app demo actions | T11,T07 |
| T13 | Evaluation endpoints + dashboard with actual metrics | T05,T07,T09 |
| T14 | E2E, performance, robustness tests and README | T10,T12,T13 |
| T15 | Academic report, presentation, limitations, final demo | T14 |

Keep ticket descriptions narrow: include input, output, tests and acceptance criteria before coding.

## 7. Dataset and experiment tracking plan

### Starter data

- First use Speech Commands subsets for quick LSTM training and class-level evaluation. The mini set's eight labels include `no/yes/down/go/left/up/right/stop`; it has **no built-in `next`/`pause` class**.
- For the final `unknown` class and stronger evaluation, include unrelated words from permitted source recordings; check source speaker separation carefully.
- Create a separate metadata CSV for continuous test events and report which clips are natural versus synthetic.

### Experiment ledger example

| Run | Architecture | Features | Dataset split | Tracked results |
|---|---|---|---|---|
| E01 | LSTM-64, 1 layer | MFCC-40 | isolated-v1 | macro-F1, confusion, epoch time |
| E02 | LSTM-128, 2 layers | MFCC-40 | same isolated-v1 | compare macro-F1, latency, size |
| E03 | Chosen LSTM + window detector | same features | continuous-v1 | event-F1, false alarms/hour |
| E04 | Same detector with noise augmentation | same features | same continuous-v1 | robustness difference |

These are **planned experiment IDs**, not achieved scores. Compare configurations on a common held-out set.

## 8. CPU-first performance plan

1. Begin with <10 target labels including background classes, small feature inputs, hidden width 64 or 128.
2. Cache features offline where possible; decode long recordings in manageable chunks.
3. Start with batch 16/32 and measure RAM; use early stopping and a limited epoch budget.
4. Limit concurrent analysis jobs to 1 initially to preserve UI responsiveness.
5. Instrument wall time, CPU use, process peak RAM and model size for every benchmark.
6. Profile on the **actual** Ryzen 5 5500U; do not invent timing estimates in the report.
7. Treat Colab or an optional discrete-GPU system as an acceleration option, **not a requirement**.

## 9. Privacy, local actions, and operational checklist

- [ ] All real speech recording stays on device unless the user explicitly exports/uploads it.
- [ ] Uploaded WAV and detected events can be deleted through the UI.
- [ ] Live microphone turns off when the user stops, closes the page or disconnects.
- [ ] The backend binds to `127.0.0.1`; validate browser origins and uploaded data.
- [ ] Recognized words trigger only named, pre-authorized actions.
- [ ] No shell execution from microphone-derived text.
- [ ] Logs omit raw audio and sensitive content.
- [ ] A dataset inventory contains source, license, consent and retention notes.

## 10. Final demonstration script

1. **Open dashboard:** show active model identity, supported labels and readiness.
2. **Upload labeled example WAV:** process it and show actual `stop` / `go` detection markers at approximate times.
3. **Click event:** jump to the corresponding spoken segment and replay it.
4. **Open Voice Control:** grant microphone, start listening, say `right`, `left`, `go`, `stop`, and show UI actions.
5. **Test rejection:** say an unrelated word, remain silent, or introduce moderate background sound and inspect whether actions are suppressed.
6. **Open Evaluation:** show true model results, learning curves, confusion matrix, event precision/recall/F1 and measured runtime.
7. **Explain limitations:** limited trained vocabulary, imperfect localization, errors in noise and why sliding-window detection differs from full ASR.

## 11. Acceptance and release checklist

- [ ] Every required PRD P0 story has a passing demo/test or an explicit failed acceptance report.
- [ ] The exact tested model checkpoint and feature pipeline are available.
- [ ] Demo recordings and keywords are contained in supported trained labels.
- [ ] Reproducible metrics and ground-truth sources are provided.
- [ ] The project runs from a clean install on the CPU-only target laptop.
- [ ] No critical data-loss, arbitrary-command or microphone-lifecycle issue remains.
- [ ] Repository README points to all four documents and states known limitations.

## 12. Optional future work (do not start before P0)

- Collect data for human-friendly phrases such as `next`, `pause`, `volume up`.
- Compare LSTM, BiLSTM and CNN-LSTM under identical splits/compute budgets.
- Try an LSTM+CTC sequence detector if current natural-speech localization is inadequate.
- Add explicit Windows system volume and targeted PowerPoint control with permission and allowlisting.
- Add better noise robustness, more accents and improved hard-negative collection.
- Package the local frontend/backend into a Windows installer only after it is validated.

---

**Implementation rule:** If we can demonstrate the trained LSTM, real detection results, bounded voice actions, and honestly measured performance on the user's CPU-only laptop, the academic project succeeds as a well-defined system. Advanced capabilities are justified by evidence, not by making the feature list longer.
