# Product Requirements Document (PRD)

**Project:** LSTM-Based Keyword Spotting for Continuous Audio Analysis and Voice Control  
**Version:** 1.0 (proposed) · **Date:** 15 October 2026 · **Status:** Definition / not yet implemented  
**Related:** [Overview](KWS_PROJECT_OVERVIEW.md) · [SRD](KWS_SRD.md) · [Implementation Plan](KWS_IMPLEMENTATION_PLAN.md)

## 1. Product summary

A local-first application that detects **a known set of spoken keywords** from recorded audio and live microphone input using an LSTM-based classifier. Users can inspect detections on an interactive waveform, trigger limited voice-controlled actions and evaluate the deep-learning model through a transparent dashboard.

### Problem to solve

A basic keyword classifier only returns a label for a short audio clip. It does not offer usable navigation of recordings, live command execution, or sufficient evidence about errors and performance. The project will turn trained KWS into a useful, testable application while keeping the DL model itself central.

### Target audience

1. **Course evaluators/students** — inspect ML methodology, model performance and comparisons.
2. **Audio reviewer** — search an audio clip for trained keywords and jump to occurrences.
3. **Voice-control user** — operate a contained presentation/media demo without using hands.

### Product principles

- **Model-first:** real PyTorch inference, not a stub or external transcription API.
- **Offline by default:** no speech uploads to a remote model service.
- **Honest output:** only supported keywords; confidence is not ground-truth accuracy.
- **Safe commands:** a short, visible, allowlisted command-to-action map.
- **Local CPU feasibility:** a small model and short recordings initially.

## 2. Product objectives and measures

| Objective | Measurable evidence | Proposed acceptance target* |
|---|---|---|
| Recognize a selected small keyword set | Isolated-word holdout test; macro-F1 and confusion matrix | Macro-F1 ≥ 0.85 |
| Find keywords in recorded speech | Independently annotated continuous test clips; event precision/recall/F1 | Event F1 ≥ 0.70 on a documented in-domain test set |
| Reduce accidental actions | Negative microphone sessions and false-action counts | No more than 1 false action per 30 minutes of controlled negative testing |
| Make voice commands responsive | End-of-utterance to UI-action time | Median ≤ 1.0 seconds on target laptop |
| Show reproducible evaluation | Repeat the same evaluation with checkpoint, split and config | Same saved artifacts reproduce reported metrics within expected numerical tolerance |
| Support local CPU-first workflow | Measured peak RAM, CPU use and training duration | No GPU dependency; no out-of-memory failure in standard run |

*Targets are **engineering hypotheses**, not results or promises. If continuous-speech detection fails, report the result and the gating decision. Acceptance thresholds may be revised after a documented baseline with faculty approval; never silently alter the definition of success.

## 3. User journeys

### Journey A — Analyze a recording

1. User opens **Audio Analysis**.
2. User uploads a WAV file and sees filename, duration and supported keywords.
3. User selects keywords and starts analysis.
4. Application displays progress/status and returns detection candidates.
5. User sees highlighted waveform ranges and a searchable detections table.
6. Clicking a row seeks to the time range and starts playback.
7. User exports detections to CSV or deletes the recording and its outputs.

**Acceptance:** for a known labeled demonstration clip, the displayed detections use real model outputs, expose timestamps and link to playable regions. Errors are shown rather than substituted with invented matches.

### Journey B — Control the demo by speaking

1. User opens **Voice Control** and grants microphone permission.
2. User explicitly clicks **Start listening** (no background microphone capture on app launch).
3. User says a supported keyword such as `right`.
4. The model evaluates incoming speech and emits a candidate event.
5. The decision layer accepts or rejects it according to threshold/cooldown rules.
6. If accepted, the controlled in-app presentation advances once.
7. User sees the recognized keyword, decision, confidence and action status.
8. User can stop listening immediately.

**Acceptance:** the app reliably maps supported, validated words to the documented in-app actions in controlled test cases, reports rejected/unrecognized speech, and does not execute arbitrary system commands.

### Journey C — Inspect model performance

1. User opens **Model Evaluation**.
2. User selects a saved model checkpoint and labeled evaluation run.
3. Dashboard displays class-level classification metrics and event-level detection metrics separately.
4. User explores training/validation loss charts, confusion matrix and false alarms.
5. Model settings, split, dataset version and evaluation timestamp are visible.

**Acceptance:** every number is computed from a persisted, identifiable labeled dataset and run; pages show `Not evaluated` when metrics do not exist.

## 4. Functional requirements (prioritized)

**Priority legend:** P0 = required for first complete course deliverable; P1 = desirable after P0; P2 = future expansion.

| ID | Priority | Requirement | Acceptance criteria |
|---|---|---|---|
| PR-01 | P0 | Upload local WAV audio | Reject invalid/oversize files with an actionable error; duration/format displayed |
| PR-02 | P0 | Normalize audio consistently | Both inference modes use versioned 16 kHz mono feature pipeline |
| PR-03 | P0 | Select supported keywords | UI lists only labels in active checkpoint; unsupported terms disabled |
| PR-04 | P0 | Analyze recordings with LSTM | Every detection includes keyword, start/end seconds and score |
| PR-05 | P0 | Visualize detections on waveform | Clicking a marker seeks to corresponding segment |
| PR-06 | P0 | Start/stop live microphone detection | Permission is requested, listening status clear, stream stops on user request |
| PR-07 | P0 | Execute in-app command actions | `right/left/go/stop/up/down` map to documented in-app actions |
| PR-08 | P0 | Debounce and reject uncertain detections | Configured threshold, smoothing and cooldown applied before any action |
| PR-09 | P0 | Expose labeled model metrics | Classification + event metrics shown with dataset/run identifiers |
| PR-10 | P0 | Display training results | Training and validation loss/accuracy plotted from real saved logs |
| PR-11 | P0 | Persist history | Recording jobs, detections, model metadata and evaluations survive refresh |
| PR-12 | P0 | Remove local data | User can delete recording + related detections and source file |
| PR-13 | P0 | Show system/model readiness | Missing model, permissions and backend connection failures are explicit |
| PR-14 | P1 | CSV export | CSV contains recording ID, label, start/end, score, model version |
| PR-15 | P1 | Noise test comparison | Dashboard can compare clean/noisy labeled evaluation sets |
| PR-16 | P1 | Optional Windows volume control | User opts in; backend safely changes volume on same Windows machine |
| PR-17 | P1 | Model comparison | Compare baseline LSTM variants under identical evaluation datasets |
| PR-18 | P2 | Train words such as `next`, `pause`, `urgent` | Requires additional licensed/consented recordings and retraining |
| PR-19 | P2 | Higher-quality continuous word localization | Evaluate LSTM+CTC or another sequence detector in a separate experiment |
| PR-20 | P2 | External PowerPoint/media automation | Explicit target and action permissions; Windows-only integration |

## 5. User interface and navigation

The UI has three primary areas with a persistent model-status indicator:

### Audio Analysis

- Upload field, selected model and supported-word chips.
- WaveSurfer waveform + timeline + colored regions.
- Job progress, detection results and sortable timestamp table.
- Playback, seek, export and delete controls.

### Voice Control

- Prominent **Start/Stop listening** control, live status and minimal level meter.
- Visible supported command list and action mappings.
- Demo presentation/media player inside the app.
- Last detection, score, accepted/rejected state and last action result.

### Model Evaluation

- Model version/architecture/config and dataset split metadata.
- Cards for macro precision, recall, F1 and event false alarms/hour.
- Training and validation curves; class confusion matrix.
- Experiment list, dataset context and notes on limitations.

**UI note:** all metric examples in mockups are placeholders until actual evaluation runs exist; the shipped UI must label demos/mock data as such or remove them.

## 6. Dataset and labeling requirements

- Start from **Google Speech Commands** (isolated one-second samples); initial commands: `left`, `right`, `up`, `down`, `go`, `stop`.
- Train a usable **unknown** class with unrelated words (may require the larger dataset or extra source audio) and **silence/noise** examples; a mini eight-class subset alone is insufficient for realistic unknown-word rejection.
- Preserve or build **speaker-disjoint** train/validation/test splits to avoid leakage.
- Produce a separate **continuous evaluation set**, comprising:
  - natural sentences containing known keywords;
  - hard negatives where similar words occur;
  - silence/music/background noise;
  - clips from speakers not in training, as feasible.
- Annotate each event with `keyword, start_time, end_time`; split continuous recordings by source/speaker before fitting or tuning thresholds.
- Document recording consent, source licensing, sample rates and annotation procedure.

Synthetic concatenations of isolated words are useful for early tests but **not sufficient** to validate real continuous-speech spotting.

## 7. Success criteria / definition of done

A P0 deliverable is complete when all the following hold:

- A fresh local clone can run model inference and serve the frontend/backend on the documented Windows development setup.
- An LSTM is trained from real labeled data and its checkpoint is loadable without retraining.
- At least one held-out isolated-word evaluation and one separately labeled continuous-event evaluation are published as reproducible reports.
- An uploaded WAV displays **actual** detections in the waveform and navigable table.
- The microphone demo recognizes the documented words and performs the corresponding **in-app** actions.
- The analytics page shows traceable results, not mock scores.
- Error handling works for no microphone, silent audio, unsupported input, missing checkpoint and empty detection results.
- The README accurately describes operating limits and unachieved metric targets.

If the continuous event-F1 target is not met, the system can still be demonstrated as an experimental detector, but the **P0 continuous-detection acceptance criterion is not met**. The report must state this directly.

## 8. Non-functional product expectations

| Area | Expectation |
|---|---|
| Performance | Operates on Ryzen 5 5500U/16 GB without CUDA; benchmarks recorded on that machine |
| Privacy | Local inference and storage; microphone disabled until explicit action |
| Reliability | No duplicate action on one utterance; errors are visible and recoverable |
| Accessibility | Keyboard controls and readable error/status messages |
| Transparency | Labeled ground truth required for precision/recall; confidence clearly labeled |
| Maintainability | ML pipeline independent from web UI and from action execution |
| Reproducibility | Versioned configs, splits, seeds, model weights and evaluation artifacts |

## 9. Assumptions, exclusions, and open decisions

### Assumptions

- Development and in-app demo are on the same Windows laptop as the Python backend.
- Internet is only needed to install packages/download datasets; running inference is offline.
- Initial prerecorded input is PCM WAV; multi-format transcoding is deferred.
- This is a local single-user course project, not a hosted multi-tenant service.

### Explicit exclusions

Full ASR, general natural-language commands, arbitrary desktop automation, cloud transcription, custom wake-word training, multi-language speech, and safety-critical alerting are not included in P0.

### Open decisions before implementation

1. Exact benchmark vocabulary and number of clips available for speaker-disjoint evaluation.
2. Minimum length and quantity of natural continuous recordings we can annotate.
3. Whether voice control is fully browser-hosted or uses a small local Windows companion for OS-level actions later.
4. Threshold/event matching tolerances and revised evidence-based success gates after the feasibility experiment.

## 10. Dependencies and delivery risks

- **Highest risk:** isolated-word LSTM may perform poorly on naturally spoken continuous audio.
- **Second:** background conversations might accidentally trigger voice actions.
- **Third:** microphone device/sample-rate behavior differs across browsers and hardware.
- **Fourth:** slow CPU training could delay iteration if the dataset/model is oversized.

Mitigate by testing the detector on annotated continuous speech early, using explicit microphone activation, building a small baseline, caching features and prioritizing error analysis over UI polish.

## 11. Traceability

The [SRD](KWS_SRD.md) gives implementation contracts for each P0 feature; the [Implementation Plan](KWS_IMPLEMENTATION_PLAN.md) schedules the proof-of-feasibility gates and tests. The [Overview](KWS_PROJECT_OVERVIEW.md) describes the product at a high level.
