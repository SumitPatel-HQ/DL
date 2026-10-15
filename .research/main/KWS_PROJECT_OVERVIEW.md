# Project Overview — LSTM-Based Keyword Spotting for Continuous Audio Analysis and Voice Control

> Status: Proposed course-project specification · Version 1.0 · 15 October 2026  
> Platform: Local-first, Windows-oriented · Target hardware: Ryzen 5 5500U, 16 GB RAM, integrated AMD Radeon graphics  
> Companion documents: [PRD](KWS_PRD.md) · [SRD](KWS_SRD.md) · [Implementation Plan](KWS_IMPLEMENTATION_PLAN.md)

## 1. Project at a glance

We are building a **local speech-keyword detection application** using a **self-trained Long Short-Term Memory (LSTM) network**. It will support two related workflows:

1. **Analyze recorded speech:** upload a recording, search for a **predefined trained vocabulary** of keywords, display approximate time ranges for detections, and jump to the relevant audio segment.
2. **Use short voice commands:** listen through a microphone, detect supported words, and trigger explicitly mapped actions such as next/previous slide, play/pause, and volume up/down.

A third part, the **Model Evaluation Dashboard**, will expose evidence of model performance: training curves, per-class confusion, precision, recall, F1, false detections and measured inference timing. It must distinguish real test-set metrics from a prediction's confidence score.

**This is keyword spotting, not a general-purpose speech-to-text assistant.** Only trained and validated keywords can be recognized. Natural-language understanding, arbitrary voice instructions, and speech transcription are not part of the initial project.

## 2. Problem statement

Many simple speech-command demonstrations classify isolated, clean, one-second audio samples. That does not adequately demonstrate the challenges of finding spoken keywords in longer recordings or acting on commands amid silence, noise and unrelated speech. Such applications may also report only overall accuracy, without exposing false positives or timing.

The proposed project addresses a **bounded practical problem**: make a small, locally runnable keyword spotter that can serve *both* recorded-audio search and responsive command detection, while showing measurable evidence of how reliably it works.

### Users and use cases

- **Student/researcher:** trains the LSTM, compares experiments, reviews errors and demonstrates the underlying DL concepts.
- **Audio reviewer:** searches an uploaded meeting, lecture or sample recording for an already-supported keyword and plays the corresponding segment.
- **Demo user:** says a trained command such as `right`, `left`, `go`, `stop`, `up` or `down`, and sees a mapped action occur in the application.

The longer-term use cases are not claims about the first version. A limited vocabulary, clear limitations and repeatable benchmarks are part of the design.

## 3. Proposed solution

```text
                  WAV upload                 Microphone
                      |                          |
                      +----------+---------------+
                                 |
                 Audio validation + resampling
                        16 kHz mono PCM
                                 |
                Windowing + MFCC/Log-Mel features
                                 |
                       PyTorch LSTM model
                                 |
                   Per-window keyword scores
                                 |
                  Threshold + temporal smoothing
                                 |
                 +---------------+----------------+
                 |                                |
       Recording analysis                 Live command mode
       detections + times                 allowlisted actions
                 |                                |
                 +---------------+----------------+
                                 |
                     Dashboard + SQLite
                 predictions, events, metrics
```

### Four-step core working principle

1. **Audio input:** accept WAV recordings or microphone speech.
2. **Feature extraction:** resample, normalize, divide into windows, compute MFCC or Log-Mel features.
3. **Keyword detection:** run a trained LSTM and decide whether a known keyword is present.
4. **Output:** highlight detection intervals in the recording **or**, in explicit command mode, execute a safe mapped action.

### Why an LSTM?

Speech is time-dependent. An LSTM processes a sequence of audio feature frames, allowing the model to learn temporal characteristics of short spoken words. The course directly covers recurrent neural networks, LSTM, activations, training, loss functions, optimization, and regularization. A single LSTM does **not** inherently solve word localization in arbitrary continuous speech; this requires a detection strategy and empirical evaluation.

## 4. Intended user experience

### A. Recorded-audio analysis

1. Open **Audio Analysis** and upload a supported WAV file.
2. Choose one or more keywords from the deployed model's supported vocabulary.
3. Press **Analyze** and view processing status.
4. See a waveform with highlighted approximate detection spans.
5. Click a detection to seek and play around that timestamp.
6. Review the keyword label, confidence score and analysis duration. Export a CSV of detections.

Example: a recording contains `stop` at approximately 00:22 and `go` at 00:54; both appear as markers. Words such as `deadline` or `urgent` will **not** work until corresponding training data and model support are added.

### B. Live voice-control demo

1. Open **Voice Control**, grant microphone permission, and explicitly start listening.
2. Speak one of the supported keywords.
3. If the model score and stability rules pass the configured threshold, the application dispatches a predefined action.
4. The user sees `heard → accepted/rejected → action`, with a cooldown to avoid repeated triggers.
5. Stop listening at any time.

**Default command-to-action map for the first working demo:**

| Recognized word | Action | Notes |
|---|---|---|
| `right` | Next slide in the application's sample presentation | Safe in-app action |
| `left` | Previous slide | Safe in-app action |
| `go` | Play an in-app audio/video sample | Safe in-app action |
| `stop` | Pause the sample | Safe in-app action |
| `up` | Increase demo-player volume | In-app initially |
| `down` | Decrease demo-player volume | In-app initially |

This mapping demonstrates voice control **without pretending that Speech Commands already contains `next`, `pause`, or `volume up` recordings**. Those exact phrases can be supported later with newly collected/appropriately sourced labeled data. Windows system-wide volume or PowerPoint integration is a later, permission-gated enhancement.

### C. Model evaluation

The dashboard presents model metadata, labeled evaluation runs, a confusion matrix, class-level precision/recall/F1, training/validation loss, false alarms per hour and inference latency. Metrics are calculated from labeled test data. An uploaded recording with no annotations shows **detections and confidence**, not a fabricated “accuracy” percentage.

## 5. Core features and boundaries

| Area | Course-project MVP | Later expansion |
|---|---|---|
| Dataset | A small subset of Speech Commands; properly separated train/validation/test | Larger corpus, more accents and keywords |
| Model | Compact MFCC → LSTM → dense classifier | BiLSTM, CNN-LSTM, CTC/sequence detector comparisons |
| Recording mode | Short WAV files, selected trained words, approximate timestamp ranges | Longer batch recordings and finer localization |
| Live mode | User-activated microphone stream, small trained command set | Always-on/low-power background operation |
| Actions | In-app slides/player | Opt-in, allowlisted Windows actions |
| Evaluation | Classification + event-level detection benchmarks | More extensive robustness and cross-device tests |
| Data storage | SQLite and local files | Multi-user storage only if truly needed |

### Explicit non-goals for MVP

- Full transcription, arbitrary-language speech understanding, or untrained custom words.
- Cloud model/API requirements, proprietary speech services, LLMs or speech assistant agents.
- Automatic control of arbitrary OS applications or execution of user-spoken shell commands.
- Guaranteed performance in emergencies, noisy industrial settings, or multiple simultaneous speakers.
- Training a large speech foundation model.

## 6. Technology stack

| Layer | Technologies | Reason |
|---|---|---|
| UI | **Next.js, TypeScript** | Dashboard, uploads, controls and UI routes |
| UI system | **Tailwind CSS, shadcn/ui** | Consistent reusable components |
| Visualization | **WaveSurfer.js** (+ Regions), **Recharts** | Audio playback/markers and ML charts |
| Backend API | **FastAPI**, Pydantic, WebSockets | Local HTTP API and optional streaming results |
| DL | **PyTorch** (`torch.nn.LSTM`) | Train/evaluate the model with CPU support |
| Audio/features | **librosa, SoundFile, NumPy** | Decode WAV, resample and calculate MFCC/Log-Mel |
| Training metrics | **scikit-learn**, pandas (optional) | Reports, confusion matrix, metrics |
| Live capture | Browser `getUserMedia`/AudioWorklet | Microphone input into local UI/backend |
| Persistence | **SQLite** with SQLAlchemy or `sqlite3` | Recordings, detection records and evaluation metadata |
| Checkpoints | Local `.pt` weights + JSON metadata | Reproducible model loading |
| Testing | **pytest, pytest-asyncio**, Playwright (selected flows) | API, ML and UI validation |
| Optional Windows integration | `pycaw` or `pywin32` in local Python backend | Only for later approved Windows actions |

No GPU-specific library is required. The baseline will run on CPU; processing and training times must be measured on the target machine.

## 7. Machine-learning plan in brief

- **Starter vocabulary:** `left`, `right`, `up`, `down`, `go`, `stop`, with explicit `unknown` and `silence/noise` handling.
- **Dataset:** Speech Commands is suitable for *isolated-word training*. Maintain speaker-disjoint splits. Additional **annotated continuous recordings** (including negatives) are required for credible continuous-word evaluation.
- **Features:** 16 kHz mono audio, 25 ms analysis frames, 10 ms hop, ~40 MFCC coefficients (initial hypothesis; tune on validation).
- **Baseline:** 1–2 layer LSTM (hidden size 64 or 128), dropout where applicable, linear classification head, cross-entropy loss, Adam, early stopping.
- **Continuous inference experiment:** move ~1-second overlapping windows over audio, score each window, smooth/merge consecutive predictions, return approximate intervals.
- **Critical validation gate:** test on *natural connected speech* with human-labeled keyword occurrences. Do not label the feature “reliable continuous KWS” merely because it works on concatenated isolated words.
- **If the baseline fails:** document the failure and evaluate a sequence-learning alternative (e.g., LSTM+CTC) as a **separate research extension**, not a silent architecture substitution.

## 8. Academic deliverables

- Reproducible training and inference code, preprocessing configuration and saved checkpoint.
- Proper data splits and an evaluation set with labeled continuous recordings.
- Model performance report: per-class classification metrics, event precision/recall/F1, false alarms per hour, latency, CPU/RAM usage.
- Working local app: recorded-audio detection, live command demonstration and dashboard.
- Short ablation: at least two LSTM configurations or LSTM versus a simple baseline, with identical splits.
- README explaining reproduction, limitations, dataset attribution and demo steps.

## 9. Main risks and mitigations

| Risk | Mitigation |
|---|---|
| Isolated-word training performs poorly in conversations | Test in Phase 2; collect labeled continuous clips; reconsider detection model if necessary |
| False command triggers | User-activated mode, class thresholds, smoothing, cooldown, unknown/noise examples |
| Ryzen 5500U CPU training is slow | Small vocabulary, small LSTM, cached features, incremental experiments, early stopping |
| Misleading metrics | Separate confidence from labeled ground-truth metrics and identify data splits |
| Command integration harms user workflow | Safe in-app default actions, explicit permissions, no arbitrary command execution |
| Source recordings contain private speech | Local storage, deletion control, no uploading by default, minimal logging |

## 10. References and background

These links support technical choices; the target features and success criteria above are **proposals**, not performance claims.

- [Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition](https://arxiv.org/abs/1804.03209)
- [TensorFlow simple audio recognition tutorial](https://www.tensorflow.org/tutorials/audio/simple_audio) — example isolated keyword data and workflow
- [PyTorch LSTM reference](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [FastAPI WebSocket documentation](https://fastapi.tiangolo.com/advanced/websockets/)
- [WaveSurfer.js and Regions plugin](https://wavesurfer.xyz/)

---

**Project identity:** This is an **LSTM keyword-spotting system** with two applications (audio search + bounded voice control) and one evidence-driven analytics dashboard. It is not a general voice assistant.
