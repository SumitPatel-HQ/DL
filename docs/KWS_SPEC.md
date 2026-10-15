# P0 Spec — LSTM Keyword Spotting (Audio Analysis + Voice Control + Evaluation)

Source: `.research/main/KWS_PROJECT_OVERVIEW.md`, `KWS_PRD.md`, `KWS_SRD.md`, `KWS_IMPLEMENTATION_PLAN.md` + grill Q1–Q8 (approved). Glossary: `CONTEXT.md`. Decisions: `docs/adr/0001`–`0005`. P1/P2 remain roadmap-only per `KWS_PRD.md`, not built in P0.

## Problem Statement

Students and demo users have only isolated one-second keyword classifiers with overall accuracy. They cannot search a recording for a spoken Keyword and jump to it, trigger a bounded demo action by voice without false actions, or see honest evidence of when the model fails in continuous speech, noise, and Silence.

## Solution

A local-first app with one trained LSTM classifier reused in two modes: (a) upload a WAV Recording, run sliding-window inference, show Detections with approximate times on a waveform for seek/play; (b) explicit mic Live session mapping accepted Detections to in-app actions with threshold, smoothing, and cooldown. Every number comes from a persisted Evaluation Run over labeled data; unlabeled audio shows Confidence only, never fabricated accuracy.

## User Stories

1. As an audio reviewer, I want to upload a PCM WAV Recording, so that I can analyze local speech without cloud upload.
2. As an audio reviewer, I want invalid, corrupt, or oversize files rejected with an actionable error, so that I know what to fix.
3. As an audio reviewer, I want to see filename, duration, and supported Keywords for the active model, so that I only search for trained words.
4. As an audio reviewer, I want to select one or more Keywords and start analysis, so that I control what is detected.
5. As an audio reviewer, I want each Detection to show Keyword, start_s, end_s, and Confidence with threshold metadata, so that I can judge approximate locations.
6. As an audio reviewer, I want Detections highlighted as regions on a waveform with timeline, so that I can inspect approximate spans.
7. As an audio reviewer, I want to click a Detection row or region to seek and play around that timestamp, so that I can verify by listening.
8. As an audio reviewer, I want an explicit empty state with zero Detections rather than a guessed label, so that I trust negative results.
9. As an audio reviewer, I want to delete a Recording with its audio bytes and child Detections, so that private speech stays controllable.
10. As a voice-control user, I want to explicitly Start/Stop listening with clear permission and status, so that the mic never runs in the background.
11. As a voice-control user, I want `right` to advance and `left` to go back in the in-app sample presentation, so that I can demo hands-free slides.
12. As a voice-control user, I want `go` to play and `stop` to pause the in-app sample, so that transport is voice-driven.
13. As a voice-control user, I want `up`/`down` to change demo-player volume in-app, so that I can show graded control safely.
14. As a voice-control user, I want uncertain, Unknown, or Silence input to be rejected with visible accepted/rejected state, so that accidents are suppressed.
15. As a voice-control user, I want one utterance to cause at most one action via smoothing and cooldown, so that repeats do not double-trigger.
16. As a voice-control user, I want to see last Keyword, Confidence, decision, and action result with history, so that behavior is auditable.
17. As a student, I want training and validation loss/accuracy curves from real saved logs, so that I can explain learning and overfitting.
18. As a student, I want a confusion matrix with per-class precision/recall/F1 including Unknown and Silence, so that errors are visible beyond averages.
19. As a student, I want isolated classification metrics separated from continuous-event precision/recall/F1 and false alarms/hour, so that sliding-window limits are honest.
20. As a student, I want every chart traceable to checkpoint hash, dataset/split version, config, and timestamp, so that results reproduce.
21. As a student, I want `Not evaluated` when no valid Evaluation Run exists, so that missing evidence is explicit.
22. As a student, I want model readiness, missing checkpoint, backend disconnect, and mic denial shown as distinct errors, so that setup failures are recoverable.
23. As a course evaluator, I want a clean CPU-only install on Ryzen 5 5500U/16GB with measured training/inference time and peak RAM, so that feasibility is proven without GPU.
24. As a course evaluator, I want at least one held-out isolated Evaluation Run and one natural continuous-event Evaluation Run from disjoint speakers, so that transfer is tested.

## Implementation Decisions

Seams (highest possible, fewest new): (1) versioned feature pipeline shared by train/batch/live; (2) Keyword Detection service behind REST plus WebSocket live session; (3) SQLite persistence for models, recordings, Detections, Evaluation Runs, live events. New code prefers these three seams; no direct UI-to-model or mic-text-to-shell paths.

- P0 vocabulary is 8-class `left/right/up/down/go/stop/unknown/Silence`, configurable label list for later words; model never advertises untrained words. Per ADR-0001.
- Canonical audio is 16kHz mono float32; baseline features 25ms frame, 10ms hop, 40 MFCC, versioned with normalization, class order, and checkpoint hash. Identical pipeline for training, batch, and live with a parity test on the same WAV fixture. Per Q5.
- Baseline E01 is 1-layer LSTM hidden-64, then E02 2-layer hidden-128 on identical speaker-disjoint splits; compare macro-F1, params, size, CPU latency. Cross-entropy plus Adam plus early stopping with fixed seeds; dropout outside single-layer LSTM; batch 16/32; cached features. No BiLSTM/CNN/CTC in P0. Per Q6 and ADR-0001.
- Continuous Detections reuse the isolated model over ~1.0s windows, ~0.2–0.25s stride, with validation-tuned threshold, adjacent-window consistency, merging, and duplicate suppression. Window/stride/thresholds and ±0.5s center-tolerance match rule are parameters frozen before test, not guarantees. Go/no-go gate after held-out natural test: continue, one bounded retune, or separate CTC prototype. Per ADR-0002.
- Live path is browser `getUserMedia`/AudioWorklet mono PCM_s16le over versioned WebSocket to loopback FastAPI, backend owns resampling and bounded ring buffer, pushes Detection plus action events, closes on Stop/disconnect/backpressure. No raw mic retained by default. Per ADR-0003.
- Actions are strict allowlist to in-app functions with cooldown (~1.2s initial, validation-tuned) and rejection logging. No shell, arbitrary app control, or OS volume/PowerPoint in P0. Per ADR-0004.
- Backend exposes health plus active model, recording upload/analyze/status/Detections/audio/delete, evaluations list/get, and command map; one bounded analysis worker with queued/running/completed/failed states; loopback bind with origin and size/rate limits. Playback supports range if waveform needs it.
- Persistence uses SQLite tables for models, recordings, Detections, Evaluation Runs, live events with foreign-key cascade on recording delete; audio and checkpoints as local files; logs omit speech content.
- Targets are hypotheses: isolated macro-F1 ≥0.85, event-F1 ≥0.70 in-domain, ≤1 false action per 30min negatives, median end-of-utterance to action ≤1s, reproducible reruns, no OOM on 16GB. Unmet targets are reported, never silently redefined.
- Phase-0 scaffold follows Plan layout with backend app plus ML configs/data/features/networks/train/evaluate/infer/spotting, frontend routes/components/lib, artifacts, scripts, gitignored data; Ruff/format, pytest plus TS lint, CPU torch smoke, `/api/health` model-unavailable until checkpoint installed.

## Testing Decisions

Good tests assert external behavior at seams, not internals: API status codes and JSON contracts, event times within tolerance, action allowlist enforcement, metric math on fixtures, and UI seek/play/error states. Deterministic fixtures with fixed seeds; no test fits normalization on test data or tunes on test splits.

- Preprocessing: resample, mono mix, Silence, NaN/Inf, MFCC shape, train/inference parity.
- ML: checkpoint config compatibility, logits shape, deterministic reload fixture.
- Event detector: overlap merging, tolerance matching, duplicate/cooldown suppression.
- Metrics: known TP/FP/FN to F1, confusion matrix, false alarms/hour math.
- Backend integration: upload → analyze → poll → retrieve Detections → delete cascade; corrupt/oversize/unsupported-Keyword/missing-checkpoint/no-Detection errors.
- Streaming: handshake, wrong rate/channels, disconnect/backpressure cleanup, no lingering actions.
- Actions: valid words map, Unknown/low-Confidence rejected, injection with executable path or command rejected.
- Frontend: WAV upload to marker seek, mic start/stop, empty and error states, delete updates dashboard.
- Performance/validity: CPU epoch/inference time, peak RAM, end-to-utterance latency, held-out natural-event evaluation plus controlled negatives.

## Out of Scope

P1 and P2 per `KWS_PRD.md` stay roadmap-only, not built in P0: CSV export, clean/noisy comparison view, opt-in Windows volume via pycaw/pywin32, multi-variant model comparison UI, new vocab (`next/pause/urgent`), LSTM+CTC localizer beyond the gate prototype decision, external PowerPoint/media automation, multi-format transcoding beyond WAV PCM, hosted multi-tenant service, Windows installer packaging.

## Further Notes

- Continuous set needs 10–20min natural annotated speech with positives, hard negatives, Silence/noise, speaker/source-disjoint splits; synthetic concatenations are debug-only.
- `Unknown` requires unrelated words from a larger permitted source; mini-8 alone is insufficient.
- Recording limit up to 10min configurable but must validate on target hardware; fail early otherwise.
- Keep P0 slices small with executable gates per phase; never merge fake inference to unblock UI — use clearly labeled fixtures in tests only.
- Ticket backlog T01–T15 and phased gates in Implementation Plan remain the execution order; this spec freezes scope, not estimates.
