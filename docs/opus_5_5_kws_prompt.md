# Role & Task
You are an expert Machine Learning Engineer specializing in Speech Processing, Keyword Spotting (KWS), and Audio Event Detection.
Analyze the following deep learning codebase context, diagnose the severe false-positive flooding occurring in the application, and provide production-ready solutions and drop-in code fixes.

---

## 1. User Problem Statement & Observed Behavior
In our Keyword Spotting web workbench (`/analyze`), when analyzing continuous audio recordings (5 to 10-minute conversational speech from the AMI Meeting Corpus):
1. **Full-Timeline Flooding**: When selecting keywords (such as `left` and `right`), the system marks virtually the entire 10-minute audio as containing these keywords, even though they only appear a few times in reality.
2. **False-Positive Saturation**: The model constantly produces false-positive detections across the audio stream.
3. **Pill & Filter Breakage**: Because of the flood of detections, all keyword count pills and waveform lanes become saturated with false alarms.
4. **1-Second Window Duration Mismatch**: In continuous speech, a spoken keyword lasts only 150–300 ms, but the detector evaluates fixed 1.0-second sliding windows, causing severe boundary and recognition mismatches.

---

## 2. Technical Context & Current Codebase Implementation

### A. Audio & Feature Pipeline (`backend/ml/features/pipeline.py`)
- **Audio Format**: Canonical 16 kHz mono 16-bit integer PCM WAV, loaded as `float32` in `[-1.0, 1.0]`.
- **Spectrogram / MFCC Specs**:
  - Sample rate: 16,000 Hz
  - Window: 25 ms (400 samples)
  - Hop: 10 ms (160 samples)
  - Mel filterbanks: 40 bands (20 Hz to 7,600 Hz)
  - MFCC coefficients: 40
  - Normalization: fixed per-coefficient mean/std computed on the isolated training split (`Normalization` in `backend/ml/features/pipeline.py`). There is no per-utterance or gain normalization, so recording level shifts the features.

### B. Models (`E02`, checkpoint `e47e49f5`; active service model `LIVE1`, checkpoint `669693a9`)
LIVE1 (`backend/ml/train_live.py`) is the same network retrained with gain/noise/shift augmentation and TTS words; it has the same failure on fluent speech. Measured on the AMI meetings, every checkpoint scores real Keyword windows no higher than background speech (ranking AUC 0.35-0.61), so detector settings can reduce false alarms but cannot recover recall.
- **Architecture** (`backend/ml/networks`):
  - 2-layer Bidirectional/Unidirectional LSTM, hidden dimension 128, linear classification head.
  - Vocabulary: `["left", "right", "up", "down", "go", "stop", "unknown", "silence"]`.
- **Training Data**:
  - Trained exclusively on **isolated single-word clips** (Google Speech Commands subset), where each word is roughly centered in a 1-second crop.
  - The `unknown` class was trained on isolated non-target words; `silence` was trained on background noise clips. **The model was never trained on fluent conversational speech.**

### C. Current Detector Logic (`backend/ml/spotting.py`)
```python
@dataclass(frozen=True)
class DetectorConfig:
    # Code defaults are 0.25 / 0.5 / 2; the service loads the detector JSON, which
    # used stride 0.2, threshold 0.5, min_consecutive 1 for both E02 and LIVE1.
    stride_s: float = 0.25
    threshold: float = 0.5
    min_consecutive: int = 2
    suppression_s: float = 1.0
    window_s: float = 1.0

def detect(
    scores: WindowScores,
    config: DetectorConfig,
    keywords: tuple[str, ...],
    labels: tuple[str, ...],
) -> list[Detection]:
    top = scores.probabilities.argmax(axis=1)
    top_score = scores.probabilities.max(axis=1)
    wanted = [labels.index(keyword) for keyword in keywords]
    votes = np.where(np.isin(top, wanted) & (top_score >= config.threshold), top, -1)
    candidates = []
    first = 0
    for i in range(1, len(votes) + 1):
        if i < len(votes) and votes[i] == votes[first]:
            continue
        if votes[first] >= 0 and i - first >= config.min_consecutive:
            candidates.append(
                Detection(
                    labels[votes[first]],
                    float(scores.starts_s[first]),
                    float(scores.starts_s[i - 1] + config.window_s),
                    float(top_score[first:i].max()),
                    config.threshold,
                    i - first,
                )
            )
        first = i
    return suppress_duplicates(candidates, config.suppression_s)
```

### D. Architectural Findings Already Documented in Repo (`ADR-0006`, `continuous-v1/README.md`)
- On a 10-minute held-out continuous speech test recording (`IN1002`, 600s), the detector produced:
  - **367 false detections and only 2 true hits** (Event-F1 = 0.010, False Alarms/Hour = 2,216).
  - 161 of 367 false detections scored confidence $\ge 0.85$ (many $> 0.99$).
  - Simple threshold tuning failed: increasing the threshold to 0.85 or 0.90 did not separate true hits from false positives because the softmax layer is overconfident on out-of-distribution phoneme sequences.
  - Adjacent-window merging (`min_consecutive = 1`) and `suppression_s = 1.0` cause detections to chain across several consecutive seconds.

---

## 3. Tasks & Required Deliverables

Please provide a detailed, highly technical response structured into the following sections:

### 1. In-Depth Root Cause Diagnosis
- Mathematically explain why an isolated-word LSTM outputs high-confidence softmax predictions on continuous speech transitions.
- Explain the temporal resolution mismatch: why a 1.0s window with 0.20s stride causes words of 200 ms duration to be either drowned out by `unknown` speech or falsely detected.
- Explain why the current `suppress_duplicates` and `min_consecutive` algorithms exacerbate timeline flooding.

### 2. Immediate Code-Level Fixes (No Retraining Required)
Provide drop-in algorithmic improvements to `backend/ml/spotting.py` that can immediately suppress false alarms using the existing model weights:
- **Voice Activity / Energy Pre-Gating (VAD)**: Calculate short-term energy/RMS to drop silent/low-energy windows before running inference.
- **Confidence Margin & Entropy Filter**: Replace raw `top_score >= threshold` with:
  1. Margin test: $P(\text{keyword}) - P(\text{unknown}) \ge \Delta$
  2. Shannon entropy threshold: Reject predictions where the softmax distribution has high entropy or lacks a distinct peak.
- **Stricter Temporal Voting & Persistence**: Change window voting to require genuine peak-like temporal activation rather than any single window above threshold (`min_consecutive >= 2` or `3`, with minimum hop duration).
- **Non-Maximum Suppression (NMS)**: Implement center-distance and temporal-overlap IoU suppression across competing keyword classes.

### 3. Production Python Implementation
Provide the complete Python code for an upgraded, robust detection module (`spotting_robust.py`) compatible with the existing `LoadedModel`, `Audio`, and `Detection` dataclasses. The code must:
- Include RMS energy gating.
- Include margin testing against `unknown` and `silence`.
- Include entropy / peak verification.
- Output the exact `Detection` dataclass shape expected by the backend service:
  `(keyword: str, start_s: float, end_s: float, confidence: float, threshold: float, windows: int)`.

### 4. Medium-Term Model Transition Strategy
- How to transition from 1.0s sliding windows to a frame-level sequence model (e.g., using CTC loss or the prototype BiLSTM frame classifier in `backend/ml/temporal.py`).
- How to construct hard negative mining for continuous speech using unannotated sections of the AMI corpus.
