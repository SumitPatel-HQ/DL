# DL Project Q&A and Explanations Record

> Complete export of the conversation history, containing every question asked and the exact detailed explanations provided about the Keyword Spotting Deep Learning project (LIVE3 model, architecture, dataset, training, MFCC features, and activations).

---

## Question 1

$unslop 

explain the project in simple way 

take basic basic quetions about the DL project and explain them how we used in the project

### Explanation / Answer 1

## What the project is

Keyword spotting app. Finds 6 trained words (`left`, `right`, `up`, `down`, `go`, `stop`) in audio. Two uses:
- Upload a WAV, get a list of where each word appears (Audio Analysis, `/analyze`).
- Speak into mic, words trigger in-app actions (Voice Control, `/live`).

Stack: Python backend (FastAPI, PyTorch) in `backend/`, Next.js frontend in `frontend/`, model weights in `artifacts/models/`.

## Basic questions

**What is keyword spotting?**
Model listens for a short list of words and ignores everything else. It does not transcribe full speech. `CONTEXT.md` defines Keyword as a trained word the model can detect.

**What does the model get as input?**
Raw audio turned into MFCC features. Audio is 16 kHz mono. Each frame is 25 ms, hop 10 ms, 40 MFCC values per frame. One 1-second window gives 98 frames. See `backend/ml/features/pipeline.py`.

**What is the model?**
LSTM, a recurrent network that reads frames in order. Mean-pools the outputs over all frames, then a linear layer gives 8 scores (6 Keywords plus `unknown` and `silence`). See `backend/ml/networks/lstm.py`.
- E01: 1 layer, hidden 64.
- E02: 2 layers, hidden 128.

**Where does training data come from?**
- Google Speech Commands v0.02 clips, 1-second word recordings. Used for isolated classification (`data/speech_commands_v0.02`, `data/speech_commands_full`).
- AMI meeting recordings, auto-annotated, for natural continuous speech (`data/continuous-ami`).
- Own TTS recordings for the live model.

**How is it trained?**
Cross-entropy loss, Adam optimizer, early stopping on validation loss, fixed seeds. Writes `model.pt`, `training.json`, `curves.csv`. Entry point: `backend/ml/train.py`. Live model uses `backend/ml/train_live.py`.

**Why train, validation, and test splits?**
- Train: fits weights.
- Validation: picks model and tunes threshold.
- Test: final score, run once.
Splits are speaker-disjoint. Same speaker never in train and test, so score shows real generalization. Rule in `README.md`: choose models on validation, not test.

**How are words found in long audio?**
Sliding window. Score each 1 s window with stride 0.2 s. Keep windows above a threshold, merge neighbors, suppress duplicates. Output is a Detection: Keyword, start, end, Confidence. Code in `backend/ml/spotting.py`.

**How is quality measured?**
- Macro-F1: average F1 over classes, so rare classes count equally.
- Per-class precision and recall, confusion matrix.
- Event-F1 for continuous audio, with ±0.5 s tolerance around each word center.
- False alarms per hour on negative audio.
Every result is saved as an Evaluation Run tied to checkpoint, split, and config.

**How does voice control avoid false actions?**
- Model must clear a threshold (live3 uses 0.7), a margin between top and second class (0.5), and a minimum audio level.
- Smoothing and cooldown stop one word causing two actions. Cooldown blocks only repeats of the same Keyword.
- Actions are an allowlist of in-app functions. No shell, no arbitrary app control.
- Rejections (Unknown, Silence, low score) show on screen with reason.
ADR: `docs/adr/0004-in-app-actions-only.md`.

**How does mic audio reach the model?**
Local: browser captures mic, sends PCM over WebSocket `/api/live` to FastAPI. Deployed on Vercel: model runs in browser with ONNX (`backend/deploy/model.onnx`), mic audio never leaves page. ADR: `docs/adr/0003-local-websocket-mic.md`.

**Optional Windows actions?**
Volume up/down and slide next/previous through Windows APIs. Off by default, opt-in per session. ADR: `docs/adr/0009-experimental-opt-in-os-actions.md`.

**Did it work?**
Partly. Honest results from the docs:
- Isolated test macro-F1: E02 0.740, target 0.85. Live3 0.797 on shared test split.
- Continuous event-F1: 0.010, target 0.70, with 2,216 false alarms/hour. Continuous spotting marked no-go (ADR-0006).
- Live3 on user's own mic: 6 of 6 keywords in one test, but short words like `up` inside sentences are missed.
Unmet targets are reported, not changed. Full report: `docs/evaluations/validation-report.md`.

**How to run it?**
From repo root:
```
npm run setup
npm run api
npm run web
```
Open http://127.0.0.1:3000. Demo script: `docs/demo.md`.

---

## Question 2

give this explain them and how and why they are used 
2. optimizer used 
3. about the dataset 
4. loss fuction 
5. training funtion 
6. explain on the main core feeatue how it used and why

### Explanation / Answer 2

Item 1 is missing from your list. I cover the model briefly first, since everything else depends on it.

## 0. Model (LSTM)

- Input: 98 frames × 40 MFCC values per 1 s window.
- LSTM reads frames in order. Mean over all frame outputs. Linear layer → 8 scores (`left right up down go stop unknown silence`).
- Why LSTM: audio is a sequence. LSTM keeps state across frames, so it sees how sound changes over time. Small enough for CPU-only training (course requirement).
- Code: `backend/ml/networks/lstm.py`.
- Configs: E01 = 1 layer, hidden 64. E02 = 2 layers, hidden 128, dropout 0.2. Live model = 2 layers, hidden 128, dropout 0.4.

## 2. Optimizer: Adam

- `torch.optim.Adam`. Adaptive per-weight learning rate from running mean and variance of gradients.
- Why: works well out of box, little tuning, converges fast on small CPU runs.
- Settings:
  - E02: `lr=0.002`, batch 32.
  - Live model (`train_live.py`): `lr=0.002`, `weight_decay=1e-3`. Weight decay = L2 penalty, shrinks weights to reduce overfit.
- Project rule: P0 allows Adam only. `ExperimentConfig.from_json` raises error for anything else.

## 3. Dataset

- **Google Speech Commands v0.02**: short 1 s word recordings, many speakers. Source for Keywords and Unknown words. Folders: `data/speech_commands_v0.02`, subset `data/isolated`, full `data/speech_commands_full` (live model, ~3000 clips per Keyword).
- **Speaker-disjoint split**: `backend/ml/data/manifest.py`. Speaker ID hashed with SHA-256 → bucket 0–99 → 80% train, 10% validation, 10% test. Same speaker never in two splits. Why: same voice in train and test inflates score.
- **Unknown**: all other Speech Commands words (`backward`, `bed`, `cat`, ...). Teaches model to reject non-Keyword speech.
- **Silence**: noise crops from background recordings at random gain, plus synthetic noise. Why: model must not fire on quiet room or fan noise.
- **AMI meetings** (`data/continuous-ami`): real natural speech, auto-annotated. Used for continuous (sliding-window) evaluation, and as hard negatives in live training.
- **TTS clips** (`data/tts`, from `scripts/gen_tts.ps1`): synthetic speech, added to live model for laptop-mic domain. Keywords plus other words as Unknown. Holdout option keeps some words out of training.
- **Features cached once** as `.npz` per split, with `normalization.json` fitted on train only. Why: fast epochs, and no test data leaks into normalization.
- Live training subsamples synthetic Unknown so the model cannot separate real from synthetic clips by domain alone.

## 4. Loss function: cross-entropy

- `nn.functional.cross_entropy(logits, labels)`. Takes raw 8 scores, applies softmax internally, penalizes low probability on the true class.
- Why: standard for multi-class classification with one correct label per clip. Project allows only this loss (`from_json` check).
- Used for both training loss and validation loss. Validation loss drives early stopping.
- Metrics (accuracy, macro-F1) are reported separately from the loss. Loss is for training decisions, F1 is for judging quality.

## 5. Training function

`train()` in `backend/ml/train.py`, isolated models. `train_model()` in `backend/ml/train_live.py`, live model. Steps per epoch:

1. Seed everything (42) for repeatable runs.
2. Load cached train and validation splits. Check normalization hash matches manifest hash.
3. Build `LstmClassifier`, Adam optimizer.
4. Shuffle train indices with seeded generator.
5. Per batch: zero gradients → forward → cross-entropy → backward → clip gradient norm to 5.0 → Adam step. Clipping stops exploding gradients, common in LSTMs.
6. Run validation pass (no gradient). Log `train_loss`, `validation_loss`, accuracies, epoch time to `curves.csv`.
7. Early stopping: keep best weights by lowest validation loss. Stop after `patience` epochs with no improvement (E02: 10, live: 20).
8. Save `model.pt`, `training.json` (config, hashes, best epoch), `curves.csv`.

Live training extras: augmentation per clip (gain, noise, time shift), worker processes compute features one epoch ahead, and the live model uses 60 epochs max.

Rule: choose model on validation loss, report test once.

## 6. Core feature: keyword detection on audio

How it works and why each piece exists:

1. **Canonicalize audio**: resample to 16 kHz mono float32. One sample rate for training, file analysis, and live mic. Mismatch here breaks model silently.
2. **MFCC**: 25 ms frames, 10 ms hop, 40 coefficients, Hamming window. Mimics human hearing, compact. Same code in train and inference (parity test in spec).
3. **Sliding window**: 1.0 s windows, stride 0.2 s (live: 0.1 s hop). Model scores every window. Why: words can start anywhere in a recording.
4. **Gate** (`backend/ml/spotting.py`):
   - Threshold: top score must reach 0.7 (live3).
   - Margin: top score must beat second score by 0.5. Stops confusable pairs (e.g. `up` vs `unknown`) from firing.
   - Level: window loud enough (≥18 dB).
   - Consecutive windows and duplicate suppression: merge repeated hits of one word into one Detection.
5. **Detection**: Keyword, start_s, end_s, Confidence. Shown on waveform in `/analyze`.
6. **Live actions** (`backend/app/services/live.py`): Detection → smoothing → cooldown (same Keyword only) → allowlisted in-app action (`right` → next slide, `go` → play, `up` → volume up, etc.). Rejections logged with reason.

Why this design: one model reused for files and mic. Gate settings are hand-tuned to cut false alarms, because raw classifier output fires on ordinary speech.

Honest limit: isolated test macro-F1 ~0.74–0.80 (target 0.85). Continuous event-F1 0.010 (target 0.70), ~2,216 false alarms/hour. Short word `up` weakest, missed inside fluent sentences. Source: `docs/evaluations/validation-report.md`, `artifacts/models/live3/training.json`.

---

## Question 3

your explaination is goob but i am confused with your explaination keep it explain

### Explanation / Answer 3

Here is the same idea again, with fewer words and an everyday comparison.

## The big picture

The project teaches a computer to hear one word out of many. Think of a child learning to spot the word "stop" in a noisy room.

1. We give the computer thousands of short recordings, each labeled with the word said.
2. The computer guesses. We tell it how wrong it was.
3. It adjusts itself a little and tries again. Repeat many times.
4. Later, we test it on recordings it never saw. That shows whether it really learned.

Every item below is one piece of that loop.

## 1. The model (LSTM)

The model is the "brain" that makes guesses.

- Audio is a sequence, like a sentence. The sound at one moment depends on what came before.
- An LSTM is a type of network that remembers earlier parts of a sequence while it reads the next part.
- Our model reads one second of audio, cut into small slices. It outputs 8 scores, one for each label: `left`, `right`, `up`, `down`, `go`, `stop`, `unknown`, `silence`. The highest score wins.
- Why LSTM: it handles sequences well and is small enough to train on a normal laptop CPU, which the course requires.
- Code: `backend/ml/networks/lstm.py`.

## 2. Optimizer (Adam)

The optimizer decides how to change the model after each mistake.

- Analogy: a teacher correcting homework. The optimizer decides how big each correction should be.
- We use Adam because it adjusts step size on its own for each weight. It needs little tuning.
- Our setting: learning rate `0.002`. That is the step size. Too big and it jumps around. Too small and it learns slowly.
- Live model also uses `weight_decay=1e-3`. This keeps weights small, so the model does not memorize the training clips.

## 3. Dataset

The dataset is the set of recordings we learn from.

- **Speech Commands**: about 1 second clips of people saying words like "left" and "cat". Many different speakers. We use it for the 6 Keywords and for "unknown" words.
- **Speaker split**: we split by speaker, not by clip. Each speaker goes to only one of three groups: train (80%), validation (10%), test (10%). Why: if the same voice is in train and test, the test score looks better than the real result.
- **Unknown**: other words, like "bed" or "bird". Teaches the model to say "not one of my 6 words".
- **Silence**: background noise, fan hum, quiet room. Teaches the model to say "nothing here".
- **AMI meetings**: real people talking in meetings. Used to check how the model behaves on normal speech.
- **TTS clips**: computer-generated speech. Added to the live model so it works with laptop mics.
- Why we cache features: we compute the MFCC features once and save them. Training then runs much faster.

## 4. Loss function (cross-entropy)

The loss is a single number that says how wrong the model is.

- If the model gives the correct label a high score, the loss is small.
- If the correct label gets a low score, the loss is big.
- Training tries to make this number smaller.
- Why cross-entropy: it is the standard choice when each clip has one correct label.
- We also compute the loss on the validation set. We use it to decide when to stop training.

## 5. Training function

The training function runs the learning loop. Here is what one round (called an epoch) does:

1. Shuffle the training clips.
2. Take a small group of clips (a batch, here 32).
3. Ask the model for its guesses.
4. Measure the loss.
5. Work out how each weight should change (backward pass).
6. Cap very large changes (gradient clipping at 5.0). This stops training from blowing up.
7. Let the optimizer apply the change.
8. Repeat for all batches. Then check the validation set.
9. Save the best model so far. If validation stops improving for 10 to 20 rounds, stop early.

Code: `backend/ml/train.py` for the isolated model, `backend/ml/train_live.py` for the live model.

## 6. Core feature: finding words in audio

This is what the app actually does.

1. **Turn sound into numbers.** Each audio slice becomes 40 MFCC numbers. MFCC keeps the parts of sound that matter for speech and drops the rest.
2. **Slide a 1 second window.** The window moves 0.2 seconds at a time across the recording. The model scores each window.
3. **Filter the guesses.** A guess counts only if:
   - the top score is at least 0.7,
   - it beats the second-best score by at least 0.5,
   - the audio is loud enough.
   Why: without filters, the model fires on ordinary speech too often.
4. **Merge repeats.** One spoken word can match several windows. We merge them into one result.
5. **Result.** Each result has a Keyword, a start time, an end time, and a confidence score. The web page shows these on a waveform.
6. **Live mic.** Same model, same filters. Then a cooldown stops one spoken word from triggering two actions. Each accepted word runs one allowed action, like "next slide".

Why one model for both files and mic: we train once and reuse it, so the results match.

## Honest result

- Clean test clips: about 74 to 80 percent macro-F1. Target was 85 percent.
- Long natural speech: almost no correct finds, and many false alarms. The continuous target was not met.
- Short words like "up" are the hardest. The model often misses them inside full sentences.

---

## Question 4

what is cross entrpy

### Explanation / Answer 4

Cross-entropy is a score for how wrong the model is on one guess. Lower is better.

**How it works:**

1. The model outputs 8 raw scores, one per label.
2. Softmax turns those scores into probabilities that add up to 1. For example: `left` 0.7, `right` 0.1, `up` 0.1, and the rest 0.1 combined.
3. Look at the probability the model gave the correct label. Call it p.
4. Loss = −log(p).

**Examples with the correct label being `left`:**

| Probability given to `left` | Loss |
|---|---|
| 0.9 (very sure, correct) | 0.11 (small penalty) |
| 0.5 | 0.69 |
| 0.1 (mostly wrong) | 2.30 |
| 0.01 (confidently wrong) | 4.61 (big penalty) |

**Why use it:**

- It gives a big penalty when the model is confident and wrong.
- It gives a small penalty when the model is almost right.
- It works well for "one correct label per clip" problems like ours.

**In our project:**

- `nn.functional.cross_entropy(logits, labels)` in `backend/ml/train.py` and `train_live.py`.
- It runs on every batch of 32 clips. The average loss is used to adjust the model.
- The same loss on the validation set decides when to stop training.

---

## Question 5

explain more more basic things used in the project

### Explanation / Answer 5

Here are the basic terms, grouped by topic. Each one says what it is and where we use it.

## Learning terms

**Epoch.** One full pass through all training clips. We run up to 80 epochs for E02.

**Batch.** A small group of clips processed together. We use 32 clips per batch. Smaller batches update more often. Larger batches use more memory.

**Gradient.** A number that says which direction and how much to change each weight to reduce the loss. Training follows these numbers.

**Backpropagation (backward pass).** The step that computes gradients, working backward from the loss through the network. In code: `loss.backward()`.

**Learning rate.** How big each weight change is. We use `0.002`. Too big and training jumps around. Too small and it crawls.

**Gradient clipping.** If a gradient is too large, cap it. We cap at `5.0`. Stops training from blowing up in LSTMs.

**Weight.** A number inside the model that gets learned. The model is just a big set of weights.

**Weight decay.** Pulls weights slightly toward zero each step. Keeps the model from memorizing. We use `1e-3` in the live model.

**Overfitting.** The model learns the training clips too well, including their quirks, and does worse on new clips. We watch the validation loss to catch this.

**Early stopping.** Stop training when validation loss has not improved for a set number of epochs (patience). We use 10 for E02 and 20 for the live model. Keep the best model, not the last one.

**Seed.** A fixed starting number for random choices. Same seed gives the same shuffle and the same results every run. We use `42`.

## Data terms

**Training set.** Clips the model learns from. 80% of speakers.

**Validation set.** Clips used during training to pick the best epoch and tune settings. 10% of speakers.

**Test set.** Clips used once at the end to report the final score. 10% of speakers. Never used to make choices. That would cheat.

**Speaker-disjoint split.** Each speaker appears in only one of the three sets. Otherwise the test score looks better than reality.

**Label.** The correct answer for a clip, such as `left` or `silence`.

**Class.** One label the model can predict. We have 8 classes.

**Normalization.** Shifting and scaling features so values sit in a similar range. We compute the numbers from training data only, then apply them to all sets.

**Cache.** Saved results of slow work, so we do not repeat it. We save computed MFCC features as `.npz` files.

## Audio terms

**Sample rate.** How many audio measurements per second. We use 16,000 per second (16 kHz). Every file gets converted to this rate.

**Mono.** One audio channel. Stereo is converted to mono.

**Frame.** A short slice of audio, 25 ms long, which is 400 samples.

**Hop.** How far the next frame starts after the previous one. 10 ms, which is 160 samples.

**MFCC (Mel-frequency cepstral coefficients).** A compact description of one audio frame, using 40 numbers. It keeps the parts of sound that matter for speech. One 1 second window gives 98 frames.

**Window.** A 1 second chunk of audio that the model looks at.

**Stride.** How far the window moves each time. We use 0.2 s for files and 0.1 s for live hop.

**Gain.** Loudness. Augmentation changes gain so the model handles quiet and loud voices.

**dB (decibel).** A loudness scale. We reject windows below 18 dB.

## Measurement terms

**Accuracy.** Percent of clips labeled correctly. Simple, but misleading when some classes are rare.

**Precision.** Of the times the model said `up`, how many were really `up`? Low precision means many false alarms.

**Recall.** Of the real `up` clips, how many did the model find? Low recall means many misses.

**F1.** One number that balances precision and recall. Higher is better.

**Macro-F1.** The F1 for each class, averaged with equal weight. Rare classes count as much as common ones. Our main score. Target 0.85.

**Confusion matrix.** A table. Rows are true labels, columns are predictions. Shows which words get mixed up, such as `up` vs `unknown`.

**Event-F1.** Like F1, but for detections in long audio. A detection counts as correct if its center is within ±0.5 s of the real word.

**False alarm.** The model fires when no keyword was said.

**Confidence.** The model's score for a detection. Not proof of correctness.

## Software terms

**Checkpoint.** A saved model file, with its weights and settings. Ours: `model.pt`. We reload it to test or run without retraining.

**ONNX.** A standard model file format. We export the model to `model.onnx` so the browser can run it on Vercel.

**Inference.** Using a trained model to make predictions. No learning happens.

**API (FastAPI).** The backend. The frontend asks it for work over HTTP.

**WebSocket.** A connection that stays open so the browser can stream mic audio to the backend in real time.

**Next.js.** The frontend framework that builds the web pages.

**SQLite.** A small database stored in one file. Holds recordings, detections, and evaluation runs.

**Evaluation Run.** A saved test result. Records which checkpoint, which split, and which settings were used. The dashboard only shows these.

If you want, I can go deeper on any one group, or pick the terms you see most in the code.

---

## Question 6

u are checking E02 -- use live3 that is the model we are using

### Explanation / Answer 6

You are right. The active model is LIVE3. Here is what LIVE3 actually uses, taken from `artifacts/models/live3/training.json` and `train_live.py`. E02 is the older isolated model and is not used by the app.

## Model (LIVE3)

- Same LSTM design as E02, but tuned for laptop mics.
- 2 layers, hidden size 128, dropout 0.4 (E02 used 0.2).
- Mean pooling over all frames, then one linear layer to 8 scores.
- Total parameters: 220,168.
- Why dropout 0.4: randomly turns off 40% of values during training. Forces the model not to depend on any single value. Helps with overfitting.

## Optimizer (LIVE3)

- Adam.
- Learning rate `0.002`.
- `weight_decay=1e-3`. E02 did not use weight decay.
- Gradient clipping at `5.0`.

## Dataset (LIVE3)

- Base: Speech Commands full set (`speech_commands_full`), about 3,000 clips per Keyword.
- Unknown: 29 other words (`backward`, `bed`, `cat`, `dog`, `marvin`, `sheila`, `zero`, ...). At most 400 clips per word (`unknown_cap`).
- Silence: background noise crops and synthetic noise.
- Augmentation: gain change, added noise, time shift. Makes the model handle quiet and loud voices and different mics.
- Synthetic TTS speech: extra clips, with TTS words that are not Keywords labeled Unknown. This is the "tts-oov" part. Adds laptop-like speech.
- Real AMI meeting clips (natural speech) are mixed in for both training and validation. Keywords cut from meetings are 1.5 s clips centered on the word. Other meeting audio becomes Unknown or Silence.
- Split still by speaker, so train, validation and test do not share speakers.

## Loss function (LIVE3)

- Cross-entropy, same as E02. Explained earlier: penalty is `-log(probability of correct label)`.
- Used on training batches and on the validation set.
- Validation loss decides which epoch to keep.

## Training function (LIVE3)

Code: `backend/ml/train_live.py`, function `train_model()`.

Setup:
- Seed `42` for all random choices.
- Features computed by background worker processes, one epoch ahead.
- Batch size 32.

Each epoch:
1. Shuffle all training clips.
2. For each batch of 32: forward pass, cross-entropy loss, backward pass, clip gradients to 5.0, Adam step.
3. Run validation set without updating weights. Record `train_loss`, `val_loss`, `val_acc`.
4. Keep the weights with the lowest validation loss.
5. Stop early if validation loss does not improve for `patience` epochs.

What the saved LIVE3 run shows:
- Best epoch: 3, validation loss 0.440, validation accuracy 0.861.
- Later epochs: training loss keeps falling (1.30 → 0.46), but validation loss goes up and down (0.44 → 0.74 → 0.70). This is the overfitting sign. So the saved model is epoch 3, not the last epoch.
- The file lists 9 epochs and total training time 1,056 s (about 17.6 min).

Be careful about one detail. The script default is 60 max epochs and patience 20, but the saved file does not list the exact patience used in that run. Stopping after 9 epochs with best at 3 means the run used a shorter patience than the default. I cannot confirm the exact setting from the saved files.

## What this means for the app

- Model file: `artifacts/models/live3/model.pt`, copied to ONNX for the Vercel build.
- Detector settings (`detector.json`): stride 0.2 s, threshold 0.7, margin 0.5, minimum level 18 dB, cooldown 1.0 s. These were chosen by hand, not by the frozen test protocol.

---

## Question 7

explain more more basic things used in the project [Assistant quote](t3-citation://v1/7955535f-e5bd-4591-8135-ed4f8b9f574d/56c029b9-8d33-4886-8ca5-4f270c2c75d1/message%3Aprovider%3AclaudeAgent%3Anative-item%3A792794a9-5186-4167-8df3-0577b6284a21?text=Learning+terms&start=90&end=104&prefix=what+it+is+and+where+we+use+it.+&suffix=+Epoch.+One+full+pass+through+al)

> **Citation Details:**
> - **Referenced Topic from Previous Explanation:** *"Learning terms"*

### Explanation / Answer 7

## Deep learning basics

**Tensor.** A grid of numbers with any number of dimensions. A batch of audio features is a 3D tensor: 32 clips × 98 frames × 40 MFCC values. PyTorch uses tensors for everything.

**PyTorch.** The Python library that builds the model, runs training, and computes gradients. In our project: `import torch`.

**Forward pass.** Feed input through the model to get outputs. Training and validation both do this. In code: `model(x)`.

**Logits.** The raw 8 scores the model outputs before softmax. They are not probabilities yet. They can be any number, positive or negative.

**Softmax.** Turns 8 raw scores into 8 probabilities that add up to 1. Cross-entropy uses these probabilities.

**Hidden size.** How many numbers the LSTM keeps as memory at each step. LIVE3 uses 128. More memory means more capacity and more parameters.

**Layers (LSTM).** Stacked LSTMs. The output of layer 1 feeds layer 2. LIVE3 has 2 layers.

**Parameters.** All the learned numbers (weights) in the model. LIVE3 has 220,168. Bigger number means bigger file and slower run.

**Mean pooling.** Average the LSTM output over all 98 frames, so one vector represents the whole second of audio. We use this instead of only the last frame because words often end before the clip ends.

**Train mode and eval mode.** `model.train()` turns dropout on. `model.eval()` turns it off. Validation must run in eval mode, or scores are noisy.

**No-grad.** Tells PyTorch not to track gradients. Used during validation, which saves memory and time because no learning happens.

**Gradient norm.** One number for the size of all gradients together. Clipping compares this number to 5.0.

**CPU vs GPU.** CPU is the general processor. GPU is a graphics chip that is faster at matrix math. Our project runs on CPU only, as the course requires. Training LIVE3 took about 17.6 minutes.

**Threads.** Parallel workers inside one process. We set 4 for training math. This is separate from the feature workers.

**Worker process.** Separate Python process that computes audio features in the background while training runs. Uses `multiprocessing.Pool`.

**Float32 and float16.** Number sizes. float32 is more precise. float16 is half the size. We store queued features in float16 to save RAM (16 GB laptop).

## Audio basics

**Waveform.** The raw sound as a list of numbers over time. Each number is the air pressure at one moment.

**Sample.** One number in a waveform. At 16 kHz, one second has 16,000 samples.

**PCM.** Plain uncompressed sound data. A WAV file is PCM. In live mode the browser sends PCM as 16-bit integers (`int16`).

**Resampling.** Changing the sample rate. Recordings at 44.1 kHz or 48 kHz become 16 kHz so training and live use the same rate.

**Spectrogram.** A picture of sound. Horizontal axis is time, vertical axis is frequency, color is loudness. MFCC is built from this.

**FFT (Fast Fourier Transform).** The math that turns one frame of waveform into frequencies. Used before MFCC.

**Hamming window.** A smooth edge applied to each frame before the FFT. Stops hard cuts from creating fake noise. Our window is `symmetric-hamming`.

**Mel scale.** A frequency scale that matches human hearing. Low frequencies are spaced finely, high frequencies coarsely. MFCC uses this.

**Cepstral coefficients.** The final step of MFCC. Compresses the mel spectrum into 40 numbers. Keeps the shape of the sound, drops fine detail.

**Padding and truncation.** Make every clip exactly 1 second. Short clips are padded with silence. Long clips are cut.

**dB (decibel).** A loudness scale using logs. Small changes in dB match how humans hear loudness. We use dB to reject quiet windows (under 18 dB).

**Gain.** Multiplies the waveform by a number, making it louder or quieter. Used in data augmentation.

**Augmentation.** Make new training examples from old ones: change gain, add noise, shift in time. Helps the model handle real-world variation.

**Noise floor.** The background level when nobody speaks. Speech must rise above it to be found.

## Software basics

**Library and dependency.** Ready-made code you import. Our Python dependencies live in `backend/pyproject.toml` and are locked in `uv.lock`.

**uv.** A Python package and environment tool. `uv run --locked` runs a command with exact locked versions.

**npm.** The Node.js package tool. Installs the frontend packages. `npm run setup` and `npm run check` are wrappers defined in `package.json`.

**Virtual environment.** A private folder of Python packages for this project only. Keeps versions from clashing with other projects.

**Lint and type check.** Ruff checks code style and mistakes. mypy checks types (for example, that a number is not passed where text is expected). `npm run check` runs them.

**Unit test and pytest.** Small automatic checks that code does what we expect. `pytest` runs them. Test files live in `backend/tests/`.

**JSON.** Text format for structured data. Our config files (`e02.json`, `detector.json`) and API responses use it.

**REST endpoint.** A URL the frontend calls over HTTP, like `GET /api/health` or `POST /api/evaluations/import`.

**SQLite.** A database stored in one file (`data/service/`). Holds recordings, detections and evaluation runs.

**Foreign key cascade.** When a recording is deleted, its detections are deleted too. Keeps the database clean.

**React and Next.js.** React builds the page parts (components). Next.js runs the React app and handles routes like `/analyze`, `/live`, `/dashboard`.

**getUserMedia.** Browser function that asks permission for the microphone. The mic runs only after the user presses Start.

**AudioWorklet.** Browser code that reads mic audio in small blocks, in real time, without blocking the page.

**onnxruntime-web.** Runs the ONNX model inside the browser. Used on Vercel so mic audio never leaves the page.

**Environment variable.** A setting passed in from outside the code. Vercel sets `VERCEL=1`, which switches the app into deployed mode.

**Deployment (Vercel).** Puts the frontend and backend online together. Config in `vercel.json`.

**Architecture decision record (ADR).** A short file in `docs/adr/` that says what was decided and why. Example: ADR-0006 says continuous spotting was a no-go.

If you tell me which words still confuse you, I can explain those with a small example.

---

## Question 8

[Assistant quote](t3-citation://v1/7955535f-e5bd-4591-8135-ed4f8b9f574d/56c029b9-8d33-4886-8ca5-4f270c2c75d1/message%3Aprovider%3AclaudeAgent%3Anative-item%3A64069c8b-8b2d-4569-bf55-ae545743d572?text=Layers+%28LSTM%29&start=814&end=827&prefix=e+capacity+and+more+parameters.+&suffix=.+Stacked+LSTMs.+The+output+of+l&comment=which+are+in+layer+%3F)

> **Citation Details:**
> - **Referenced Topic from Previous Explanation:** *"Layers (LSTM)"*
> - **User Specific Question / Focus:** **which are in layer ?**

### Explanation / Answer 8

Each layer has its own set of weights. Here is what sits in each one for LIVE3 (2 layers, hidden size 128).

**Layer 1 (reads the MFCC frames)**
- Input: 40 numbers per frame (the MFCC values).
- Learned weights: 4 groups, one per LSTM gate (input, forget, cell, output). Each group takes input from the frame and from the layer's own memory from the previous frame.
- Parameters: 87,040.

**Layer 2 (reads layer 1's output)**
- Input: 128 numbers per frame (layer 1's hidden state).
- Same 4 gate groups, now sized for 128 inputs.
- Parameters: 132,096.

**Output head (after the layers)**
- One linear layer that turns the 128 pooled numbers into 8 scores.
- Parameters: 1,032 (128 × 8 weights plus 8 biases).

**Total: 87,040 + 132,096 + 1,032 = 220,168.** This matches `training.json`.

The layers stack like this:

```
40 MFCC per frame → Layer 1 (128 memory) → Layer 2 (128 memory) → mean over frames → Head → 8 scores
```

Why two layers: layer 1 learns simple sound patterns. Layer 2 learns patterns built from those. More layers means more parameters and more training time, so we keep it at 2 for CPU training.

---

## Question 9

[Assistant quote](t3-citation://v1/7955535f-e5bd-4591-8135-ed4f8b9f574d/56c029b9-8d33-4886-8ca5-4f270c2c75d1/message%3Aprovider%3AclaudeAgent%3Anative-item%3Aa1240b56-3b3e-4c66-aae9-a777a5fc4ee8?text=MFCC+frames%29&start=127&end=139&prefix=n+size+128%29.+Layer+1+%28reads+the+&suffix=+Input%3A+40+numbers+per+frame+%28th&comment=what+is+this)

> **Citation Details:**
> - **Referenced Topic from Previous Explanation:** *"MFCC frames)"*
> - **User Specific Question / Focus:** **what is this**

### Explanation / Answer 9

An MFCC frame is one short slice of sound, turned into a list of 40 numbers.

**Step by step:**

1. Cut the 1 second of audio into small slices. Each slice is 25 ms long (400 samples at 16 kHz).
2. Move forward 10 ms and cut the next slice. The slices overlap, so no sound is missed at the edges.
3. For each slice, compute 40 numbers that describe its sound. These 40 numbers are the MFCCs.

So one MFCC frame = one slice = 40 numbers.

**Count for one second of audio:**
- 98 slices (frames), each 10 ms apart.
- 98 frames × 40 numbers = a grid of 3,920 numbers.
- This grid is the model's input for that second.

**Analogy:** think of a movie. Each frame is one still picture. An MFCC frame is a short "fingerprint" of one still of the sound. The model watches the sequence of fingerprints, like watching a movie, to recognize a word.

**Why 40 numbers and not the raw samples?** Raw samples are 400 numbers per slice and change very fast with tiny shifts in timing. The 40 MFCC numbers keep the parts of sound that carry speech (how the sound is shaped) and drop the rest. That makes the input smaller and more useful for learning.

---

## Question 10

what is activation used ?

### Explanation / Answer 10

Activation functions turn a layer's raw numbers into a shaped output. Our project uses them in three places.

**1. Inside the LSTM gates (built in, not written by us)**
- Sigmoid: squeezes values to 0 to 1. Used by the input, forget and output gates. A value near 0 means "block", near 1 means "let through".
- Tanh: squeezes values to -1 to 1. Used for the candidate memory and when reading the cell state out.
- Code: `nn.LSTM` in `backend/ml/networks/lstm.py`. We do not write these functions ourselves.

**2. After the head (no activation)**
- The final `nn.Linear(128, 8)` outputs raw logits with no activation. Logits can be any number.
- We do not use ReLU in this model.
- Why no activation here: `cross_entropy` applies softmax itself, so adding another one would be wrong.

**3. Softmax (for probabilities)**
- Softmax turns the 8 logits into probabilities that add to 1.
- It runs inside `cross_entropy` during training.
- At detection time the threshold (0.7) and margin (0.5) apply to these probabilities. So softmax runs there too. I did not re-read the scoring code to confirm the exact line.

**Summary**

| Where | Activation | Purpose |
|---|---|---|
| LSTM gates | sigmoid | 0 to 1 gate values |
| LSTM memory | tanh | -1 to 1 values |
| Output head | none | raw logits |
| Loss and scoring | softmax | probabilities |

---

