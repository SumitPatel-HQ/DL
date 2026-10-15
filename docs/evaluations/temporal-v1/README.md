# Temporal prototype Evaluation Runs: T01 frame BiLSTM (P1-04)

Machine-written files copied from ignored `artifacts/`. This is the ADR-0006
remediation: one bounded temporal model, scored through the frozen continuous
protocol, reported as a separate experimental Evaluation Run. It never replaces the
isolated-classifier core.

- [`detector-t01.json`](detector-t01.json): post-processing parameters, tuned on validation only.
- [`t01-continuous-validation.json`](t01-continuous-validation.json): the tuning split. These results are optimistic.
- [`t01-continuous-test.json`](t01-continuous-test.json): the single held-out test run.
- [`t01-training.json`](t01-training.json): training history, best epoch, dataset provenance and dev frame report.

Each run names the checkpoint SHA-256, the detector file SHA-256, the dataset/split
version and the Recording hashes. The run also carries a `baseline` block that points
at the sliding-window E02 run on the same split. The files keep the `continuous` run
shape, so the existing import and dashboard read them unchanged. The dashboard labels
them experimental and shows the temporal operating point.

## Result

Same data, tolerance and match rule as [continuous-v1](../continuous-v1/README.md).

| | Validation (tuned on) | **Test (held out)** | E02 sliding window, test |
|---|---|---|---|
| Detections | 11 | 54 | 369 |
| True / false positives | 5 / 6 | 2 / 52 | 2 / 367 |
| Missed Keywords | 20 | 14 | 14 |
| Event precision | 0.455 | **0.037** | 0.005 |
| Event recall | 0.200 | **0.125** | 0.125 |
| **Event F1** | 0.278 | **0.057** | 0.010 |
| False alarms/hour (Unknown + Silence) | 31 | **316** | 2,216 |

The event-F1 ≥ 0.70 target is **unmet** on test (0.057). It is not redefined.

The prototype cuts false alarms about seven-fold and event-F1 about six-fold against
the 0.010 baseline. It finds exactly the same number of Keywords as the baseline
(2 of 16). The gain comes from emitting far fewer Detections, not from hearing more
Keywords. Validation event-F1 (0.278) is five times the test value, so the tuned
operating point does not carry over to unseen speakers. Test has 16 events, so one
hit moves recall by 0.06. Per-Keyword numbers are anecdotal.

Scoring 600 s takes about 4.3 s on the development laptop (real-time factor 0.007),
peak RSS about 0.8 GB. Speed is not the problem.

## Error review

`error_review` in each run lists every hit, miss and false Detection. For each miss the
run gives the top frame class near the reference center.

- **Every miss is read as background.** On test, 14 of 14 missed Keywords have
  `background` as the top frame class. On validation, 20 of 20 do. The model absorbs
  short Keywords in fluent speech, as the sliding-window classifier did.
- **False alarms stay confident.** 33 of 52 test false Detections score ≥ 0.85. Fifty fall in
  Unknown speech, none in Silence.
- **Hits are well timed.** Both test hits have center errors within 0.03 s.
- **Dev signal agreed.** On the held-out dev meetings the best mean Keyword frame F1
  was 0.016 (epoch 18). Frame-level scores on real meeting speech were poor before
  any test run.

Probable causes: only a few hundred real Keyword tokens exist in the training
meetings (`left` and `stop` only a handful), the labels are automatic forced
alignments, and part of the Keyword evidence is pasted Speech Commands clips, which
differ acoustically from words inside fluent speech.

## Protocol (frozen before test)

- Model: 2-layer bidirectional LSTM, hidden 128, 7 frame classes (6 Keywords +
  background), MFCC-v1 features at a 10 ms hop. Long audio runs in 10 s chunks with
  2 s context each side.
- Training: AMI meetings only, speakers disjoint from the validation (EN2004a) and
  test (IN1002) crops. 18 train meetings plus 48 augmented 120 s streams, where
  Speech Commands **train-split** Keyword clips are pasted over meeting audio. 4 dev
  meetings with other speakers pick the best epoch. Normalization is fitted on train
  only. Validation and test crops are never read by training.
- Decoding: smoothed frame score per class; a frame votes for a Keyword only when that
  Keyword is its top class and the score reaches the threshold; consecutive votes merge
  into one Detection spanning those frames; runs shorter than `min_frames` are dropped;
  Confidence is the maximum smoothed score in the run.
- Duplicate suppression, match rule, ±0.5 s center tolerance, truncated-reference
  handling and false-alarm exposure are the code paths of `ml.evaluate_continuous`.
- Tuning grid, on validation only: smoothing {1, 5, 9} frames, threshold 0.30–0.95 in
  steps of 0.05, `min_frames` {3, 5, 8, 12}. The objective is maximum event-F1. Ties go
  to fewer false alarms, then to stricter settings.
- `ml.evaluate_temporal tune` refuses test data. `run` on test refuses a detector file
  for a different checkpoint, tolerance, dataset version or format; one that was not
  tuned on validation; or one whose tuning Recordings overlap test.

Selected parameters: smoothing 1 frame, threshold 0.65, `min_frames` 3, suppression 1.0 s.

## Gate verdict

**Keep experimental.** See [ADR-0007](../../adr/0007-temporal-prototype-keep-experimental.md).

## Reproduce

From `backend/`, after preparing the isolated cache and the AMI data (see
[AMI preparation](../../data/ami-preparation.md)):

```powershell
uv run --locked python -m ml.data.ami_temporal_prepare --root ../data/temporal-ami --annotations ../data/research_ami/ami_public_manual_1.6.2.zip --isolated-manifest ../data/isolated/manifest.json --speech-root ../data/speech_commands_v0.02
uv run --locked python -m ml.train_temporal --config ml/configs/temporal.json --root ../data/temporal-ami --output ../artifacts/models/t01
uv run --locked python -m ml.evaluate_temporal tune --checkpoint ../artifacts/models/t01/model.pt --root ../data/continuous-ami --isolated-manifest ../data/isolated/manifest.json --output ../artifacts/evaluations/detector-t01.json
uv run --locked python -m ml.evaluate_temporal run --checkpoint ../artifacts/models/t01/model.pt --root ../data/continuous-ami --isolated-manifest ../data/isolated/manifest.json --detector ../artifacts/evaluations/detector-t01.json --split test --baseline ../artifacts/evaluations/e02-continuous-test.json --output ../artifacts/evaluations/t01-continuous-test.json
```

Use `--split validation` and `--baseline ../artifacts/evaluations/e02-continuous-validation.json`
for the reference run on the tuning split. Training is seeded; the AMI downloads are
large (about 1.7 GB) and are cached under ignored `data/`.
