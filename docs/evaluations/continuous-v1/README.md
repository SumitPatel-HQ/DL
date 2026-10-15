# Continuous-event Evaluation Runs: E02 window detector

This directory holds machine-written files copied from ignored `artifacts/`:

- [`detector-e02.json`](detector-e02.json): the frozen detector, tuned on validation only.
- [`e02-continuous-validation.json`](e02-continuous-validation.json): the tuning split. These results are optimistic.
- [`e02-continuous-test.json`](e02-continuous-test.json): the single held-out test run.

Each run names the checkpoint SHA-256, the detector parameters and the detector
file SHA-256. It also names the dataset/split version and the Recording hashes.
It includes every Detection, the per-Keyword counts and the error review lists.
The files were produced from an uncommitted working tree (`code_revision.dirty`
is `true`). Scoring and tuning are deterministic, so a rerun after commit should
give the same metrics.

## Protocol

The protocol follows ADR-0002 and was frozen before test:

- Model: E02 (2-layer LSTM, hidden 128), selected on isolated validation macro-F1 in
  ticket 3. Checkpoint `e47e49f57e81…`.
- Windows: 1.0 s windows, scored with the shared MFCC-v1 pipeline and the unchanged
  isolated classifier.
- Voting: a window votes for a Keyword only if that Keyword is its top class and its
  score reaches the threshold. Unknown or Silence as the top class rejects the window.
- Merging: consecutive votes for one Keyword merge into one Detection. The Detection
  spans the union of its windows. Confidence is the maximum window score in the run.
- Consistency: runs shorter than `min_consecutive` windows are dropped.
- Duplicate suppression: if two Detections of the same Keyword have centers less
  than 1.0 s apart, only the more confident Detection is kept. Different Keywords
  never suppress each other. The committed result files below predate this
  per-Keyword rule: they were produced when suppression also applied across
  Keywords. Regenerate them with the commands under "Reproduce".
- Match rule: a match needs the same Keyword. The Detection center must lie within
  ±0.5 s of the reference center. Each reference matches at most one Detection.
- Truncated reference: one publisher-flagged truncated `up` in validation is
  ignored. It counts as neither a hit nor a miss.
- False alarms/hour: unmatched Detections centered inside annotated Unknown or
  Silence spans, divided by those spans' total hours. Unlabeled gaps are excluded
  from the negative exposure.
- Tuning grid, on validation only: stride {0.2, 0.25} s, threshold 0.30–0.95 in
  steps of 0.05, and `min_consecutive` {1, 2, 3}. The objective is maximum event-F1.
  Ties go to fewer false alarms, then to stricter settings.
- `ml.evaluate_continuous tune` refuses test data. `run` on test refuses a
  detector file in these cases: a different checkpoint, tolerance or dataset
  version; a tuning split that is not validation; or tuning Recordings that overlap test.

Selected parameters: stride 0.2 s, threshold 0.50, `min_consecutive` 1, suppression 1.0 s.

## Data

The data is AMI `ami-natural-p0-v1`: natural meeting speech, with speakers and
meetings disjoint from each other and from Speech Commands.
See [AMI preparation](../../data/ami-preparation.md).

| | Validation (EN2004a, 2 × 300 s) | Test (IN1002, 600 s) |
|---|---|---|
| Speakers | FEE078, FEE080, FEE081, FEO079 | MIO022, MIO050, MIO076, MIO078 |
| Keyword events scored | 25 (+1 ignored truncated) | 16 |
| Unknown / Silence seconds | 534.6 / 41.7 | 561.4 / 8.8 |
| Unlabeled seconds | 14.4 | 25.1 |

The references come from automatic forced alignment and an RMS heuristic. Nobody
has verified them by listening. The Keyword counts are small, so per-Keyword
numbers are anecdotal.

## Results

| | Validation (tuned on) | **Test (held out)** |
|---|---|---|
| Detections | 188 | 260 |
| True / false positives | 5 / 183 | 1 / 259 |
| Missed Keywords | 20 | 15 |
| Event precision | 0.027 | **0.004** |
| Event recall | 0.200 | **0.062** |
| **Event F1** | 0.047 | **0.007** |
| False alarms/hour (Unknown + Silence) | 1,124 | **1,559** |
| — in Unknown speech | 1,199 /h (178) | 1,578 /h (246) |
| — in Silence | 173 /h (2) | 408 /h (1) |

The preliminary target is event-F1 ≥ 0.70 with acceptable false-alarm behavior.
The test result misses that target by two orders of magnitude.

Scoring 600 s takes about 3.1 s on the development laptop, which is a real-time
factor of 0.005. Peak RSS is about 480 MB. Speed is not the problem.

## Error review

`error_review` in each run lists every hit, miss and false Detection. For each
miss, the run also gives the top class of the nearest window. For each false
Detection, the run gives the annotated region it falls in. The `--examples`
option writes ±1.5 s WAV clips of the first five items of each kind to ignored
`artifacts/evaluations/e02-continuous-*-examples/`. These clips are listening aids.
No human listening verdicts are recorded yet.

The automatic review shows these patterns:

- **Most false alarms are confident.** On test, 137 of 259 false Detections score
  ≥ 0.85. Many score above 0.99, for example `stop` at 267.9 s, `go` at 455.2 s
  and `up` at 358.2 s. All of these are inside ordinary Unknown speech. A higher
  threshold cannot separate them from true hits. On validation, the false-alarm
  rate is still 537/h at threshold 0.80, and recall falls to 0.08.
- **Sustained misfires.** Some false Detections persist across many windows. For
  example, test `go` at 18.0–22.8 s spans 20 consecutive windows. Adjacent-window
  consistency therefore does not suppress them. Validation tuning selected
  `min_consecutive` 1.
- **Keywords are absorbed as Unknown.** On test, the nearest window to 8 of the 15
  missed Keywords predicts Unknown. On validation, 15 of 20 do. Other misses become
  `left` or another Keyword. Short Keywords inside fluent speech, about 0.1–0.3 s
  long, rarely dominate a 1 s window.
- **False alarms are in speech, not Silence.** 246 of the 259 test false Detections
  fall in Unknown spans. One falls in Silence, two overlap other Keyword spans and
  ten fall in unlabeled gaps.
- **Hits are well timed.** The six hits across both splits have center errors of
  0.15–0.34 s, inside the ±0.5 s tolerance.

The probable causes follow from the data. The classifier only saw isolated,
clipped words. Its Unknown class was trained on 1 s clips of other isolated words,
not on fluent, overlapping meeting speech. Its isolated test macro-F1 is already
only 0.740. Fluent speech is out of distribution, and the softmax scores stay
overconfident there.

## Gate decision

**Decision: no-go for the sliding-window detector as a working feature. Prototype
a separate temporal model.** See
[ADR-0006](../../adr/0006-sliding-window-gate-no-go.md).

The plan's middle option is one bounded retune, but that option requires a
credible, small improvement path. None exists here. The best validation event-F1
over the whole grid is 0.047. The dominant errors are confident misfires that no
threshold, stride or consistency setting removes. Closing a gap from 0.007 to
0.70 needs a different model or training data, not detector tuning. The target
is unmet and has not been redefined.

## Reproduce

From `backend/`, after preparing the isolated cache, AMI data and E02 checkpoint:

```powershell
uv run --locked python -m ml.evaluate_continuous tune --checkpoint ../artifacts/models/e02/model.pt --root ../data/continuous-ami --isolated-manifest ../data/isolated/manifest.json --output ../artifacts/evaluations/detector-e02.json
uv run --locked python -m ml.evaluate_continuous run --checkpoint ../artifacts/models/e02/model.pt --root ../data/continuous-ami --isolated-manifest ../data/isolated/manifest.json --detector ../artifacts/evaluations/detector-e02.json --split test --examples ../artifacts/evaluations/e02-continuous-test-examples --output ../artifacts/evaluations/e02-continuous-test.json
```

Use `--split validation` for the reference run on the tuning split.
