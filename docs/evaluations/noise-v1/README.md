# Noise test comparison: E02 clean vs noisy (P1-03)

Machine-written files from ignored `artifacts/evaluations/noise-v1/`. Each pair
holds one clean and one noisy Evaluation Run of the E02 checkpoint
(`e47e49f57e81…`). Only the dataset condition differs. The `compare-*.json`
files are summaries; the importer skips them and the API reads the stored runs.

| Kind | Clean | Noisy |
| --- | --- | --- |
| Isolated, test | macro-F1 0.727, accuracy 0.732 | macro-F1 0.188, accuracy 0.269 |
| Continuous, test | event-F1 0.010, 2216 false alarms/hour | event-F1 0.011, 2134 false alarms/hour |

The E02 checkpoint is not noise-robust: at 10 dB SNR its isolated macro-F1 falls
by 0.54 and keyword recall collapses. The continuous Detector was already near
zero on clean audio; noise changes neither its hits (2 of 16) nor its
false-alarm rate meaningfully. Both numbers stay experimental under ADR-0006.

## Dataset version `speech-commands-v0.02-p1-noise-v1`

Split version `speaker-sha256-80-10-10-noise-sources-v2`. Speaker assignment is
unchanged from P0. Silence now has three noise sources per split instead of two:

| Split | Noise sources |
| --- | --- |
| train | `doing_the_dishes`, `pink_noise`, `blue_noise` |
| validation | `dude_miaowing`, `running_tap`, `brown_noise` |
| test | `exercise_bike`, `white_noise`, `violet_noise` |

Every P0 source stays in its P0 split, so the E02 checkpoint (trained on the P0
train split) never trained on a test noise source. `blue_noise`, `brown_noise`
and `violet_noise` are synthesized by `ml.data.noise` with fixed seeds; the
manifest attribution says so. They are not natural recordings.
Manifest SHA-256: `c165b989bfb3daffae06e3cd8d7ed706e6e44af5d0978775775caaebf8ce12df`.

`violet_noise` is the only test source E02 had never seen in a similar form, and
it has the lowest Silence recall (0.70). Silence recall per source is identical in
both conditions because Silence clips are the noise sources themselves and are
not remixed.

## Protocol

- Noisy isolated clips: each non-Silence test clip is mixed with a test-split
  Silence noise clip at 10 dB SNR (`ml.data.noise.mix_noise`, seed 2026). Silence
  clips are left as they are. Features reuse the E02 normalization.
- Noisy continuous: the three test-split noise files are concatenated and mixed
  into every test Recording at 10 dB SNR, starting at a seeded random offset
  (seed 2026) per Recording. Annotations, the frozen detector
  (`continuous-v1/detector-e02.json`), the ±0.5 s tolerance and the
  duplicate-suppression rule are unchanged. Unlabeled gaps stay out of negative
  exposure in both conditions.
- Clean continuous reproduces the committed `continuous-v1` test numbers.
- The isolated runs need `--allow-different-dataset` because the checkpoint was
  trained on the P0 dataset version. Their `dataset` block names the v2 versions.

## Reproduce

From `backend/`, with the E02 checkpoint and the P0 caches in place:

```powershell
uv run --locked python -m ml.data.noise_prepare --root ../data/speech_commands_v0.02 --sources ../docs/data/speech-commands-sources.json --output ../data/noise-isolated --normalization ../data/isolated/features/normalization.json
uv run --locked python -m ml.evaluate run --checkpoint ../artifacts/models/e02/model.pt --features ../data/noise-isolated/features --split test --condition clean --allow-different-dataset --output ../artifacts/evaluations/noise-v1/e02-isolated-test-clean.json
uv run --locked python -m ml.evaluate run --checkpoint ../artifacts/models/e02/model.pt --features ../data/noise-isolated/features --split test --condition noisy --allow-different-dataset --output ../artifacts/evaluations/noise-v1/e02-isolated-test-noisy.json
uv run --locked python -m ml.evaluate_continuous run --checkpoint ../artifacts/models/e02/model.pt --root ../data/continuous-ami --isolated-manifest ../data/isolated/manifest.json --detector ../docs/evaluations/continuous-v1/detector-e02.json --split test --condition clean --noise-manifest ../data/noise-isolated/manifest.json --output ../artifacts/evaluations/noise-v1/e02-continuous-test-clean.json
uv run --locked python -m ml.evaluate_continuous run --checkpoint ../artifacts/models/e02/model.pt --root ../data/continuous-ami --isolated-manifest ../data/isolated/manifest.json --detector ../docs/evaluations/continuous-v1/detector-e02.json --split test --condition noisy --noise-manifest ../data/noise-isolated/manifest.json --noise-file ../data/speech_commands_v0.02/_background_noise_/exercise_bike.wav ../data/speech_commands_v0.02/_background_noise_/violet_noise.wav ../data/speech_commands_v0.02/_background_noise_/white_noise.wav --output ../artifacts/evaluations/noise-v1/e02-continuous-test-noisy.json
uv run --locked python -m ml.evaluate compare-noise --runs ../artifacts/evaluations/noise-v1/e02-isolated-test-clean.json ../artifacts/evaluations/noise-v1/e02-isolated-test-noisy.json --output ../artifacts/evaluations/noise-v1/compare-e02-isolated-clean-noisy-test.json
```

`ml.evaluate compare-noise` (isolated and continuous) refuses pairs that
differ in checkpoint hash, dataset/split version, kind, or (continuous) detector
configuration and tolerance. Runs were produced from a clean tree at commit
`8118525`. Each run records the noise sources of all three splits
(`noise_sources_by_split`). The isolated runs record the dataset version the
checkpoint trained on (`checkpoint_dataset_version`), because
`--allow-different-dataset` bypasses the dataset-version check; the normalization
hash is still checked. The API names a pair with a `comparison_id`
(`<clean run id>..<noisy run id>`).

## E01 and LIVE3 isolated noise runs

The same isolated clean/noisy protocol was run for E01 and LIVE3 on
2026-10-18 (same noise cache, 10 dB SNR, test split, 1,077 clips):

| Checkpoint | Clean macro-F1 | Noisy macro-F1 | Δ |
|---|---|---|---|
| E01 `519de98b` | 0.643 | 0.248 | −0.395 |
| E02 `e47e49f5` | 0.727 | 0.188 | −0.539 |
| LIVE3 `2d27a2f0` | 0.798 | 0.726 | −0.072 |

LIVE3 trained with gain/noise/shift augmentation (train-split noise plus
synthetic noise) on full Speech Commands, which is the likely reason it keeps
most of its score; its weakest noisy classes are `up` (F1 0.621), `down` (0.636)
and `stop` (0.664, many false `stop` detections). The comparison is not fully
like-for-like: the test noise files (`exercise_bike`, `violet_noise`,
`white_noise`) were never read in training, but `ml.train_live` also mixes in
synthetic white, pink and brown noise, so the white-noise part of the noisy
test is close to LIVE3's training distribution. E01 and E02 saw no synthetic
noise. Continuous noise runs exist only for E02.

```powershell
uv run --locked python -m ml.evaluate run --checkpoint ../artifacts/models/live3/model.pt --features ../data/noise-isolated/features --split test --condition noisy --allow-different-dataset --output ../artifacts/evaluations/noise-v1/live3-isolated-test-noisy.json
```

Repeat with `--condition clean` and for `e01`.
