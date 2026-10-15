# Isolated Evaluation Runs: E01 vs E02

Machine-written Evaluation Runs copied from ignored `artifacts/`. Every file names
its checkpoint SHA-256, config hash, dataset/split version, manifest and
normalization hashes, code revision and timestamp. The checkpoints themselves stay
local in `artifacts/models/e01/` and `artifacts/models/e02/`.

These runs were regenerated on 15 October 2026 from a clean working tree
(`code_revision.commit` `cd2fd37`, `dirty` `false`) with the current code.
Metrics, curves, and confusion values are unchanged from the previous
committed runs; the confusion-matrix column label is now `detected`
(the import API still accepts legacy `predicted` files). Training and
evaluation are deterministic, so a rerun reproduces them; see the
[ticket 07 validation report](../validation-report.md) for the rerun evidence.

Data: `speech-commands-v0.02-p0-subset-v1`, split
`speaker-sha256-80-10-10-noise-source-round-robin-v1`, 1,051 train, 1,018
validation and 1,017 test examples. Both models share this cache, features,
seed 42, Adam (lr 0.002), cross-entropy, batch 32, gradient clip 5, dropout 0.2,
4 training threads, and early stopping on validation loss (patience 10). Only
size differs. Configs: [`e01.json`](../../../backend/ml/configs/e01.json),
[`e02.json`](../../../backend/ml/configs/e02.json).

Measured on the development laptop (AMD Family 23 Model 104, 12 logical CPUs,
16 GB RAM, Windows 11, CPU-only PyTorch 2.14.1). Latency is batch-1 for one
one-second window, median of 200 runs after warm-up, 6 inference threads.

| | E01 | E02 |
|---|---|---|
| LSTM | 1 layer, hidden 64 | 2 layers, hidden 128 |
| Parameters | 27,656 | 220,168 |
| Checkpoint size | 118,077 B | 888,997 B |
| Best epoch / epochs run | 9 / 19 | 10 / 20 |
| Mean train epoch | 0.38 s | 2.13 s |
| Training peak RAM | 469 MiB | 549 MiB |
| Validation macro-F1 | 0.547 | 0.675 |
| **Test macro-F1** | **0.641** | **0.740** |
| Test accuracy | 0.644 | 0.734 |
| Model forward, median | 0.60 ms | 2.04 ms |
| MFCC + forward, median | 1.73 ms | 3.27 ms |

E02 was selected on validation macro-F1 before reading test results. Test
results confirm the order. Neither model meets the macro-F1 ≥ 0.85 hypothesis
target. Per-class precision/recall/F1, confusion matrices and training curves
are in the `*-test.json` and `*-validation.json` files.

Observed failure modes, from the confusion matrices:

- Unknown is the weakest class (test F1 0.32 for E01, 0.46 for E02). Keywords
  are often predicted as Unknown. E01 confuses Unknown mostly with `left` and
  `right`; E02 mostly with `down` and `go`.
- Silence depends heavily on the noise source. Train noise is `doing_the_dishes`
  and `pink_noise`. Validation noise (`dude_miaowing`, `running_tap`) gets E02
  Silence recall 0.51. Test noise (`exercise_bike`, `white_noise`) gets 0.98.
  Two noise sources per split are too few to claim general Silence robustness.
- Training accuracy keeps rising after validation loss bottoms out (E02 reaches
  0.97 train accuracy at epoch 20 versus 0.67 validation accuracy at its best
  epoch), so the models overfit the 128-clips-per-Keyword training set.

The largest available lever is more training data. The subset keeps 128 clips
per original word per split, while Speech Commands has far more training-speaker
clips. A larger subset is a new dataset version and needs a fresh download,
manifest and cache, then a rerun of both experiments.

Mean pooling over LSTM outputs replaced final-state pooling after a validation-
only probe. A final-state E01 with learning rate 0.001 and patience 8 reached
best validation loss 1.87; the committed mean-pooled E01 reaches 1.54.

Reproduce from `backend/`:

```powershell
uv run --locked python -m ml.train --config ml/configs/e01.json --features ../data/isolated/features --output ../artifacts/models/e01
uv run --locked python -m ml.evaluate run --checkpoint ../artifacts/models/e01/model.pt --features ../data/isolated/features --split test --output ../artifacts/evaluations/e01-test.json
uv run --locked python -m ml.evaluate compare --runs ../artifacts/evaluations/e01-test.json ../artifacts/evaluations/e02-test.json --output ../artifacts/evaluations/compare-e01-e02-test.json
```

Repeat for `e02` and `--split validation`. A second E01 training run reproduced
identical per-epoch losses and identical test metrics. Checkpoint file hashes
differ between reruns because the checkpoint stores its creation timestamp.
