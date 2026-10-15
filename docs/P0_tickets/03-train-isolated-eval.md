# 03: Train and isolated evaluation

Parent: #1 (P0 Spec — LSTM Keyword Spotting)

**What to build:** A real trained LSTM classifier with honest isolated-word evidence: E01 (1-layer, hidden-64) then E02 (2-layer, hidden-128) on identical splits with cross-entropy plus Adam plus early stopping and fixed seeds, reloadable checkpoint carrying class order and feature config, plus held-out Evaluation Run with macro-F1, per-class precision/recall/F1 including Unknown and Silence, confusion matrix, training curves, and CPU time plus peak RAM.

**Blocked by:** 02 (data, features, and annotations).

**Status:** implemented; isolated macro-F1 target not met

- [x] Training command produces a reloadable checkpoint without retraining to load
- [x] Evaluate command reproducibly reports macro-F1 and confusion matrix on held-out split
- [x] E01 vs E02 compared on identical split with size and CPU inference latency
- [x] No BiLSTM, CNN, or CTC in this ticket

E01 and E02 are trained and evaluated on the same speaker-disjoint cache.
E02 was selected on validation and reaches test macro-F1 0.740 against the 0.85
hypothesis target; E01 reaches 0.641. See [the Evaluation Runs](../evaluations/isolated-v1/README.md)
for per-class metrics, confusion matrices, curves, size, latency and peak RAM.
The main limits are 128 training clips per Keyword, weak Unknown separation and
only two noise sources per split for Silence.
