# 03: Train and isolated evaluation

Parent: #1 (P0 Spec — LSTM Keyword Spotting)

**What to build:** A real trained LSTM classifier with honest isolated-word evidence: E01 (1-layer, hidden-64) then E02 (2-layer, hidden-128) on identical splits with cross-entropy plus Adam plus early stopping and fixed seeds, reloadable checkpoint carrying class order and feature config, plus held-out Evaluation Run with macro-F1, per-class precision/recall/F1 including Unknown and Silence, confusion matrix, training curves, and CPU time plus peak RAM.

**Blocked by:** 02 (data, features, and annotations).

**Status:** ready-for-agent

- [ ] Training command produces a reloadable checkpoint without retraining to load
- [ ] Evaluate command reproducibly reports macro-F1 and confusion matrix on held-out split
- [ ] E01 vs E02 compared on identical split with size and CPU inference latency
- [ ] No BiLSTM, CNN, or CTC in this ticket
