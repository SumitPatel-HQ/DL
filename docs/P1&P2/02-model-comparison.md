# P1-02: Model comparison view (PR-17)

Parent: P1+P2 spec ([P1&P2 Spec](P1&P2%20Spec.md)), #1 (P0 Spec — LSTM Keyword Spotting)

**What to build:** A Model Evaluation comparison view over two stored
Evaluation Runs of the same kind (baseline LSTM variants, e.g. E01 vs E02)
under identical dataset/split versions: side-by-side macro precision/recall/F1,
per-class table, confusion matrices, and training curves, all read from stored
runs alone.

**Blocked by:** P0-06a (Evaluation Runs API), P0-07 (acceptance report).

**Status:** ready-for-agent

- [ ] Comparison loads only from two stored run IDs; mismatched kind or dataset/split version is rejected with an actionable error
- [ ] Switching models or deleting a run updates every dependent view; deleted runs clear to Not evaluated with no cross-run fallback
- [ ] Isolated classification and continuous-event numbers never share a view
- [ ] API test covers matched pair, mismatched pair, and deleted-run clearing
