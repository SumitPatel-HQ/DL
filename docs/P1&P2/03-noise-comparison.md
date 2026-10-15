# P1-03: Noise test comparison (PR-15)

Parent: P1+P2 spec ([P1&P2 Spec](P1&P2%20Spec.md)), #1 (P0 Spec — LSTM Keyword Spotting)

**What to build:** A clean-vs-noisy Evaluation pair on labeled sets: new
dataset version adding noise sources per split (fixing the known
two-sources-per-split Silence weakness), fresh isolated + continuous Evaluation
Runs for the active checkpoint on both conditions, and a dashboard comparison
naming dataset/split versions and noise sources.

**Blocked by:** P0-07 (acceptance report), new dataset version + ticket-03 protocol rerun.

**Status:** done

- [x] Clean and noisy runs share checkpoint hash and protocol; only the dataset condition differs
- [x] Provenance names noise sources per split; Silence recall is reported per source where feasible
- [x] Unlabeled gaps are excluded from negative exposure (existing rule)
- [x] CLI + API tests cover the paired-run import, mismatched-provenance rejection, and Not evaluated empty states
