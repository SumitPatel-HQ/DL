# P2-01: Vocabulary expansion — `next`, `pause`, `urgent` (PR-18)

Parent: P1+P2 spec ([P1&P2 Spec](P1&P2%20Spec.md)), #1 (P0 Spec — LSTM Keyword Spotting)

**What to build:** New Keywords behind a new dataset version: licensed/consented
recordings for the new words, rebuilt Unknown/Silence coverage, speaker-disjoint
splits, retrained checkpoint with updated class order, and full fresh isolated +
continuous Evaluation Runs. UI Keyword chips and the allowlisted action map gain
the new words only after the runs pass review.

**Blocked by:** P0-07 (acceptance report); sensible after P1-04 (Unknown rejection must work before growing it).

**Status:** ready-for-agent

- [ ] Dataset manifest records new-word sources, licensing/consent, speaker IDs, and bumped version
- [ ] Unknown still rejects unrelated speech (per-class Unknown F1 reported, must not regress below the P0 0.46 baseline without a written verdict)
- [ ] Unsupported terms stay disabled in the UI until their checkpoint + runs exist
- [ ] Old checkpoints keep serving their own vocabulary with correct chips until replaced
