# 02: Data, features, and annotations

Parent: #1 (P0 Spec — LSTM Keyword Spotting)

**What to build:** The complete data foundation: permitted Speech Commands subset with provenance and speaker IDs, speaker-disjoint train/val/test splits, proper Unknown from unrelated words plus Silence examples, canonical 16kHz mono float32 pipeline with 25ms frame, 10ms hop, and 40 MFCC, plus 10–20min of natural annotated continuous speech with hard negatives and Silence/noise as a disjoint CSV event set. Synthetic audio stays debug-only.

**Blocked by:** 01 (scaffold, health, and tooling).

**Status:** ready-for-agent

- [ ] Dataset manifest records source, license, consent, speaker IDs, and split versions
- [ ] Train/val/test have no speaker leakage; Unknown is usable, not mini-8-only
- [ ] Same versioned feature pipeline with parity test on one WAV fixture for train and inference
- [ ] Continuous annotation CSV (`recording_id,keyword,start_s,end_s`) from different speakers, natural sentences plus negatives
