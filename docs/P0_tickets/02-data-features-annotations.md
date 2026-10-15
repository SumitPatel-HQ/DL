# 02: Data, features, and annotations

Parent: #1 (P0 Spec — LSTM Keyword Spotting)

**What to build:** The complete data foundation: permitted Speech Commands subset with provenance and speaker IDs, speaker-disjoint train/val/test splits, proper Unknown from unrelated words plus Silence examples, canonical 16kHz mono float32 pipeline with 25ms frame, 10ms hop, and 40 MFCC, plus 10–20min of natural annotated continuous speech with hard negatives and Silence/noise as a disjoint CSV event set. Synthetic audio stays debug-only.

**Blocked by:** 01 (scaffold, health, and tooling).

**Status:** independent implementation complete; natural collection and annotation pending

- [x] Dataset manifest records source, license, consent, speaker IDs, and split versions
- [x] Train/val/test have no speaker leakage; Unknown is usable, not mini-8-only
- [x] Same versioned feature pipeline with parity test on one WAV fixture for train and inference
- [ ] Continuous annotation CSV (`recording_id,keyword,start_s,end_s`) from different speakers, natural sentences plus negatives

The user confirmed that natural recordings do not yet exist and requested that
collection and annotation remain pending without blocking independent work.
See [preparation and collection instructions](../data/README.md), the empty
annotation templates, and [local isolated preparation evidence](../data/isolated-preparation.json).
The prepared subset has 3,086 examples, including five unrelated Unknown words
and source-disjoint noise segments. Audio, full manifests and caches stay in
ignored `data/`; no model-quality or continuous-evaluation result is claimed.
