# 02: Data, features, and annotations

Parent: #1 (P0 Spec — LSTM Keyword Spotting)

**What to build:** The complete data foundation: permitted Speech Commands subset with provenance and speaker IDs, speaker-disjoint train/val/test splits, proper Unknown from unrelated words plus Silence examples, canonical 16kHz mono float32 pipeline with 25ms frame, 10ms hop, and 40 MFCC, plus 10–20min of natural annotated continuous speech with hard negatives and Silence/noise as a disjoint CSV event set. Synthetic audio stays debug-only.

**Blocked by:** 01 (scaffold, health, and tooling).

**Status:** automated preparation complete; automatic label quality remains unverified

- [x] Dataset manifest records source, license, consent, speaker IDs, and split versions
- [x] Train/val/test have no speaker leakage; Unknown is usable, not mini-8-only
- [x] Same versioned feature pipeline with parity test on one WAV fixture for train and inference
- [x] Continuous annotation CSV (`recording_id,keyword,start_s,end_s`) from different speakers, natural sentences plus negatives, using AMI forced alignment and inferred negative labels

The user selected the AMI Meeting Corpus and authorized automatic preparation.
The pipeline produced 20 minutes in three intact mono 16 kHz PCM WAV crops:
IN1002 for test, and two EN2004a crops for validation, with disjoint global
speaker sets and all six Keywords in both partitions. The existing dataset
validator passes the actual audio and generated CSV files. See
[AMI preparation and limitations](../data/ami-preparation.md),
[AMI preparation evidence](../data/ami-preparation.json), and
[isolated preparation evidence](../data/isolated-preparation.json).
The prepared subset has 3,086 examples, including five unrelated Unknown words
and source-disjoint noise segments. Audio, full manifests and caches stay in
ignored `data/`; no model-quality or continuous-evaluation result is claimed.

AMI word times are automatically aligned and have not been manually verified.
Unknown, hard-negative candidates and Silence/noise labels are automatically
derived and remain acoustically unverified. One `up` token is publisher-marked
truncated and flagged separately; both partitions cover all Keywords without it.
Unlabeled gaps cannot be counted as verified negative exposure. Sparse counts,
overlapping ordinary speech and unknown real-person overlap across the public
corpora remain limitations. Human-verified ground truth is still unavailable;
the preparation report explicitly keeps `scoring_ready` false.
