# Data preparation

Ticket 2 implements isolated-word preparation and the shared feature pipeline.
The 10–20 minute natural continuous-speech corpus has not been collected.
Collection, manual annotation and continuous-event evaluation remain pending.
No generated audio is offered as evidence that this requirement is complete.

## Speech Commands

Use the original Google Speech Commands v0.02 archive, rather than the mini-8
subset. The [publisher's dataset card](https://huggingface.co/datasets/google/speech_commands)
and [Google's release announcement](https://research.google/blog/launching-the-speech-commands-dataset/)
describe the public source and CC-BY-4.0 release. Retain the archive's `LICENSE`,
`README.md` and `_background_noise_/README.md`. The latter explicitly covers
the noise files under the same license, and distinguishes recorded sounds from
generated pink/white noise. See [source metadata](speech-commands-sources.json).
These notes record the publisher's release terms; the project does not hold
individual participants' consent forms.

[Local preparation evidence](isolated-preparation.json) records the verified
3,086-example subset, class counts, speaker counts, Unknown vocabulary and artifact
hashes. It is a data preparation report, not a model Evaluation Run. This subset
used download quota 128 and manifest cap 128; use those same values to reproduce
its selection. Full manifests and caches remain local under ignored `data/`.

Extract the archive into ignored `data/speech_commands_v0.02/`, preserving
`word/speaker_nohash_N.wav` and `_background_noise_/*.wav`. The archive download
is about 2.4 GiB; full extraction needs considerably more space. A bounded local
subset is sufficient provided every split has all six Keywords, at least three
unrelated words for Unknown, and an independent noise recording for Silence.

To download a bounded subset, run from `backend/` into an empty destination:

```powershell
uv run --locked python -m ml.data.download --output ../data/speech_commands_v0.02 --per-word-split 128
```

This streams the whole archive but only retains 128 clips per original word per
split for the six Keywords and `yes/no/cat/dog/tree`, plus all six noise sources
and license evidence. It records selection details in `acquisition.json`.
An interrupted download is partial; retry into a new empty directory.

Then prepare the manifest and cache from `backend/`:

```powershell
uv run --locked python -m ml.data.prepare manifest --root ../data/speech_commands_v0.02 --sources ../docs/data/speech-commands-sources.json --output ../data/isolated/manifest.json --max-per-class 1000
uv run --locked python -m ml.data.prepare cache --root ../data/speech_commands_v0.02 --manifest ../data/isolated/manifest.json --output ../data/isolated/features
```

`uv --project` selects dependencies; it does not change Python's working directory.

Splits use the first eight hex digits of SHA-256 of the original speaker ID modulo
100: 0–9 test, 10–19 validation, 20–99 train. This is our versioned split, not
Google's official benchmark partition. Cap selection happens after speaker
assignment. Unknown words are interleaved to preserve diversity. Each noise
recording is assigned wholesale to a split by sorted filename round-robin before
one-second, non-overlapping segments are taken. Noise speaker IDs are explicitly
source IDs because no speech identity is claimed. The manifest checks speaker,
parent-source and identical-file hash leakage. Tiny inputs fail rather than
silently moving speakers across splits. Caps apply per speech class and per noise
source; the resulting dataset is not guaranteed to have balanced class counts.

The manifest records original words, source IDs, speaker IDs, file SHA-256,
segment bounds, labels, split and dataset versions, and source license/consent/
retention. Changing subset contents changes its manifest hash. Keep that exact
manifest with each run. The default cap is a starting point for CPU experiments,
not a validated dataset size or model-quality claim.

## Feature contract

`ml.features.FeaturePipeline` accepts integer PCM WAV through `from_wav`, or
normalized floating PCM through `from_audio`. WAV supports 8/16/24/32-bit integer
PCM. Floating-point WAV and other codecs are outside P0. Channel mixing averages
channels; polyphase resampling produces 16 kHz mono float32. Nonfinite, empty or
truncated audio fails. No peak normalization changes quiet audio into loud audio.

MFCC-v1 uses a one-second window, right zero-padding or truncation, 400-sample
symmetric Hamming frames, 160-sample hops, 512-point real FFT power divided by
512, 40 unscaled triangular HTK Mel filters spanning 20–7600 Hz, natural log
with floor 1e-10, and orthonormal DCT-II including coefficient zero. No centering,
pre-emphasis, deltas or liftering. Output is `[98, 40]` float32 for each window.
The explicit contract is in `FEATURE_CONFIG` and the baseline config.

The cache command verifies audio checksums and bounds, fits streaming population
mean/std per coefficient on training frames only, floors std at 1e-6, then applies
the same statistics to every split. `normalization.json` carries feature config,
ordered labels, training frame count, manifest hash and split/dataset versions.
NPZ files contain `features`, integer `labels`, JSON `examples`, and JSON
`metadata`; read with `allow_pickle=False`. Cache metadata includes normalization
hash. Regenerate caches if source, manifest or pipeline changes. Inference loads
the saved statistics and rejects incompatible feature metadata or class order.
Ticket 3 must bind these artifacts to the actual checkpoint hash; no checkpoint
exists yet. Tests compare a training cache row with inference on the same WAV.

## Natural continuous-speech collection, pending

Prepare 10–20 minutes total across at least two consenting new speakers. Use
distinct speakers and recording sessions for validation and test, and no Speech
Commands contributors or reused source recordings. If identity separation from
the public corpus cannot be established, record that uncertainty explicitly.
Prefer recruiting known new volunteers who confirm they did not contribute to
Speech Commands. Use pseudonymous IDs, never names in manifests.

1. Explain local academic use, storage location, retention/deletion date and who
   can listen. Obtain and retain explicit consent outside Git. Allow withdrawal.
2. Record uncompressed 16-bit PCM WAV at 16 kHz mono where possible. Higher-rate
   integer PCM WAV can use the same canonical conversion. Keep each recording
   at most 10 minutes, with several minutes from each speaker. Preserve the
   original recording; do not splice isolated clips or use generated speech.
3. Record natural sentences containing `left`, `right`, `up`, `down`, `go`, and
   `stop` in different positions and speaking styles. Cover all six in each
   partition. Include ordinary conversation and near-sounding negative words
   such as "glow", "stock", "write" and "bright", plus quiet and non-speech
   noise. Avoid private content, overlapping speakers and clipped microphones.
4. Assign speakers/sessions to validation or test before any tuning. Use only
   validation to choose thresholds; keep test annotations/audio held out.
5. Listen and mark each Keyword's audible onset and offset in seconds, not the
   whole sentence. Manually review every Keyword occurrence, including incidental
   ones. Mark reviewed unrelated-speech spans `unknown`, including hard negatives,
   and non-speech spans `silence`. Rows must not overlap. Unannotated gaps are
   not automatically verified negatives. A second listener should check labels
   and boundaries before the event evaluation gate.

Copy the templates into ignored `data/continuous/`. Events have the exact header:

```csv
recording_id,keyword,start_s,end_s
```

Labels are the six Keywords plus lowercase `unknown` and `silence`. Empty
templates are not a completed dataset. Add one metadata entry per recording:

```json
{
  "recording_id": "session-001",
  "path": "audio/session-001.wav",
  "speaker_ids": ["local:speaker-001"],
  "source_group": "local:session-001",
  "source_id": "local-contributors-v1",
  "split": "validation",
  "duration_s": 0,
  "sha256": "replace with the actual file SHA-256",
  "natural": true,
  "consent_reference": "private-consent-record-id"
}
```

Replace the duration and checksum with measured values. `sources[source_id]`
must contain `url`, `version`, `license`, `attribution`, `consent`, `permitted_use`
and `retention`, using the same structure as the isolated source file. Use a local
source identifier for `url` if no public URL exists. Record the actual permission
terms, not CC-BY by default. Keep consent records and identity mappings private.

From `backend/`, validate when the data exists:

```powershell
uv run --locked python -m ml.data.annotations --root ../data/continuous --recordings ../data/continuous/recordings.json --events ../data/continuous/events.csv --isolated-manifest ../data/isolated/manifest.json
```

Validation checks 600–1200 seconds total, natural-only records, consent references,
speaker/source separation, checksums, actual durations, supported labels,
non-overlapping in-bounds events and all Keyword/negative/Silence coverage in each
partition. It cannot verify consent authenticity, speech naturalness, annotation
completeness, acoustic diversity or speaker identity by listening. Human review
is required before the corpus is accepted for evaluation. Preserve metadata,
manifest/event hashes and split versions with later Evaluation Runs.
