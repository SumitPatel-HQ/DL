# AMI continuous-speech preparation

The user selected AMI as the public source for Ticket 2's natural continuous
speech. `ml.data.ami_prepare` downloads the required source mixes, creates intact
crops and exports automatic annotations. It does not record speech, synthesize
audio, concatenate unrelated clips, run a model, or claim human verification.

## Reproduce

From `backend/`, after preparing the isolated Speech Commands manifest:

```powershell
uv run --locked python -m ml.data.ami_prepare --root ../data/continuous-ami --isolated-manifest ../data/isolated/manifest.json
```

To reuse the annotation ZIP from the research, add:

```powershell
--annotations ../data/research_ami/ami_public_manual_1.6.2.zip
```

The official AMI manual-annotation 1.6.2 ZIP is pinned by SHA-256. The package's
name does not mean its word timings were placed by people. AMI documents that
word timings were produced by forced alignment. Only two selected headset-mix
WAVs are downloaded, not individual microphones, videos, other meetings, or the
automatic-annotation package. Large WAV downloads use checked 4 MiB HTTP ranges
with an ETag condition, then compute a full-file SHA-256. Cached files are checked
on reuse. Incomplete downloads retain a `.part` file and are restarted on retry.
Audio bytes from the selected recordings are kept under ignored `data/`.

## Selection

The algorithm freezes the researched IN1002 test interval. It searches genuine
non-scenario meetings for a validation source with no shared global participant
IDs. Fictitious IB tasks and scenario role-play are excluded. Within each eligible
meeting it considers two non-overlapping 300-second crops centered around target
occurrences. It prefers coverage across Keywords using capped counts, with stable
meeting/time tie-breaks. Selection uses transcripts, never model predictions or
test performance. It rejects crop boundaries that cut Keywords and rejects
overlapping Keyword occurrences rather than silently dropping them.

The official 1.6.2 annotations select:

| Split | Source | Original interval in seconds | Duration |
| --- | --- | --- | ---: |
| Test | IN1002 | 482.14 to 1082.14 | 600 s |
| Validation | EN2004a | 1100.21 to 1400.21 | 300 s |
| Validation | EN2004a | 2955.90 to 3255.90 | 300 s |

Total duration is 1,200 seconds. IN1002 has global participants `MIO022`, `MIO050`,
`MIO076`, `MIO078`; EN2004a has `FEE078`, `FEE080`, `FEE081`, `FEO079`.
Keep both EN2004a crops in validation. A different meeting or microphone channel
alone does not prove different people. The pipeline also rejects identical source
audio under different meeting IDs and runs the existing isolated-overlap checks.
Global AMI IDs cannot establish real-person non-overlap with Speech Commands'
anonymous contributor IDs; this remains an explicit limitation.

## Outputs

`data/continuous-ami/` contains:

- `audio/*.wav`: three separate intact crops, mono 16 kHz signed 16-bit PCM.
- `recordings.json`: dataset/split versions, global speakers, source session groups,
  CC BY license/attribution, publisher consent evidence, crop offsets, source URLs,
  checksums and per-crop quality information.
- `events.csv`, `validation.csv`, `test.csv`: exact columns
  `recording_id,keyword,start_s,end_s`, with crop-relative times in seconds.
- `keyword-provenance.json`: each exported Keyword's global speaker, original
  start/end times, crop offset, truncation flag and automatic timing provenance.
- `report.json`: validation result, class counts, durations, speaker/source
  intersections, negative coverage, unlabeled duration, hashes and limitations.
- `sources/`: only the required downloaded audio, optional annotation ZIP and
  cache checksum sidecars.
- `ATTRIBUTION.txt`: publisher credit, license link and audio modification notice.

The pipeline uses the existing canonical audio reader and preserves source gain.
It clips only to representable PCM limits before writing PCM_s16le. It never adds
silence or normalizes the level of a quiet recording. Original recordings and
crop offsets are retained so exported samples can be traced back to their source.

## Automatic annotation and its limits

Exact, case-insensitive lexical tokens `left/right/up/down/go/stop` become
Keyword rows when the publisher supplies positive-length times. Invalid target
timings are reported as unscorable issues, not repaired with invented intervals.
Publisher-marked truncated words remain flagged candidates, with separate
unflagged/truncated counts in the report. EN2004a contains one flagged `up` token;
both partitions cover all six Keywords without that token.
Timed unrelated lexical spans are merged and subtracted from guarded Keyword
intervals to form Unknown rows. Some words such as `going`, `good`, `stopped` and
`write` are separately counted as potential hard negatives. Their acoustic
difficulty has not been verified by listening.

Silence candidates require both a gap in all speakers' timed lexical/nonverbal
activity and low audio energy. The frozen heuristic uses whole-gap RMS at
most 0.01, 100 ms annotation guards and at least 500 ms of quiet or background
noise. Gaps alone do not become Silence. This method can still miss soft speech or
annotation errors. Other gaps stay unlabeled and must not count as verified
negative exposure in later false-alarm calculations.

`manual_review_performed` and `scoring_ready` remain false. A structural validation
pass checks the schema, event bounds/non-overlap, all eight labels per split,
600-1200 seconds, actual WAV durations/checksums, provenance and speaker/source
separation. It cannot prove word alignment accuracy, annotation completeness,
correct Silence labels or strong model reliability from sparse positives.
Later Evaluation Runs must identify these as automatic reference annotations
and preserve these limitations. Human-verified ground truth remains unavailable.

To rerun the existing validator separately from `backend/`:

```powershell
uv run --locked python -m ml.data.annotations --root ../data/continuous-ami --recordings ../data/continuous-ami/recordings.json --events ../data/continuous-ami/events.csv --isolated-manifest ../data/isolated/manifest.json
```

The reviewed preparation summary is [ami-preparation.json](ami-preparation.json).
Raw recordings and full generated annotations remain ignored. No model metrics
are produced by this pipeline.

## Publisher evidence

- [AMI corpus and CC BY 4.0 release](https://groups.inf.ed.ac.uk/ami/corpus/).
- [Official annotation and per-meeting downloads](https://groups.inf.ed.ac.uk/ami/download/).
- [Human transcripts and automatically aligned word timings](https://groups.inf.ed.ac.uk/ami/corpus/transcription.shtml).
- [Natural versus elicited non-scenario meeting descriptions](https://groups.inf.ed.ac.uk/ami/corpus/nonscenariomeetings.shtml).
- [Global participant mappings and audio formats](https://groups.inf.ed.ac.uk/ami/corpus/signals.shtml).
- [Publisher ethics and participant consent](https://groups.inf.ed.ac.uk/ami/corpus/ethicsandconsent.shtml).
