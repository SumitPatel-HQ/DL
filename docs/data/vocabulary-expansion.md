# Vocabulary expansion, pending data and review

Ticket [P2-01](../P1&P2/05-vocab-expansion.md) adds `next`, `pause`, and `urgent`
through the existing manifest, feature cache, training, and evaluation pipeline.
The offline class order is:

```text
left,right,up,down,go,stop,next,pause,urgent,unknown,silence
```

The local Speech Commands subset has none of the three new words. There is no
expanded speech checkpoint or expanded Evaluation Run. Generated audio in the
tests checks pipeline behavior only and is not speech-quality evidence. The
existing E02 checkpoint remains the app default with its own eight classes.

Expanded checkpoints can be trained and evaluated offline. Installing one as
`KWS_CHECKPOINT` returns `vocabulary_not_reviewed`, with no active model or live
session. The app's Keyword chips and action map keep the six P0 Keywords. There
is deliberately no opt-out flag for this gate. Serving and action-map expansion
need a later change supported by the reviewed evidence below, including a defined
in-app action for `urgent`.

## Prepare the recordings

Use a separate ignored root such as `data/vocab-expansion/raw`. Retain the original
licensed speech, unrelated-word and noise recordings there, alongside the new
recordings. Keep the existing dataset and caches intact.

- Put speech WAVs in `<word>/<speaker_id>_nohash_<take>.wav`, including directories
  named `next`, `pause`, and `urgent`. Audio uses the existing PCM WAV pipeline.
- Speaker IDs must be stable across words and sources. Do not give the same person
  different IDs to force a split. The existing speaker hash assigns splits before
  capping clips; all eleven classes must exist in every split.
- Supply a sources JSON with keys `speech`, `noise`, `next`, `pause`, and `urgent`.
  Each uses the fields in [the existing source record](speech-commands-sources.json):
  URL or local consent reference, version, license, attribution, consent,
  permitted use, and retention. Each new-word example points at its word's source
  record, and retains its speaker ID, audio hash and interval in the manifest.
  A nonempty record is a structural check; the operator must verify its contents.
- Include at least three unrelated Unknown word types per split. New Keywords
  are removed from Unknown during manifest construction. Silence is rebuilt
  from parent noise recordings assigned to disjoint splits under the existing
  noise-source policy. Its source counts remain a limitation to report.

From `backend/`, after genuine recordings and source records exist:

```powershell
uv run python -m ml.data.prepare manifest --vocabulary expanded --root ../data/vocab-expansion/raw --sources ../data/vocab-expansion/sources.json --output ../data/vocab-expansion/manifest.json --max-per-class 128
uv run python -m ml.data.prepare cache --manifest ../data/vocab-expansion/manifest.json --root ../data/vocab-expansion/raw --output ../data/vocab-expansion/features
```

The manifest uses dataset version `speech-commands-v0.02-p2-vocab-v1` and the
existing speaker/noise split policy. Normalization fits training audio only.
Caches, normalization and checkpoints carry the ordered labels; mismatched class
orders are rejected. Old normalization and checkpoint files remain loadable.

Copy an existing LSTM experiment config to a new file with a distinct
`experiment_id`, then use `python -m ml.train` and `python -m ml.evaluate run`
with the new cache/output paths. The existing CLI options are unchanged. The
network output size follows the cache's labels. Retain the config, checkpoint,
training log, manifest and hashes together; do not overwrite E02.

## Evidence required before enabling the words

1. Review licensing/consent and speaker identities. Publish the real manifest's
   provenance and counts. Verify both isolated splits and the separate natural
   continuous set are disjoint by speaker, source and audio.
2. Retrain and run fresh held-out isolated evaluations. Report all per-class
   metrics, especially Unknown F1 against the ticket's P0 baseline of 0.46. A
   value below 0.46 requires a written verdict; existing P0 metrics cannot stand
   in for this measurement.
3. Annotate a new 600-1200 second natural continuous set with all nine Keywords,
   Unknown and Silence in both validation and test. Use the existing
   `recordings.json` and `events.csv` formats with bumped dataset/split versions.
   Automatic labels require verification before claiming scoring readiness.
4. Run `python -m ml.data.annotations` with the expanded isolated manifest,
   then `python -m ml.evaluate_continuous tune` on validation and `run` on test.
   These existing CLIs now use the expanded manifest/checkpoint class order.
   Keep the frozen 0.5 second tolerance, one-to-one matching, duplicate
   suppression and exclusion of unlabeled gaps from negative exposure.
5. Save and review the fresh runs, checkpoint/config hashes, dataset/split
   versions, timestamps and reproducibility evidence. Document unmet targets
   explicitly. The temporal prototype's prior failed gate remains in force;
   vocabulary support does not establish continuous-speech quality.

Until these steps are complete, the ticket's data and quality acceptance items
remain open and all three new words remain unavailable in the app.
