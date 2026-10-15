# Local data and artifacts

No dataset or speech recordings ship in Git. Ticket 2's manifest records the
source URL, version, license and attribution for each downloaded dataset before use.
Check the source terms rather than assuming all audio shares one license.

Record consent, permitted uses and retention for contributed recordings. Keep raw
audio and derived features under the ignored root `data/` directory. Split by speaker
and source before fitting normalization or tuning parameters. Natural continuous
speech is required for the feasibility gate; synthetic concatenations are debug data.

Checkpoints and Evaluation Run outputs live in ignored `artifacts/models/` and
`artifacts/evaluations/`. Review any small metadata or summaries before explicitly
adding them to Git. Never commit private speech, datasets, local databases or secrets.
The scaffold does not access a microphone, accept uploads or retain live audio.

See [data preparation](data/README.md) for the versioned Speech Commands subset,
train-only normalization, and pending natural-speech collection/annotation process.
