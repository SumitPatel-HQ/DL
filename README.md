# Keyword Spotting

Local LSTM keyword spotting for recorded audio and bounded in-app voice control.
Ticket 1 provides the API health contract, Next.js shell and CPU-only tooling.
Ticket 2 adds Speech Commands subset manifests, speaker-disjoint splits, cached
MFCC features, train-only normalization and continuous-annotation validation.
See [data preparation and collection instructions](docs/data/README.md).
Natural continuous speech comes from automatically annotated AMI meetings.
Ticket 3 adds CPU LSTM training, self-describing checkpoints and isolated
Evaluation Runs. See [E01 vs E02 results](docs/evaluations/isolated-v1/README.md):
E02 reaches test macro-F1 0.740, below the 0.85 target.
Ticket 4 adds the experimental sliding-window detector and the continuous-event
gate. See [continuous results](docs/evaluations/continuous-v1/README.md): on held-out
natural AMI speech, event-F1 is 0.010 with 2,216 false alarms/hour. The gate
decision is no-go for continuous spotting as a working feature
([ADR-0006](docs/adr/0006-sliding-window-gate-no-go.md)). Ticket 5 adds recording
jobs and persistence. Ticket 6 adds the web app: Audio Analysis (`/analyze`) with a
linked waveform, Detection table and playback; Voice Control (`/live`) with an
explicit mic session over a loopback WebSocket; and the Evaluation dashboard
(`/dashboard`) built only from stored Evaluation Runs. See the
[API contract](docs/backend-api.md).
Ticket 7 adds the [validation report](docs/evaluations/validation-report.md)
with per-story verdicts, the rerun proof, and the documented limits, plus the
[faculty demo script](docs/demo.md).

## Setup on Windows

Install Node.js 24 LTS and [uv](https://docs.astral.sh/uv/getting-started/installation/).
Run from the repository root in PowerShell:

```powershell
uv python install 3.13
npm run setup
npm run check
npm run sysinfo
```

`setup` installs the locked Python environment in `backend/.venv` and runs `npm ci`
in `frontend`. No environment activation is needed. The first install needs internet
and downloads the CPU PyTorch wheel from its explicit
[CPU index](https://pytorch.org/get-started/locally/). Subsequent runtime is local.
The project uses Python 3.13.
Stop development servers before rerunning setup on Windows so native package files
can be replaced.

Start these in separate terminals, both at the repository root:

```powershell
npm run api
```

```powershell
npm run web
```

Open http://127.0.0.1:3000. The API serves http://127.0.0.1:8000/api/health
and API docs at http://127.0.0.1:8000/docs. Both launch commands bind to IPv4
loopback, `127.0.0.1`. The API launcher fixes port 8000; the web dev/start commands
default to port 3000. Use these IPv4 addresses, not `localhost` or `[::1]`.
`API_BASE_URL` accepts only `http://127.0.0.1:8000` with an optional trailing slash.
The backend also filters HTTP Host headers to `127.0.0.1`. That filter is separate
from socket binding; its TestClient test does not prove the listening address.
Stop each with Ctrl+C. If port 3000 is occupied, Next.js reports its alternate port.
The overview page fetches API health on the server. The three app slices call the
API from the browser; the API allows only the `127.0.0.1:3000` and `localhost:3000`
origins.
Optional server-side configuration is documented in `frontend/.env.example`.

## Health and model setup

HTTP 200 from `/api/health` means the API is running. The separate model status is
`unavailable`, with code `checkpoint_not_installed` and an explicit `setup_action`.
No trained labels are advertised. A disconnected API has a distinct web-page state.
The frontend also accepts `model.status: "ready"` with `code: "model_ready"` and
displays "Model ready". This defines the future response contract; the scaffold
backend continues to return unavailable. Unavailable responses require
`code: "checkpoint_not_installed"` and a string `setup_action`.

[Ticket 3](docs/P0_tickets/03-train-isolated-eval.md) trains checkpoints into
`artifacts/models/` with ordered labels, feature config and normalization.
`ml.checkpoint.load_checkpoint` validates and reloads them without retraining.
The API does not load a checkpoint yet, so health still reports unavailable.
Ticket 5 connects the active model to the API.

## Training and isolated evaluation

Prepare the cache first (see [data preparation](docs/data/README.md)). Then, from
`backend/`:

```powershell
uv run --locked python -m ml.train --config ml/configs/e01.json --features ../data/isolated/features --output ../artifacts/models/e01
uv run --locked python -m ml.evaluate run --checkpoint ../artifacts/models/e01/model.pt --features ../data/isolated/features --split test --output ../artifacts/evaluations/e01-test.json
```

Training uses cross-entropy, Adam, validation-loss early stopping, fixed seeds,
deterministic kernels and a fixed thread count. It writes `model.pt`,
`training.json` and per-epoch `curves.csv`. Reruns reproduce the same losses and
metrics. Evaluation accepts only validation or test splits. It refuses caches
whose manifest or normalization differ from the checkpoint. It reports macro-F1,
per-class precision/recall/F1 including Unknown and Silence, the confusion
matrix, training curves, CPU latency and peak RAM. `ml.evaluate compare` refuses
runs from different splits. Choose between models on validation, not test.

## Continuous-event evaluation

From `backend/`, tune the detector on validation, then run the frozen detector once on test:

```powershell
uv run --locked python -m ml.evaluate_continuous tune --checkpoint ../artifacts/models/e02/model.pt --root ../data/continuous-ami --isolated-manifest ../data/isolated/manifest.json --output ../artifacts/evaluations/detector-e02.json
uv run --locked python -m ml.evaluate_continuous run --checkpoint ../artifacts/models/e02/model.pt --root ../data/continuous-ami --isolated-manifest ../data/isolated/manifest.json --detector ../artifacts/evaluations/detector-e02.json --split test --output ../artifacts/evaluations/e02-continuous-test.json
```

`tune` accepts validation Recordings only. `run` refuses a detector tuned for
another checkpoint or tolerance, or tuned on Recordings that overlap test. It
reports event precision, recall and F1 with the ±0.5 s center-tolerance rule. It
also reports false alarms/hour on annotated negatives and an error review.

## Reproduction

From the repository root, after preparing data and the E02 checkpoint:

```powershell
uv run --project backend python scripts/reproduce.py
```

This reruns the E02 isolated test Evaluation Run and the E02 continuous
test run from the saved checkpoint and frozen configs, then checks every
metric field against the committed files within exact tolerance. It exits
0 with `RERUN PASS` only when all fields match (timings, timestamps, and
checkpoint file bytes are machine measurements and are excluded). It
needs the ignored local `data/` and `artifacts/models/` trees, so it is a
standalone script, not part of `npm run check`.

Full per-story verdicts, the failed-acceptance reports for the unmet
hypotheses (isolated macro-F1 0.740 vs 0.85, event-F1 0.010 vs 0.70),
the target-hardware gap, the unexecuted microphone protocol, and data
attribution are in the [validation report](docs/evaluations/validation-report.md).
The faculty walkthrough (upload to Detections to live actions to
dashboard) is in the [demo script](docs/demo.md). The P1-03 clean-vs-noisy
comparison and its protocol are in [noise-v1](docs/evaluations/noise-v1/README.md).
The P1-04 temporal BiLSTM prototype (experimental, event-F1 0.057 on test, target unmet) is
in [temporal-v1](docs/evaluations/temporal-v1/README.md) and
[ADR-0007](docs/adr/0007-temporal-prototype-keep-experimental.md).

## Checks and reproducibility

The [recording API guide](docs/backend-api.md) documents uploads, analysis jobs,
SQLite storage, playback, deletion, limits, and stable error codes. Continuous
analysis remains experimental under ADR-0006.

`npm run check` runs Ruff lint and format checks, strict Python typechecking,
TypeScript ESLint and typechecking, health-client contract tests using Node's test
runner, and the full pytest suite. CI runs the same
command on Windows and also builds the frontend. To run one test file:

```powershell
uv run --project backend pytest backend/tests/test_health.py
```

To check a production build, run `npm run build --prefix frontend`. It needs no
running API. Start it with `npm run start --prefix frontend`.

Known tooling limitation at setup: `npm audit` reports a high-severity `braces`
advisory through the development-only Next.js ESLint plugin. The registry currently
offers no patched `braces` release. Recheck on dependency updates; this package is
not in the production runtime dependency tree.

`npm run sysinfo` writes ignored `artifacts/system/local.json` with OS, CPU,
total/available RAM, installed package versions and CUDA status. The reviewed
baseline is in [artifacts/system/baseline.json](artifacts/system/baseline.json).
It describes the development machine, not proof of performance on the target
Ryzen 5 5500U/16GB laptop. Ticket 3 Evaluation Runs record measured CPU training
time, inference latency and peak RAM.

The [baseline config](backend/ml/configs/baseline.json) records ordered classes,
the frozen MFCC-v1 contract, fixed seed and CPU LSTM dimensions. Threshold
and consistency remain unset until validation calibration. Ticket 2 freezes
train-only normalization and the shared feature implementation.
Future checkpoints and Evaluation Runs must store the exact config, split version,
checkpoint hash, code revision and timestamp.

## Layout and scope

- `backend/app/`: FastAPI routes, response models, bounded recording worker and SQLite metadata.
- `backend/ml/`: configs, data, features, networks, train/evaluate/infer/spotting.
- `frontend/`: Next.js routes, components and server-side health client.
- `artifacts/`: local models, Evaluation Runs and system information.
- `scripts/`: loopback API launch, common checks and system capture.
- `data/`: ignored local audio and derived data, created when data work begins.

See the [spec](docs/KWS_SPEC.md), [implementation plan](.research/main/KWS_IMPLEMENTATION_PLAN.md),
[overview](.research/main/KWS_PROJECT_OVERVIEW.md), [PRD](.research/main/KWS_PRD.md),
[SRD](.research/main/KWS_SRD.md), [domain language](CONTEXT.md),
[ADRs](docs/adr/) and [data policy](docs/data-policy.md).
