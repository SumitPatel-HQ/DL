# Keyword Spotting

Local LSTM keyword spotting for recorded audio and bounded in-app voice control.
Ticket 1 provides the API health contract, Next.js shell and CPU-only tooling.
Ticket 2 adds Speech Commands subset manifests, speaker-disjoint splits, cached
MFCC features, train-only normalization and continuous-annotation validation.
See [data preparation and collection instructions](docs/data/README.md).
Natural continuous-speech collection and annotation remain pending.
Training, checkpoint loading, audio analysis, microphone sessions, persistence and
Evaluation Runs are not implemented yet. No model results are claimed.

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
The web server fetches API health directly; no browser CORS configuration is needed.
Optional server-side configuration is documented in `frontend/.env.example`.

## Health and model setup

HTTP 200 from `/api/health` means the API is running. The separate model status is
`unavailable`, with code `checkpoint_not_installed` and an explicit `setup_action`.
No trained labels are advertised. A disconnected API has a distinct web-page state.
The frontend also accepts `model.status: "ready"` with `code: "model_ready"` and
displays "Model ready". This defines the future response contract; the scaffold
backend continues to return unavailable. Unavailable responses require
`code: "checkpoint_not_installed"` and a string `setup_action`.

Complete [ticket 3](docs/P0_tickets/03-train-isolated-eval.md) to train and install
a compatible checkpoint plus ordered labels and feature metadata in
`artifacts/models/`. This scaffold has no loader, so copying an arbitrary `.pt` file
does not change readiness. Ticket 3 must add validated checkpoint loading before
reporting a model ready.

## Checks and reproducibility

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
Ryzen 5 5500U/16GB laptop. No CPU training/inference time is measured in ticket 1.

The [baseline config](backend/ml/configs/baseline.json) records ordered classes,
the frozen MFCC-v1 contract, fixed seed and CPU LSTM dimensions. Threshold
and consistency remain unset until validation calibration. Ticket 2 freezes
train-only normalization and the shared feature implementation.
Future checkpoints and Evaluation Runs must store the exact config, split version,
checkpoint hash, code revision and timestamp.

## Layout and scope

- `backend/app/`: FastAPI routes, response models, future services and database.
- `backend/ml/`: configs, data, features, networks, train/evaluate/infer/spotting.
- `frontend/`: Next.js routes, components and server-side health client.
- `artifacts/`: local models, Evaluation Runs and system information.
- `scripts/`: loopback API launch, common checks and system capture.
- `data/`: ignored local audio and derived data, created when data work begins.

See the [spec](docs/KWS_SPEC.md), [implementation plan](.research/main/KWS_IMPLEMENTATION_PLAN.md),
[overview](.research/main/KWS_PROJECT_OVERVIEW.md), [PRD](.research/main/KWS_PRD.md),
[SRD](.research/main/KWS_SRD.md), [domain language](CONTEXT.md),
[ADRs](docs/adr/) and [data policy](docs/data-policy.md).
