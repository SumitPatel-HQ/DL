# Deploying to Vercel

The frontend and the backend deploy together as one Vercel project, using
[Vercel Services](https://vercel.com/docs/services). `vercel.json` at the repository root defines both:

- `web`: the Next.js app in `frontend/`. It serves every path except `/api/*`.
- `api`: the FastAPI app in `backend/`, which handles `/api/*` on the same domain.

## One-time setup

1. Push the repository to GitHub.
2. In Vercel, select **Add New → Project** and import the repository.
3. Keep **Root Directory** at the repository root. `vercel.json` selects the services.
4. Select **Deploy**. You do not need to set any environment variables.

Vercel sets `VERCEL=1` during builds and at runtime. Both apps read this variable
and switch to deployed mode on their own.

## What changes when deployed

| Area | Local (`npm run api` + `npm run web`) | Deployed on Vercel |
|---|---|---|
| Inference | torch checkpoint `artifacts/models/live3/model.pt` | ONNX export `backend/deploy/model.onnx` (same weights, same scores) |
| Live voice | WebSocket `/api/live` to the API | Scored in the browser with `onnxruntime-web`; mic audio never leaves the page |
| Windows Actions | Available on Windows | Hidden (the server runs Linux) |
| Analysis jobs | Background worker thread | Runs inside the `analyze` request (60 s limit) |
| Upload limit | 40 MiB | 4 MiB (Vercel's request body limit is 4.5 MB). Save long WAVs as 16 kHz mono. |
| Storage | `data/service/` (SQLite + WAVs) | Instance temp directory. Recordings disappear when the instance recycles. |
| Evaluation Runs | Imported on demand | Imported automatically from `backend/deploy/evaluations/` at startup |
| Origins | `localhost:3000` only | The site's own domain |

## After changing the model or Evaluation Runs

The deployment serves committed artifacts, not `artifacts/` (that folder is gitignored).
Regenerate the artifacts and commit them:

```powershell
uv run --project backend --extra train python scripts/export_deploy.py
git add backend/deploy frontend/public/models
```

The script writes these files:

- `backend/deploy/model.onnx`, `detector.json` and `evaluations/` for the API.
- `frontend/public/models/kws.onnx` and `kws.json` for the in-browser live engine.

`backend/tests/test_deploy.py` fails if these files no longer match their detector.

If you change `LiveStream`, `LiveDecider` or the MFCC pipeline, regenerate the
browser parity fixture too:

```powershell
uv run --project backend python scripts/live_fixture.py
```

## Trying deployed mode locally

```powershell
# Terminal 1: API in deployed mode on port 8100
cd backend
$env:KWS_DEPLOYED="1"; $env:KWS_ORIGINS="http://localhost:3100"
uv run uvicorn app.main:app --host localhost --port 8100

# Terminal 2: deployed-mode frontend build on port 3100
cd frontend
$env:KWS_DEPLOYED="1"; $env:API_BASE_URL="http://localhost:8100"; $env:BACKEND_URL="http://localhost:8100"
npm run build; npx next start --hostname localhost --port 3100
```

To try only the in-browser live engine against the normal local API, open
`http://localhost:3000/live?engine=browser`.
