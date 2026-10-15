# Faculty demo script (ticket 07)

Runs end to end on CPU only, from a clean install, at
`http://127.0.0.1:3000` with the API at `http://127.0.0.1:8000`.
Allow about 15 minutes. What the audience sees at each step, and what to
say when the model fails, is written into the steps: the continuous
failure is part of the demo, not an interruption of it.

## 0. Clean install (before the audience arrives)

```powershell
uv python install 3.13
npm run setup
npm run check
npm run sysinfo
```

`check` must be green (Ruff/format, strict mypy, ESLint, TypeScript,
frontend contract suites, 97 backend tests). `sysinfo` records the demo
machine next to the reviewed baseline in `artifacts/system/baseline.json`.
Then prepare data and the E02 checkpoint (see
[reproduction](../README.md#reproduction)), and import the committed
Evaluation Runs once the API is up (step 3 recovers them live if the
database starts empty).

Start the service (two terminals at the repository root):

```powershell
npm run api
```

```powershell
npm run web
```

## 1. Upload a Recording (2 min)

Open `/analyze`. Upload a short WAV containing a Keyword (for example one
of the held-out Speech Commands clips from `data/isolated/`). Point out
the filename, duration, and the supported Keywords named by the active
model: only trained words are searchable. Upload a second file that is
corrupt (or rename a `.txt` to `.wav`) to show the actionable rejection;
then a silent clip to show the honest empty state later.

## 2. Detections on the waveform (4 min)

Select one or more Keywords, start analysis, and watch
queued → running → completed. Each Detection shows Keyword, start/end in
seconds, Confidence, and threshold metadata. Times are approximate window
spans: click a Detection row, then its waveform region, and play the
segment to verify by listening. The silent clip completes with zero
Detections and an explicit empty state, never a guessed label. Delete the
corrupt upload to show audio bytes and child Detections are removed
together.

## 3. Dashboard: honest evidence (4 min)

Open `/dashboard`. Show the E02 isolated curves (training vs validation:
accuracy keeps rising after validation loss bottoms out, i.e. overfitting
the small subset), the confusion matrix (Unknown is the weakest class),
and the isolated-vs-event separation. State the two unmet hypotheses
plainly: isolated macro-F1 0.740 vs 0.85; continuous event-F1 0.010 vs
0.70 with 2,216 false alarms/hour, almost all in ordinary speech. Delete
one Evaluation Run to show every dependent view clearing to
`Not evaluated`, then re-import it with `POST /api/evaluations/import`.
Full numbers: [validation report](evaluations/validation-report.md).

## 4. Live voice actions (4 min)

Open `/live` and press Start (explicit consent; the mic never runs
before that). Say `right` / `left` to move the sample slides, `go` to
play, `stop` to pause (saying "stop" never ends the session), `up` /
`down` for in-app volume. Mumble or stay silent to show Unknown/Silence
rejections with reasons, and repeat a Keyword fast to show cooldown
suppressing the double trigger. Every decision appears with Confidence
and accepted/rejected state; nothing leaves the laptop. Press Stop to
end the session and release the mic.

## 5. Close (1 min)

Restate the three honest sentences: the isolated classifier works on
clipped words and misses the 0.85 target; the sliding-window detector is
a measured no-go on natural speech (ADR-0006); everything the dashboard
shows traces to a saved Evaluation Run and reproduces from the
checkpoint via `scripts/reproduce.py`.
