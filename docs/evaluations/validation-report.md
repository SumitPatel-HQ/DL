# P0 validation report (ticket 07)

Ticket 07 acceptance for the LSTM keyword-spotting P0
([spec](../KWS_SPEC.md), [ticket](../P0_tickets/07-validation-report-demo.md)).
Every story is either passed with evidence or carries an explicit
failed-acceptance report below. Unmet targets are reported, never redefined.

Status date: 15 October 2026. Automated acceptance: `npm run check` green
(101 backend pytest tests including the consolidated
`backend/tests/test_acceptance_07.py` ticket-07 run, 6 frontend contract
suites, Ruff/format, strict mypy, ESLint, TypeScript). Rerun:
`scripts/reproduce.py` reports RERUN PASS (26/26 metric and provenance
fields reproduce exactly); the recorded proof is
[`rerun-07.json`](rerun-07.json) in this directory.

## 1. Story verdicts

Audio Analysis (reviewer). Evidence: `backend/tests/test_recordings.py`,
`backend/tests/test_spotting.py`, `backend/tests/test_acceptance_07.py`,
`frontend/app/analyze`,
`frontend/tests/detections.test.mjs`, `frontend/tests/wav.test.mjs`.
The ticket-07 acceptance run (`test_acceptance_07.py`, 4 tests) walks one
journey end to end: corrupt audio rejected with `invalid_wav` and no
Recording left behind; Silence analyzed to an explicit `no_detections`
empty state; unsupported Keywords rejected with `unsupported_keyword`;
missing checkpoint staying explicit; a bad live handshake closing with a
stable code followed by a working session; Unknown words rejected without
action; and a dropped live session reopening cleanly.

| # | Story | Verdict | Evidence |
|---|---|---|---|
| 1 | Upload PCM WAV, local only | Pass | Upload → analyze → poll → Detections → delete API test; no cloud path exists |
| 2 | Corrupt/oversize rejected with actionable error | Pass | Invalid, truncated, oversize, and chunked-over-limit uploads rejected with stable codes; frontend precheck blocks files the API would reject |
| 3 | Filename, duration, supported Keywords shown | Pass | `GET /api/models/active` advertises checkpoint hash, ordered labels, and Keywords; incompatible models advertise nothing |
| 4 | Select Keywords, start analysis | Pass | `POST /api/recordings/{id}/analyze` with allowlist; unsupported Keywords return 422 |
| 5 | Detection shows Keyword, start/end, Confidence, threshold | Pass | Detection schema with `threshold` and `windows`; times are labeled approximate |
| 6 | Waveform regions with timeline | Pass | Waveform lanes with overlapping-region layout; click-to-seek covered by contract tests |
| 7 | Click row/region seeks and plays | Pass | Region, row, and clock resolve to the same Detection; ranged playback incl. suffix and 416 paths |
| 8 | Explicit zero-Detection empty state | Pass | Completed-with-nothing returns `code: "no_detections"` and `[]`; never a guessed label |
| 9 | Delete Recording, audio, and child Detections | Pass | Cascade delete incl. running-job, restart-recovery, and staged-audio paths |

Voice Control (explicit mic session). Evidence: `backend/tests/test_live.py`
(17 tests), `frontend/app/live`, `frontend/tests/live.test.mjs`.

| # | Story | Verdict | Evidence |
|---|---|---|---|
| 10 | Explicit Start/Stop, permission, status, no background mic | Pass | Socket opens only after Start; `stop` message, tab-hide, exit, close, and disconnect all end the session and free audio; one session at a time |
| 11 | `right`/`left` slide actions | Pass | Strict in-app allowlist; each Keyword maps to one demo action |
| 12 | `go` play / `stop` pause; Keyword `stop` never ends session | Pass | Allowlist plus session-stop separation; spoken `stop` only pauses |
| 13 | `up`/`down` in-app volume | Pass | `volume_up`/`volume_down` demo actions; no OS volume path (ADR-0004) |
| 14 | Unknown/Silence rejected with visible state | Pass | Unknown and Silence runs rejected once each with reason; no action |
| 15 | One utterance, at most one action | Pass | Two-window smoothing, 1.2 s cooldown, one decision per utterance; stale audio dropped with `backpressure`, never queued |
| 16 | Last Keyword, Confidence, decision, history auditable | Pass | Decision messages carry id, time, label, score, verdict, reason, action, and `processing_ms`; metadata stored in `live_events`; raw mic audio never stored |

Evaluation dashboard (student). Evidence: `backend/tests/test_evaluations.py`,
`frontend/app/dashboard`, `frontend/tests/evaluations.test.mjs`.

| # | Story | Verdict | Evidence |
|---|---|---|---|
| 17 | Real training/validation curves | Pass | Per-epoch loss/accuracy plus validation macro-F1 from saved checkpoints |
| 18 | Confusion matrix, per-class P/R/F1 incl. Unknown and Silence | Pass | E02 test macro-F1 0.740 with full matrix; Unknown weakest (F1 0.461) |
| 19 | Isolated vs continuous-event metrics separated | Pass | Kind-specific responses; continuous runs never report accuracy |
| 20 | Traceable checkpoint/split/config/timestamp | Pass | Every run carries hashes and a clean-tree revision; the isolated runs were regenerated from a clean tree with identical metrics (confusion columns now read `detected`); `scripts/reproduce.py` proves the committed files reproduce exactly |
| 21 | `Not evaluated` when no valid run | Pass | Empty states per dashboard tab; deleted runs stay deleted until explicit re-import |
| 22 | Distinct readiness/disconnect/mic-denied errors | Pass | Health distinguishes unavailable/disconnected/denied; live handshake codes separate model, audio, version, and busy failures |

Course evaluator.

| # | Story | Verdict | Evidence |
|---|---|---|---|
| 23 | CPU-only install on Ryzen 5 5500U/16GB with measured time/RAM | **Fail (unmeasured on target)** | Dev-laptop numbers only, see section 4. Clean-install proof on the target laptop is still required. |
| 24 | Held-out isolated + natural continuous runs, disjoint speakers | **Fail with report** | Both runs exist with disjoint speakers; both miss their hypothesis targets, see section 2. This is the documented no-go path, not a pass. |

## 2. Failed-acceptance reports (targets unmet, not redefined)

- Isolated classification: E02 test macro-F1 **0.740** vs hypothesis ≥ **0.85**.
  Per-class detail and confusion matrices in
  [`isolated-v1/`](isolated-v1/README.md). Unknown F1 0.461 is the weakest
  class; Silence recall depends on only two noise sources per split.
- Continuous events: E02 test event-F1 **0.010** (2 hits, 367 false
  Detections, 14 misses) vs hypothesis ≥ **0.70**; false alarms/hour
  **2,216** (350 in Unknown speech, 1 in Silence). Detail and error review in
  [`continuous-v1/`](continuous-v1/README.md). Gate decision: no-go for the
  sliding-window detector as a working feature
  ([ADR-0006](../adr/0006-sliding-window-gate-no-go.md)).
- Live acceptance: ≤1 false action per 30 min of negatives and median
  utterance-to-action ≤1 s are **unmeasured**. The decision policy
  (smoothing, cooldown, single action, Unknown/Silence rejection,
  backpressure drop) is covered by fixture-based tests only; the scripted
  microphone protocol in section 5 has not been executed, and no
  target-hardware latency run exists.
- Clean-vs-noisy comparison: **not built in P0**. It is roadmap item
  P1-03, blocked on this report. No noisy-condition Evaluation Run exists.

## 3. Rerun from saved artifacts (tolerance: exact for metrics, exact for provenance)

`scripts/reproduce.py` reruns the documented CLI commands for the E02
isolated test run and the E02 continuous test run from the saved checkpoint
(`e47e49f57e81…`) and frozen configs, then compares every metric field
within 1e-9 and every provenance field exactly (run/kind/experiment
identity, checkpoint hash, training config hash, dataset/split versions,
and the frozen detector file hash plus all six detector parameters) against
the committed files in `docs/evaluations/`. Timings, timestamps, and
checkpoint file bytes are excluded as machine measurements. Result on the
development laptop: **RERUN PASS, 26/26 fields match with difference 0**;
the machine-written proof with the producing code revision is committed as
[`rerun-07.json`](rerun-07.json) in this directory.

During this ticket the continuous runs were regenerated with the current
per-Keyword suppression rule (the committed files predated it, as their
README disclosed). Tuning on validation re-selected the identical
parameters (stride 0.2 s, threshold 0.50, `min_consecutive` 1). The
regenerated runs come from a clean tree and reproduce exactly; the gate
conclusion is unchanged.

## 4. Performance evidence (development laptop, not the target)

Measured on AMD Family 23 Model 104, 12 logical CPUs, 16 GB RAM,
Windows 11, CPU-only Torch (`artifacts/system/baseline.json`):

- E02 training mean epoch 2.13 s, peak RAM 549 MiB; single-window forward
  median 2.04 ms, MFCC plus forward 3.27 ms.
- Continuous scoring of 600 s in ~4.1 s (real-time factor 0.007),
  peak RSS ~482 MB.
- Live per-decision `processing_ms` is reported per decision, but no
  end-of-utterance-to-action distribution has been measured.

None of this is proof of performance on the Ryzen 5 5500U/16GB target.
The 10-minute recording ceiling is a validation limit, not a
target-hardware performance claim.

## 5. Scripted microphone protocol (to run on the target)

For each of the six Keywords, speak it 5 times in isolation and 5 times
inside short sentences; then read 30 minutes of mixed negatives (ordinary
conversation, near-sounding words such as "glow", "stock", "write",
"bright", quiet, and non-speech noise). Log every decision from the live
session: accepted actions must equal the intended Keyword actions, and
negatives must produce at most one false action per 30 minutes. Record
end-of-utterance to action-result latency per accepted decision and report
the median against the ≤1 s hypothesis. No results are recorded yet;
recording them here without running the session would be fabrication.

## 6. Limits

Small test Keyword counts (16 continuous test events, so per-Keyword
numbers are anecdotal); AMI references are automatic forced alignment plus
an RMS Silence heuristic, never human-verified; 10–25 s of audio per
split is unlabeled and excluded from negative exposure; two noise sources
per split; models overfit the 128-clips-per-Keyword subset (E02 train
accuracy 0.97 vs validation 0.67 at best epoch); softmax Confidence stays
overconfident out of distribution and must never be read as correctness.

## 7. Attribution

- Speech Commands v0.02 (Pete Warden, 2018, arXiv:1804.03209), CC-BY-4.0;
  archive LICENSE, README, and `_background_noise_/README.md` retained in
  local data. See [`docs/data/speech-commands-sources.json`](../data/speech-commands-sources.json).
- AMI Meeting Corpus (AMI Project), CC BY 4.0; intact crops with mono
  16 kHz conversion; word timings are publisher forced alignment, not
  verified by this project. See `data/continuous-ami/ATTRIBUTION.txt`.
- Individual participant consent forms are not held by this project; see
  the sources file for the consent and retention record.
