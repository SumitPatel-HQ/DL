# P1+P2 Spec — LSTM Keyword Spotting follow-through (CSV, comparisons, temporal prototype, gated OS actions)

Parent: #1 (P0 Spec — LSTM Keyword Spotting). Status: P0 tickets 01–05 implemented
(with recorded unmet targets); 06–07 still `ready-for-agent` at time of writing.
This spec assumes P0 closes per ticket 07 before any P1+P2 work starts.

Related: [PRD](../../.research/main/KWS_PRD.md) (PR-14…PR-20) ·
[P0 tickets](../P0_tickets/07-validation-report-demo.md) ·
[ADR-0004](../adr/0004-in-app-actions-only.md) ·
[ADR-0006](../adr/0006-sliding-window-gate-no-go.md)

## Problem Statement

P0 delivers a local Keyword-spotting system (Audio Analysis, Voice Control,
Model Evaluation) but leaves the PRD's P1/P2 requirements unbuilt: there is no
CSV export of Detections, no clean/noisy comparison, no model-comparison view,
no higher-quality continuous localizer, no vocabulary growth path, and no
opt-in OS-level actions. At the same time the P0 evidence records two unmet
targets — isolated macro-F1 0.740 vs 0.85 hypothesis, continuous event-F1 0.010
vs 0.70 with 2,216 false alarms/hour — plus unverified automatic annotations
(`scoring_ready: false`). Building P1/P2 in PR-number order would wire an
experimental Detector to OS actions and grow vocabulary on a broken Unknown
rejection. The problem is how to sequence PR-14…PR-20 so safe, high-value work
ships first and risky work stays gated behind measured evidence.

## Solution

Deliver P1+P2 as six small tickets behind the ticket-07 acceptance report,
ordered by risk rather than PR number: safe data-out and comparison features
first (CSV export, model comparison), data-dependent evaluation second (noise
comparison), the ADR-0006 temporal prototype as the promoted ML ticket, and
vocabulary growth plus OS/external actions last under explicit gates. Every
dashboard number stays traceable to a stored Evaluation Run; Confidence is
never presented as correctness; continuous Detections stay labeled
experimental until a frozen-protocol Evaluation Run says otherwise.

## User Stories

1. As an audio reviewer, I want to export Detections for a Recording to CSV, so that I can review timestamps outside the app.
2. As an audio reviewer, I want the CSV to contain recording ID, Keyword label, start/end seconds, Confidence score, and model version, so that the export is self-describing.
3. As an audio reviewer, I want CSV export to reflect exactly the stored Detections of the completed analysis (including the empty-result case), so that the file never contradicts the table.
4. As a course evaluator, I want a model-comparison view of baseline LSTM variants under identical Evaluation datasets, so that I can judge E01 vs E02 fairly.
5. As a course evaluator, I want comparison charts (training curves, confusion matrices, per-class precision/recall/F1) to load from stored Evaluation Runs alone, so that switching models or deleting a run updates every dependent view.
6. As a course evaluator, I want a dashboard tab with no stored run of its kind to show Not evaluated, so that missing evidence is explicit instead of silently falling back to another run.
7. As a course evaluator, I want a clean-vs-noisy comparison on labeled Evaluation sets, so that Silence/noise robustness is measured rather than assumed.
8. As a student, I want the noise comparison to name its dataset/split versions and noise sources per split, so that the known two-sources-per-split weakness is visible until fixed.
9. As a student, I want the temporal-model prototype (LSTM+CTC or equivalent sequence Detector) evaluated with the same frozen tolerance, match rule, and held-out AMI test split as the sliding-window gate, so that its event-F1 is directly comparable to 0.010.
10. As a course evaluator, I want the temporal prototype to ship as a separately scoped experiment that never replaces the isolated-classifier core until it passes the gate, so that P0 claims stay intact.
11. As a voice-control user, I want new Keywords such as `next`, `pause`, `urgent` only after licensed/consented recordings exist and the Unknown class still rejects unrelated speech, so that vocabulary growth does not collapse rejection.
12. As a voice-control user, I want vocabulary expansion to arrive as a new dataset version plus retraining plus full isolated and continuous Evaluation Runs, so that quality claims are re-measured, not inherited.
13. As a voice-control user, I want optional Windows volume control to be off by default and explicitly opted in, so that no Detection ever changes my system without consent.
14. As a voice-control user, I want external PowerPoint/media actions to require an explicit target plus action permission on the same Windows machine, so that the app cannot drive arbitrary programs.
15. As a course evaluator, I want OS/external actions to stay blocked until the Detector meets event-F1 and false-action gates (≤1 false action per 30 min of controlled negative testing), so that safety follows measurement.
16. As a student, I want every P1+P2 Evaluation Run to carry checkpoint hash, dataset/split version, config hash, and timestamp, so that reruns from saved artifacts reproduce reported metrics.
17. As a student, I want Confidence shown as a score with its threshold metadata on every Detection surface including CSV, so that no output is mistaken for ground-truth correctness.

## Implementation Decisions

- **Ticket order (risk order, not PR-number order):** P1-01 CSV export → P1-02
  model comparison → P1-03 noise comparison (needs new dataset version) →
  P1-04 temporal prototype (promoted ex-PR-19, the ADR-0006 remediation) →
  P2-01 vocabulary expansion → P2-02 OS/external actions (gated, default
  wontfix for course scope). PR-16 + PR-20 are one combined gated ticket
  because both breach ADR-0004.
- **Seams (highest existing seam per ticket, no new serving seams until gated):**
  - P1-01 reuses the stored-Detections read seam beside the existing
    detections route; the new seam is one CSV route plus the export button.
    Ideal seam count: one.
  - P1-02 reuses the Evaluation Runs list/detail/import seam; the new seam is
    a comparison view over two stored run IDs plus import-validator support
    for comparison documents (skipped, never scored, as today).
  - P1-03 reuses the dataset-version + isolated/continuous evaluate-module
    seam; the new seam is a paired clean/noisy run pair under one comparison ID.
  - P1-04 adds one new ML module behind the existing Detector interface
    (same Detection shape: Keyword, start/end, Confidence, threshold
    metadata) and is measured only through the frozen `evaluate_continuous`
    protocol seam. No new serving or UI claim seam until the gate passes.
  - P2-01 reuses the manifest → cache → train → evaluate pipeline seam with a
    bumped dataset version; no pipeline fork.
  - P2-02 adds one isolated action-execution seam behind explicit opt-in,
    physically separated from inference and mic capture; no shell execution,
    no arbitrary command path ever crosses the API boundary (existing
    injection-rejection behavior extends to the new seam).
- **API contracts:**
  - `GET /api/recordings/{id}/detections.csv` returns the stored Detections of
    the latest completed analysis with header
    `recording_id,keyword,start_s,end_s,confidence,model_id`; empty results
    return headers-only CSV with 200; pre-completion returns the existing
    `analysis_not_completed` shape (no CSV).
  - Comparison views read only stored runs; deleting a run clears dependent
    views to Not evaluated with no cross-run fallback (existing 06a rule).
  - Continuous runs never report accuracy; isolated and continuous numbers
    never share a response (existing Evaluation Runs contract).
- **Schema changes:** no SQLite schema change for P1-01…P1-04 and P2-01
  (stored Detections and Evaluation Runs already cover them). P2-02 adds
  opt-in/action-log columns or a small action table only; no mic audio stored.
- **Architectural decisions:** run-based evaluation (ADR-0005) governs all new
  metrics; in-app allowlisted actions only (ADR-0004) stays in force until a
  new ADR explicitly permits opt-in OS control; the sliding-window Detector
  stays experimental (ADR-0006) and no UI may claim reliable natural-speech
  search for it.
- **Technical clarifications:** Confidence is a classifier score with
  threshold/window metadata; event matching keeps the frozen ±0.5 s center
  tolerance and per-Keyword duplicate suppression; thresholds tune on
  validation only; test split (AMI IN1002, disjoint speakers) stays frozen;
  unlabeled gaps never count as negative exposure.

## Testing Decisions

- What makes a good test here: external behavior at the HTTP route or CLI
  command with stored-artifact fixtures, never model weights or component
  internals; every metric test recomputes from counts in the fixture so a run
  cannot carry a number its own counts contradict.
- Modules under test: recording CSV route (headers-only, populated, gated,
  injection filenames); evaluations import/list/detail for comparison pairs;
  noise-pair evaluation commands; temporal prototype through the frozen
  continuous-evaluation protocol; opt-in action execution with negative-session
  false-action counts.
- Prior art: `backend/tests/test_recordings.py` (API plumbing with generated
  PCM + biased test checkpoint, queued/running/completed/failed, delete
  cascade, range playback), `backend/tests/test_evaluations.py` (provenance,
  kind separation, deletion across restart, rejected documents), plus the
  committed `docs/evaluations/isolated-v1` and `continuous-v1` runs as
  golden-run shape references. New tests extend these files; no new harness.

## Out of Scope

- Retuning the sliding-window Detector on test data or redefining the 0.70
  event-F1 / 0.85 macro-F1 targets without documented faculty-approved
  baseline revision.
- Multi-format transcoding beyond PCM WAV, full ASR, natural-language
  commands, cloud transcription, custom wake-word training, multi-language
  speech, safety-critical alerting.
- OS volume / PowerPoint / media automation as default-on behavior, background
  mic capture on launch, stored mic audio, multi-user or hosted service
  behavior.
- New model families beyond the one bounded temporal prototype; BiLSTM/CNN
  sweeps outside the comparison ticket.

## Further Notes

- Entry gate: ticket-07 acceptance report (including explicit failed-acceptance
  entries) must exist before P1-01 starts; the 07 rerun-from-artifacts check
  becomes the reproducibility bar for every later run.
- P2-02 default recommendation is `wontfix` for the course deliverable with a
  written rationale if the event-F1/false-action gates are unmet; shipping it
  without passing gates contradicts PRD §2 and ADR-0004.
- Vocabulary for this spec follows `CONTEXT.md`: Keyword, Unknown, Silence,
  Detection, Confidence, Evaluation Run. Avoid: command, transcription, other,
  background speech, prediction, match (for Detection), recognition, accuracy
  (for Confidence), benchmark (for Evaluation Run).
- Ticket files `01-csv-export.md` … `06-os-external-actions.md` in this folder
  are the buildable units; this spec file is the parent spec.
