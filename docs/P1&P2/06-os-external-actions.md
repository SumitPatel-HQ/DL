# P2-02: Opt-in OS volume + external media actions (PR-16 + PR-20, gated)

Parent: P1+P2 spec ([P1&P2 Spec](P1&P2%20Spec.md)), #1 (P0 Spec — LSTM Keyword Spotting)

**What to build:** Optional, off-by-default Windows volume control and external
PowerPoint/media actions on the same Windows machine: explicit user opt-in,
explicit target + action permission, one isolated action-execution seam
separate from inference, full accept/reject/action logging. Extends the
existing upload/event injection rejection to the new seam (no executable paths
or commands cross the API boundary).

**Blocked by:** P1-04 gate passed (event-F1 + ≤1 false action per 30 min
controlled negative testing) AND a new ADR superseding ADR-0004 for opt-in OS
control. Default recommendation: `wontfix` for the course deliverable if gates
are unmet, with written rationale.

**Status:** wontfix for the course deliverable. The P1-04 gate is unmet and
the target-hardware false-action gate is unmeasured. See
[ADR-0008](../adr/0008-os-external-actions-no-go.md). OS/external actions are
not implemented; ADR-0004 remains in force. The feature criteria below remain
unchecked because the feature has not shipped.

- [ ] Microphone and OS control are both explicit opt-ins; launch state does nothing
- [ ] Only allowlisted in-app actions plus the explicitly permitted OS/media targets execute; everything else is rejected and logged
- [ ] Scripted Keyword + negative-session test records false-action counts meeting the PRD target on target hardware
- [ ] New ADR documents consent, scope, and rollback; README states limits

## Gate disposition

- P1-04 is complete as an experiment, but its gate did not pass. The
  [held-out T01 Evaluation Run](../evaluations/temporal-v1/t01-continuous-test.json)
  records 2 true positives, 52 false positives and 14 false negatives:
  event-F1 0.057 against the unchanged 0.70 target.
  [ADR-0007](../adr/0007-temporal-prototype-keep-experimental.md) keeps it
  experimental; Voice Control still serves E02.
- The [validation report](../evaluations/validation-report.md#2-failed-acceptance-reports-targets-unmet-not-redefined)
  records the live negative-session target of at most 1 false action per
  30 minutes as unmeasured. Detector false alarms/hour and fixture-based
  action tests are not measurements of live false actions on target hardware.
- No ADR authorizes OS control. ADR-0008 records this no-go decision and
  retains [ADR-0004](../adr/0004-in-app-actions-only.md), rather than
  superseding it with permissions unsupported by the evidence.

The parent spec explicitly recommends `wontfix` with a written rationale when
these gates are unmet. No OS execution seam, permission API, Windows integration
dependency, or action-log schema is added for this disposition. Microphone
capture remains an explicit Start listening action; accepted Detections drive
only the existing allowlisted in-app demo actions.

Reopen only with a passing frozen-protocol event-F1 Evaluation Run and a
recorded scripted Keyword/controlled-negative session on target hardware
meeting the false-action target. Before implementing, a new ADR must supersede
ADR-0004 and define separate microphone/OS opt-ins, explicit same-machine
target/action permissions, execution separated from inference, injection
rejection, accept/reject/action logging, and consent revocation/rollback.
