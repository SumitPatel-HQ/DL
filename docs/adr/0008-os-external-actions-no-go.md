# Keep OS and external actions out of the course deliverable

Superseded by [ADR-0009](0009-experimental-opt-in-os-actions.md) at the project
owner's explicit request. The historical gate evidence below remains valid.

P2-02 combines optional Windows volume and external PowerPoint/media actions.
The [parent spec](../P1&P2/P1&P2%20Spec.md) recommends `wontfix` when its
quality gates are unmet. We take that path and retain
[ADR-0004](0004-in-app-actions-only.md); this decision does not authorize OS
control or supersede the in-app-only boundary.

The [T01 held-out Evaluation Run](../evaluations/temporal-v1/t01-continuous-test.json)
reports event-F1 0.057, below the unchanged 0.70 target, so completion of P1-04
does not satisfy its gate. [ADR-0007](0007-temporal-prototype-keep-experimental.md)
keeps that prototype experimental and E02 remains the serving model. The
[validation report](../evaluations/validation-report.md#2-failed-acceptance-reports-targets-unmet-not-redefined)
also records the target of at most 1 false action per 30 minutes of controlled
negative microphone testing as unmeasured on target hardware. Offline detector
false alarms cannot establish live false-action counts.

Adding an off-by-default executor now would still breach the ticket's explicit
instruction not to start until the blockers close. No OS executor, target
permissions, integration dependencies, or database changes are introduced.
Microphone capture requires Start listening and accepted Detections remain
limited to allowlisted in-app demo actions. No OS consent is requested or
retained, and there is no OS integration to roll back.

Reopening requires a frozen-protocol event-F1 result of at least 0.70 plus a
documented scripted Keyword and controlled-negative session on target hardware
meeting the false-action target. A separate ADR must then supersede ADR-0004
before implementation and define independent microphone and OS consent,
same-machine target/action permissions, an executor separate from inference,
injection rejection, accept/reject/action logging, immediate revocation, and
rollback to in-app-only behavior. Passing either measurement alone does not
authorize OS control.
