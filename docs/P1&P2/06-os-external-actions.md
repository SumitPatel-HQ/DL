# P2-02: Opt-in OS volume + external media actions (PR-16 + PR-20)

Parent: [P1+P2 spec](P1&P2%20Spec.md), #1. Related: [PRD](../../.research/main/KWS_PRD.md).

**Status:** implemented as an experimental opt-in integration; measured quality
acceptance remains incomplete. On 16 October 2026 the project owner explicitly
authorized implementation despite the sequencing blockers.
[ADR-0009](../adr/0009-experimental-opt-in-os-actions.md) supersedes the earlier
`wontfix` decision and permits development and explicit experimental use.

- [x] Microphone and OS control are separate explicit opt-ins; launch does nothing
- [x] Only allowlisted in-app actions and explicitly permitted target/actions execute; rejections and action outcomes are logged
- [ ] Scripted Keyword + negative-session test records false-action counts meeting the PRD target on target hardware
- [x] New ADR documents consent, scope, and rollback; README states limits

## Delivered behavior

After Start listening, the user refreshes available Windows targets, chooses
one, selects individual allowed actions and acknowledges the unmet quality gates.
The fixed mappings are `up/down` to the selected playback endpoint's volume,
`right/left` to an existing PowerPoint slideshow, and `go/stop` to a selected
Windows media session's play/pause. Only one target is enabled per session.
Unsupported or unavailable integrations do not appear as targets. No app launches.

The browser sends opaque target IDs and allowlisted action names, never paths,
native handles or executable text. Only backend model decisions can execute
actions. While enabled, a voice action lacking permission is rejected with no
in-app fallback. Keyboard and clicks remain in-app. The isolated Windows worker
retains the exact native target rather than following whichever app is active.

Revoke, Stop, disconnect, tab hide, target refresh and execution failure clear
permissions and terminate the worker. Permission changes discard old buffered
audio. Audit metadata persists; consent and microphone audio do not. Unknown
native outcomes are logged and never retried automatically. Already-dispatched
external calls may complete after revocation and cannot be undone.

## Remaining evidence gates

- [T01 held-out run](../evaluations/temporal-v1/t01-continuous-test.json):
  event-F1 0.057, below 0.70. [ADR-0007](../adr/0007-temporal-prototype-keep-experimental.md)
  still keeps it experimental. Voice Control serves E02, not T01.
- The target-hardware 30-minute controlled-negative microphone measurement is
  still unmeasured. The new recorder saves scored exposure, elapsed time,
  conservative false-action counts, hardware and active checkpoint/policy.
  Short or interrupted sessions cannot pass; fixture tests do not satisfy it.
- Fixing the event-F1 gap needs verified labels, more real Keyword tokens and a
  separately reviewed model experiment under the frozen protocol. Reusing the
  held-out test split for tuning or lowering the target is not an implementation fix.

See [setup, scripted validation and API](../os-external-actions.md). The original
quality gates now govern acceptance and reliability claims rather than preventing
implementation. Neither gate is represented as passed.
