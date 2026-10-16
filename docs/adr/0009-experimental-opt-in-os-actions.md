# Permit experimental OS actions with session consent

On 16 October 2026 the project owner explicitly requested implementation of P2-02
despite its sequencing blockers. This supersedes the P2 OS-control restriction in
[ADR-0004](0004-in-app-actions-only.md) and the `wontfix` disposition in
[ADR-0008](0008-os-external-actions-no-go.md). It permits an experimental,
off-by-default integration. It does not declare either quality gate passed or
change the PRD targets. T01 event-F1 remains 0.057 against 0.70; the 30-minute
negative microphone acceptance run still needs to be performed.

Microphone access still requires Start listening. OS control needs a second,
explicit experimental acknowledgment plus a discovered target and individually
selected actions in that same session. One target is permitted at a time:
the selected Windows playback endpoint for volume up/down, an already-running
PowerPoint slideshow for next/previous slide, or a specific Windows media
session for play/pause. No application is launched. No API accepts executable
paths, native handles, shell text, or client-supplied Detection events.

Inference emits the existing decisions. A separate consent service checks the
allowlist and dispatches native execution to a disposable process using fixed
Core Audio, PowerPoint COM and Windows media-session adapters. Permissions are
kept only in memory; audit metadata and negative-session reports persist in
SQLite. Refreshing targets revokes permissions. With OS control enabled,
unpermitted voice actions are rejected, without a demo fallback. Keyboard and
click actions remain in-app. Permission changes discard buffered audio and
invalidate in-flight inference so an older utterance cannot gain new permissions.

Stop, disconnect, tab hide, target refresh, explicit Revoke, or a native failure
clears consent and terminates the worker. A timed-out or interrupted call has an
unknown outcome, is logged as such, and is never retried automatically. An action
already dispatched to Windows or an external application may complete after
revocation; revocation cannot undo its effect. Rollback is Revoke or Stop, which
restores the in-app-only behavior; restarting never restores OS permissions.

The negative-session recorder requires an operator assertion that no trained
Keywords occur. Every accepted decision counts conservatively as a false action,
including actions rejected by target permissions. Passing requires at least
1,800 seconds of scored audio and elapsed time, at most one false action per
30 minutes, explicit Finish, and the target Ryzen 5 5500U/16 GB hardware snapshot.
Interrupted or short runs cannot pass. This evidence is separate from the frozen
event-F1 Evaluation Run and cannot promote the Detector. General claims of
reliable external voice control remain blocked on both measured targets.
