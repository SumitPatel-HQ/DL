# Experimental Windows actions

P2-02 is implemented under [ADR-0009](adr/0009-experimental-opt-in-os-actions.md)
at the project owner's request to proceed ahead of the quality gates. It is off
by default. T01 event-F1 remains 0.057 against 0.70, and no passing 30-minute
controlled-negative microphone run has been recorded. These are remaining
acceptance failures, not passed gates or revised targets.

## Setup and use

Run `uv sync --project backend --locked` to install the locked Windows-only
Core Audio, COM and WinRT dependencies. Start the backend and frontend normally.
Open Voice Control and press Start listening. This opts into the microphone only.
The Optional Windows actions panel appears after the server accepts the session.

Refresh targets, select one target, check each allowed action, acknowledge the
experimental quality limits, then press Enable selected OS actions. Nothing is
remembered across sessions. There is no manual execution API or launch path.

| Target | Keywords and permitted actions | Limits |
|---|---|---|
| Selected default playback endpoint at discovery | `up/down`: volume up/down | Changes scalar volume by 0.05, clamped to 0–1; never unmutes. A later Windows default change does not retarget the permission. |
| Existing PowerPoint slideshow | `right/left`: next/previous slide | Attach through COM; never open a presentation or start PowerPoint. Linear adjacent slide indices, not custom-show ordering or individual animations. Stops at deck bounds. |
| Specific Windows media session | `go/stop`: play/pause | Requires Windows 10 1809+ and an app exposing supported media controls. Does not send global media keys or follow the current active session. |

Only one target is permitted per mic session. With external control enabled,
other voice actions are rejected and logged, without an in-app fallback.
Keyboard and click controls continue to operate the in-app demo only.
Unavailable targets are omitted. Refresh after opening a slideshow or media app.
Discovery may take up to eight seconds; native failure or timeout revokes consent.

Revoke, Stop, disconnect, tab hide, target refresh or native failure clears the
grant and terminates the native worker. Already-dispatched Windows/COM/WinRT
calls may complete after revocation. Revocation does not undo an action.
Unknown outcomes are logged and never automatically retried. The default
in-app-only behavior is restored immediately for subsequent decisions.

## Live WebSocket contract

The existing `/api/live` handshake, binary audio stream and decision policy are
unchanged. After `ready`, the server also sends `external_state` with `supported`,
`enabled`, `target_id`, `allowed_actions`, `targets`, `experimental: true`, `error`
and `negative_running`. Target entries contain `id`, `kind`, `label`, `actions`.
IDs are random session-local tokens; native handles and executable paths are
never accepted from the client.

Control messages have exact key sets. Unknown keys and client-supplied action
execution/Detection messages are rejected and audited. Invalid control payloads
are never stored verbatim. External controls return `external_state` with a
stable `error` value, rather than ending the mic session.

```json
{"type":"external_targets"}
{"type":"external_grant","target_id":"discovered-token","actions":["volume_up"],"acknowledge_experimental":true}
{"type":"external_revoke"}
{"type":"negative_start","no_keywords_confirmed":true}
{"type":"negative_finish"}
```

Refreshing targets revokes the existing grant. Permission changes discard
buffered audio and invalidate in-flight inference. An accepted `decision` under
external control includes `external: {status, reason, target_id}`. Status is
`applied`, `no_change`, `rejected`, `failed`, or `outcome_unknown`. The browser
must not execute its in-app action or acknowledge an external result. The
backend stores `external_<status>` on the live event and a separate audit trail
for consent, accepted/rejected decisions, attempts and native outcomes.

`GET /api/external-actions/logs?limit=100` returns the newest audit rows, up to
500 per request. `GET /api/external-actions/negative-reports` returns the latest
100 persisted reports. Both use the existing loopback host/origin restrictions.
Neither accepts executable content. Raw microphone audio is never stored.

## Scripted validation on target hardware

Use the documented Ryzen 5 5500U/16 GB laptop with the local frontend/backend.
The current implementation has fixture-level validation and a read-only native
speaker-discovery check; those do not satisfy speech or external-app acceptance.

1. Before Start, verify the mic is off and Windows volume/apps do not change.
2. Start listening. Confirm external control is still off. Refresh targets and
   select Windows volume. Permit only volume up and acknowledge experimental use.
3. Say `up` once, then pause. Check one external action result and the audit,
   actual Windows volume change, and no in-app demo change. Say `down`; verify
   the permission rejection and no Windows volume change. Log missed or extra
   actions, not just successful trials. Repeat with both actions permitted.
4. Repeat in fresh sessions for an already-open PowerPoint slideshow and an
   app exposing a Windows media session. Verify `right/left` and `go/stop`, one
   action per utterance, selected-target isolation and rejection of other actions.
5. Revoke, refresh targets, hide the tab, stop, disconnect and restart in separate
   trials. Verify permission loss, mic release where applicable, no permission
   restoration and no new external actions. Record any unknown native outcome.
6. For the negative measurement, enable the intended target/action set, confirm
   no trained Keywords in the test audio, then Start controlled negative test.
   Keep the tab visible while unrelated speech, similar-sounding words, noise
   and Silence run for at least 30 minutes of scored exposure. Do not include
   steps 3–4 in this negative session. An accepted Detection counts as a false
   action even if target permissions reject it, conservatively.
7. Press Finish and save negative report after the required exposure. Download
   JSON and retain the audit alongside your scripted-trial notes. The report
   includes session ID, timestamp, target/action set, active checkpoint hash,
   live policy, hardware, scored exposure, elapsed time and false-action count.
   The scorer caps exposure at elapsed time; the first overlapping window is
   excluded and dropped/unscored intervals do not count as exposure.
8. A report passes only on target hardware, after explicit Finish, with at
   least 1,800 seconds of scored exposure and at most one conservative false
   action per 30 minutes. Stop/Revoke/disconnect save an incomplete `not_passed`
   report. A zero-count short session cannot pass. Operator audio labels are
   assertions, not independently verified annotations. Review the protocol and
   audit before treating a report as acceptance evidence.

The integration cannot fix the failed temporal event-F1 result. Further model
work needs verified labels and additional real Keyword tokens under the frozen
validation/test protocol. A passing negative-session report does not authorize
model promotion or change the event-F1 target.

## Native API references

- [Pycaw Core Audio usage](https://github.com/AndreMiras/pycaw)
- [COM attachment to an existing application](https://comtypes.readthedocs.io/en/stable/client.html)
- [PowerPoint slide navigation](https://learn.microsoft.com/en-us/office/vba/api/powerpoint.slideshowview.gotoslide)
- [Windows media-session API](https://learn.microsoft.com/en-us/uwp/api/windows.media.control.globalsystemmediatransportcontrolssession)
- [PyWinRT asynchronous calls](https://pywinrt.readthedocs.io/en/stable/types.html)
