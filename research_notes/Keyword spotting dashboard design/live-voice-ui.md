# Live Voice-Control / Mic Session UI Patterns

## How do best voice UIs handle explicit consent, mic permission denied, live VAD/level feedback, start/stop affordances so mic never runs in background?

### Takeaway
Explicit Start-gated `getUserMedia` + persistent listening state + `track.stop()` on Stop/disconnect is the privacy-safe baseline; denied permission is sticky per-origin and must be taught as a settings fix, while live level/VAD meters prove the mic is alive without implying recording.

### Cited Findings
- `getUserMedia()` must always get user permission before opening mic; browsers may offer once-per-domain persistence but must ask at least first time — [Source](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia)
- Only top-level secure-origin document can request permission unless Permissions Policy expressly grants an iframe; insecure HTTP fails — [Source](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia)
- Denial rejects with `NotAllowedError` DOMException; also thrown for prior/global denial or blocked Permissions Policy — [Source](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia)
- Branch on `err.name`, never message string; `NotAllowedError`=denied/blocked, `NotFoundError`=no device, `OverconstrainedError`=relax `err.constraint` and retry, `NotReadableError`=OS/another app holds device, `SecurityError`=not recoverable in-page — [Source](https://webrtc.smnandre.dev/calls/capture)
- Once origin is `denied`, re-calling `getUserMedia` will not re-show prompt; detect `NotAllowedError` and show site-settings instructions, do not loop re-requests — [Source](https://webrtc.smnandre.dev/calls/capture)
- Camera/mic light staying on means tracks not stopped; call `track.stop()` on every track; use `enabled=false` for reversible mute vs `stop()` for full release — [Source](https://webrtc.smnandre.dev/calls/capture)
- Chrome shows permission dialog on `navigator.mediaDevices.getUserMedia()` with allow/deny choice — [Source](https://web.dev/articles/getusermedia-intro)
- Porcupine web demo flow: wait for worker + WebVoiceProcessor init, select Keyword, then start recording to detect; mic permission requested at `subscribe()` time, not page load — [Source](https://github.com/Picovoice/porcupine/tree/master/demo/web)
- Porcupine browser mic handled via Web Audio API abstracted by WebVoiceProcessor which also downsamples; engine runs pre-packaged as Web Worker — [Source](https://github.com/Picovoice/porcupine/blob/master/README.md)
- Google conversation design: ask questions to signal turn-taking (who "has the mic"), do not monopolize, keep turns brief/relevant; multimodal: spoken prompt carries core message, display prompt is condensed scannable version, chips carry options — [Source](https://developers.google.com/assistant/conversation-design/learn-about-conversation)
- Google: focus on what user can do not what they can say; avoid teaching robotic keyword commands; when giving examples mark them as examples not exhaustive list — [Source](https://developers.google.com/assistant/conversation-design/commands)
- Alexa natural-speech checklist: ask a question only immediately before opening microphone; use implicit, explicit, and contextual cues that input was understood; display progress cues on screened devices — [Source](https://developer.amazon.com/en-US/alexa/alexa-haus/natural-speech)
- Alexa: prevent errors by asking only just before mic open to avoid user speaking over Alexa; escalate errors (repeat → explain → disambiguate → end session if abandoned) — [Source](https://developer.amazon.com/en-US/alexa/alexa-haus/natural-speech)
- Live input meter should be shown so user can tell mic works / clipping; RMS tracks loudness ("am I heard"), peak tracks clipping; serious meter shows both with slow-decay peak hold — [Source](https://webrtc.smnandre.dev/calls/audio)
- Meter must be driven at `requestAnimationFrame` cadence sampling AnalyserNode, not fast timer; compute RMS from `getFloatTimeDomainData`, map dB `-60..0`, flag clipping when peak `>0.99` — [Source](https://webrtc.smnandre.dev/calls/audio)
- AudioWorklet volume-meter pattern: `registerProcessor('vumeter')`, RMS per block, `Math.max(rms, volume*SMOOTHING_FACTOR)`, `postMessage({volume})` every ~25 ms via `intervalInFrames = ms/1000*sampleRate`; requires secure context; suspend/resume AudioContext for Start/Stop toggle — [Source](https://stackoverflow.com/questions/62702721/how-to-get-microphone-volume-using-audioworklet)
- Chrome volume-meter sample: measure mic volume with AudioWorkletProcessor — [Source](https://googlechromelabs.github.io/web-audio-samples/audio-worklet/basic/volume-meter/)
- Simple sound-meter demo pattern: Start button → grant access → green bar responds; Stop ends monitoring; status messages; fully client-side — [Source](https://github.com/Samuelson777/Simple-Sound-Level-Meter)
- hark speech detection: threshold + history (e.g. 400 ms over threshold) fires `speaking` / `stopped_speaking` + `volume_change`; gauge green <65% speech, yellow <79%, red >80% — [Source](https://webrtchacks.com/getusermedia-volume/)
- Mycroft troubleshooting: if Wake Word Detection fails, reduce mic level 25–50% and retry; verify with known-good audio played through speakers — [Source](https://mycroft-ai.gitbook.io/docs/using-mycroft-ai/troubleshooting/wake-word-troubleshooting)

### Inferences
- For ADR-0003 compliance: Start button is the only `getUserMedia`+AudioWorklet+WebSocket-open trigger; Stop/disconnect/tab-hide must `track.stop()` + close socket + suspend context; persistent red "LISTENING" pill + animated level bar + elapsed timer makes background running impossible to miss.
- Permission-denied panel should differentiate first-deny (re-click Start re-prompts) vs sticky-denied (show per-browser lock-icon steps), plus no-device and in-use variants; never auto-retry in a loop.
- Live VAD dot (Silence/Unknown/speech) plus RMS bar separates "mic alive" from "Detection emitted", reducing false privacy alarm.

### Gaps
- No reliable source found on exact browser lock-icon copy per Chrome/Edge/Firefox version for denied-mic recovery steps; needs per-target-browser screenshot verification.

## What realtime feedback patterns prevent double-trigger confusion: cooldown timers (~1.2s), smoothing, last-Keyword + Confidence + decision + action result display, accepted vs rejected visual language?

### Takeaway
One-utterance-one-action requires visible cooldown lockout plus trigger-level smoothing plus a last-Detection card (Keyword + Confidence + accepted/rejected + action result); rejected Unknown/Silence must be equally visible but visually distinct.

### Cited Findings
- TensorFlow speech-commands game demo uses `probabilityThreshold 0.80`, `overlapFactor 0.50`, `commandCooldownMs 400 ms` temporal guard to prevent rapid-fire execution from single utterance — [Source](https://deepwiki.com/joy-andraos/hci-snake-game/4.3-voice-control-(tensorflow-speech-commands))
- Same demo `stop()` closes mic stream and recognizer when switching modes/leaving — [Source](https://deepwiki.com/joy-andraos/hci-snake-game/4.3-voice-control-(tensorflow-speech-commands))
- Mycroft Precise defaults: `sensitivity 0.5` (0.0–1.0, higher=more sensitive), `trigger_level 3` (1–10 consecutive positives required, higher=fewer false activations) — [Source](https://mycroft-ai.gitbook.io/docs/using-mycroft-ai/customizations/wake-word)
- PocketSphinx `threshold` in scientific notation: smaller value (e.g. 1e-18) less sensitive/fewer false positives but more misses; larger (e.g. 1e-9) more sensitive/more false positives — [Source](https://www.digikey.co.uk/en/maker/projects/how-to-create-a-custom-wake-word-for-mycroft-ai/3f0b7486b9804687b6a9e5164ee4cf99)
- PlayAbility voice mapping: short distinct commands ("jump", "reload", "pause game"); test loop is Start Voice Detection → wait for listening → say at normal volume → confirm recognized word appears in voice feedback → confirm action activates; pause briefly between commands so recognizer treats as separate events — [Source](https://help.playability.gg/en/article/set-up-voice-commands-4e9dsv/)
- PlayAbility: voice is event not held input; use Short/Long/Double/Triple Press or Toggle (e.g. play-pause), not hold-default — [Source](https://help.playability.gg/en/article/set-up-voice-commands-4e9dsv/)
- Game voice guidance: dedicated Stop commands to release all inputs if stuck; hold + stop pair for latching actions — [Source](https://gameaccess.info/voice-control-series-a-beginners-guide-to-using-voice-controls-for-gaming/)
- Google: be brief/relevant, leverage context, vary frequent responses so confirmations do not become tiresome; confirm simply ("Done!") not formally ("Transaction complete") — [Source](https://design.google/library/speaking-the-same-language-vui)
- Alexa: hold context within conversation (no re-state/re-confirm); allow over-answer and under-answer; do not state the obvious; display additional context on screen — [Source](https://developer.amazon.com/en-US/alexa/alexa-haus/natural-speech)
- Alexa: avoid error messages that only say "I didn't hear you" (causes verbatim repeat); explain what went wrong with best available data + clarify what is needed and why — [Source](https://developer.amazon.com/en-US/alexa/alexa-haus/natural-speech)
- Debounce/cooldown coding pattern: flag + cooldown window + visual dimming (transparency 0.75) to indicate inactive period before reactivation — [Source](https://create.roblox.com/docs/scripting/debounce)

### Inferences
- 1.2 s cooldown fits between game 400 ms guard and TF2-style spam counters: long enough to swallow LSTM ring-buffer repeats and user re-saying Keyword, short enough for slide pacing; show countdown ring/progress on last-Detection card and dim Keyword chips during lockout.
- Last-Detection card fields directly implement CONTEXT.md: Keyword label, Confidence (model score, not correctness), decision (accepted vs rejected Unknown/Silence/low-Confidence/cooldown), action result (e.g. "slide→3", "paused", "volume 60→70"); history list keeps ~20 rows with green accepted / grey-amber rejected language.
- Smoothing = require N consecutive frames + Confidence ≥ threshold + cooldown expired before emitting Detection; rejected path still logged for demo transparency.

### Gaps
- No citable source for optimal 1.2 s value itself; 400 ms is the only concrete game-voice number found — 1.2 s remains a project constraint needing empirical tuning against ring-buffer hop length.

## What in-app demo stage designs work for hands-free slides (left/right) + transport (go/stop play-pause) + volume (up/down): layout, focus management, keyboard fallback, latency display (end-of-utterance to action)?

### Takeaway
Split stage (slides) + transport + volume panel with always-visible Keyword legend, keyboard-identical fallback, and end-of-utterance-to-action latency readout makes the 6-Keyword mapping (left/right/up/down/go/stop) demonstrable hands-free without OS control.

### Cited Findings
- Google multimodal: group devices by components used; on screened devices spoken prompt carries core message, screen shows supplementary visual + continuation chips; display prompt condensed, chips minimum include offered options — [Source](https://developers.google.com/assistant/conversation-design/conversation-design-process/scale-your-design)
- Google: keep spoken and display prompts same narrative; do not lead to different topic/branch — [Source](https://developers.google.com/assistant/conversation-design/conversation-design-process/scale-your-design)
- Alexa storyboard: user journey + device type + audio response (speech + earcons/music) + visual response (screens) combined in one artifact — [Source](https://developer.amazon.com/en-US/alexa/alexa-haus/voice-fundamentals)
- Alexa testing: verify skill responds by fulfilling request or explaining; responses concise/terse, chunked, no jargon, written for ear not eye, same language as account — [Source](https://developer.amazon.com/en-US/docs/alexa/custom-skills/voice-interface-and-user-experience-testing-for-a-custom-skill.html)
- PlayAbility test contract maps to demo acceptance: command → recognized word in feedback → mapping activates → repeat several times in target app; also requires target app focus for output — [Source](https://help.playability.gg/en/article/set-up-voice-commands-4e9dsv/)
- Porcupine web demo constrains scope to in-app Keyword select + detect callback `detection.label` → UI highlight, no OS action — [Source](https://github.com/Picovoice/porcupine/blob/master/README.md)
- Deepgram/Nabla production guidance: measure partial-to-final latency, final count, stalled recognition as freshness signals — [Source](https://developers.telnyx.com/docs/voice/stt/websocket-streaming/production-patterns)
- Alexa feedback timing guidance: ask direct yes/no or 1–5 question at convenient time, do not block primary task, acknowledge feedback, allow on-demand feedback, lifetime timeout after 5 declines, never after "stop" or repeated errors — [Source](https://developer.amazon.com/en-US/alexa/alexa-haus/patterns-and-components/patterns-feedback)

### Inferences
- Layout: left = slide stage (large index, prev/next arrows flash on left/right Detection), right column = transport (play/pause icon swaps on go/stop) + volume bar (up/down steps) + latency chip (ms from end-of-utterance to action) + cooldown bar; bottom = Detection history; top = mic session bar; side legend lists 6 Keywords with keyboard equivalents (←/→/↑/↓/Space/…).
- Focus: stage container keeps `tabindex` + visible focus ring; voice and keyboard call same `dispatchAction()` so fallback is identical; on `stop` Keyword pause media but keep mic session alive (distinct from mic Stop).
- Latency display: backend emits Detection timestamp + inference ms; frontend shows `utterance-end → action` delta per row; target <500 ms local loopback, flag outliers.
- Strict allowlist (ADR-0004): each Keyword maps to exactly one in-app function; Unknown/Silence/low-Confidence/cooldown → rejected, no action, one utterance max one action.

### Gaps
- No source found giving slide+transport+volume combined demo layout; synthesis from multimodal + storyboard + Porcupine patterns, needs prototype validation for sightline/scan order.

## What Web Audio / getUserMedia / AudioWorklet + WebSocket streaming UX pitfalls exist (disconnect, backpressure, no raw mic stored) and how to surface them recoverably?

### Takeaway
Treat the mic WebSocket as disposable: versioned handshake, bounded queues with `bufferedAmount` backpressure, keepalive, explicit end marker, exponential-backoff reconnect, and a "no raw mic stored" privacy line; surface each failure with one-line cause + one-click recovery.

### Cited Findings
- Web Audio mic pipeline: `getUserMedia({audio:true})` → `AudioContext` → `createMediaStreamSource` → `audioWorklet.addModule("processor.js")` → `AudioWorkletNode` → `source.connect(processor).connect(destination)`; processor copies input to output in `process([input],[output])` — [Source](https://web.dev/patterns/media/microphone-process)
- `navigator.mediaDevices` undefined means non-secure origin (use HTTPS or localhost); empty device labels mean no grant yet — call `getUserMedia` before `enumerateDevices` — [Source](https://webrtc.smnandre.dev/calls/capture)
- Treat WebSocket session as disposable; reconnect on network failure, server close, idle timeout, restart; do not send audio until `open`; stop sending on close, preserve buffer, reconnect then resume — [Source](https://developers.telnyx.com/docs/voice/stt/websocket-streaming/production-patterns)
- Bounded producer: chunk 2048–8192 byte binary frames; max buffered 5–10 s; pause/drop low-priority audio when full; never unbounded queue — late audio increases delay/desync, prefer dropping stale — [Source](https://developers.telnyx.com/docs/voice/stt/websocket-streaming/production-patterns)
- Monitor `bufferedAmount` for backpressure; cap 64–128 KB per connection; alert at 256 KB; adaptive drop not indefinite queue; one WebSocket per audio stream (no multiplex head-of-line blocking) — [Source](https://deepgram.com/learn/nodejs-voice-ai-production-guide)
- Graceful shutdown: drain queue → send `CloseStream`/`END`/`RU9G` → wait for finals → close socket; set shutdown timeout, mark incomplete if finals miss — [Source](https://developers.telnyx.com/docs/voice/stt/websocket-streaming/production-patterns)
- Keepalive: STT `keepalive` every 10 s; TTS ping every 20–30 s; load-balancer timeout must exceed ping interval + pong timeout (e.g. ≥60 s for 25 s + 20 s); RFC 6455 ping/pong detects NAT/proxy idle drops — [Source](https://docs.slng.ai/websocket-guide)
- Reconnect: bounded exponential backoff with jitter, reset after stable connection; do not retry auth/validation errors, fix key/params first; max ~3 attempts before permanent fail — [Source](https://deepgram.com/learn/nodejs-voice-ai-production-guide)
- Replay last ~5 s circular buffer on reconnect for continuity (~160 KB for 16 kHz mono 16-bit PCM); retransmit fast but ≤1.25× realtime; realign timestamps from new-session zero with maintained offset — [Source](https://deepgram.com/learn/nodejs-voice-ai-production-guide)
- Error classification: retry CONNECTION/NET (1001 going away, 1006 abnormal, 1012 restart, 1013 overload); fail fast DATA/PROTOCOL (1003 unsupported codec, 1007 invalid payload, 1009 message too big) — [Source](https://deepgram.com/learn/nodejs-voice-ai-production-guide)
- Cooperative backpressure: server acks chunks; client keeps ≤10 s in-flight unacked audio, refrains from sending until ack clears `sent` queue (FIFO queued→sent→acked) — [Source](https://docs.nabla.com/guides/best-practices/transcription-network-resilience/)
- Browser WebSocket has no `pause()`/built-in backpressure; app-layer bounded `asyncio.Queue(maxsize)` + explicit ACK/grant per sentence/chunk required; infinite client buffer grows `bufferedAmount` to GB — [Source](https://callsphere.ai/blog/vw1c-websocket-backpressure-audio-streams-flow-control)
- Interrupt rule: cancel upstream production before draining local buffer — drop, do not finish — [Source](https://callsphere.ai/blog/vw1c-websocket-backpressure-audio-streams-flow-control)

### Inferences
- Loopback FastAPI mapping: versioned path (e.g. `/v1/stream`) rejects mismatch with "reload demo" prompt; server owns 16 kHz resample + ring buffer so client sends native mono PCM chunks; no raw mic retained by default — show lock line "mic → memory only, nothing stored/uploaded".
- Status machine: `idle → requesting → listening → reconnecting → stopped` + `error(reason)`; each state has single recovery button (Enable mic / Retry / Reload / Stop); cooldown and Detection history pause during reconnect to avoid double-count.
- Backpressure UI: when `bufferedAmount` crosses threshold show "connection slow — holding Detections" and drop stale frames with rejected-log entry, never silent loss.

### Gaps
- No local-loopback-specific keepalive/idle-timeout numbers found; cloud 10–30 s values cited — needs tuning for FastAPI loopback (likely far more lenient) and verification that `bufferedAmount` polling at animation rate does not itself cause jank.
