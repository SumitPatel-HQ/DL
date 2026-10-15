# 06d: Live voice control session

Parent: #6 (06 Analysis, live, and dashboard UI)

**What to build:** The explicit mic demo where the six trained Keywords drive in-app slide and player actions with smoothing, dedup, and cooldown, while Unknown, Silence, and low-Confidence input are visibly rejected and nothing is retained.

**Blocked by:** None (can start immediately; ticket 05 persistence seam already provides the live-event table shape).

**Status:** implemented

- [x] Explicit Start opens the mic and live session; Stop, disconnect, or tab-hide immediately stops listening and closes the session with no background running and no raw mic audio stored
- [x] The six trained Keywords map to in-app slide and player actions only, with one utterance causing at most one action via smoothing and cooldown and an immediate stop path
- [x] Live status shows last Keyword, Confidence, accepted or rejected decision, and action result with history; Unknown, Silence, and uncertain input render as rejected
- [x] Mic-denied, no-device, disconnect, backpressure, and offline-backend states are explicit with per-state recovery; voice and keyboard drive the same in-app action path

Implementation: `/live` route, `frontend/components/live/`, and the live session API
in [backend-api.md](../../backend-api.md#live-voice-session). Decision policy and
WebSocket contract: `backend/tests/test_live.py`. Session state, action path, and
message validation: `frontend/tests/live.test.mjs`.
