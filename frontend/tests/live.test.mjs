import assert from "node:assert/strict";
import { test } from "node:test";
import { applyAction, initialStage, KEY_BINDINGS } from "../lib/live/stage.ts";
import {
  addHistory, IDLE, micErrorCode, parseServerMessage, sessionReducer,
} from "../lib/live/session.ts";

test("each action changes only the in-app stage and reports its result", () => {
  let stage = initialStage(3);
  let outcome = applyAction(stage, "previous_slide");
  assert.deepEqual([outcome.changed, outcome.result], [false, "Already on the first slide"]);
  outcome = applyAction(stage, "next_slide");
  assert.deepEqual([outcome.stage.slide, outcome.result], [2, "Slide 1 → 2"]);
  stage = applyAction(applyAction(stage, "play").stage, "volume_up").stage;
  assert.deepEqual([stage.playing, stage.volume], [true, 60]);
  assert.equal(applyAction(stage, "play").changed, false);
  assert.equal(applyAction(stage, "pause").result, "Paused");
  assert.equal(applyAction({ ...stage, volume: 100 }, "volume_up").result, "Volume already 100");
});

test("keyboard binds exactly one key to each action", () => {
  assert.deepEqual(
    Object.values(KEY_BINDINGS).sort(),
    ["next_slide", "pause", "play", "previous_slide", "volume_down", "volume_up"],
  );
});

test("server messages are validated; actions only ride accepted decisions", () => {
  const decision = {
    type: "decision", id: 1, t_s: 1.25, label: "go", score: 0.9, accepted: true,
    reason: "accepted", action: "play", processing_ms: 4,
  };
  assert.deepEqual(parseServerMessage(JSON.stringify(decision)), decision);
  for (const bad of [
    { ...decision, action: "shell" },
    { ...decision, accepted: false, reason: "unknown" },
    { ...decision, accepted: true, action: null },
    { ...decision, reason: "guess" },
  ]) {
    assert.equal(parseServerMessage(JSON.stringify(bad)), null);
  }
  assert.equal(parseServerMessage("not json"), null);
  assert.deepEqual(parseServerMessage(JSON.stringify({ type: "stopped", extra: 1 })), { type: "stopped" });
});

test("mic failures map to distinct recoverable states", () => {
  const error = (name) => Object.assign(new Error("x"), { name });
  assert.equal(micErrorCode(error("NotAllowedError"), "prompt"), "mic_denied");
  assert.equal(micErrorCode(error("NotAllowedError"), "denied"), "mic_blocked");
  assert.equal(micErrorCode(error("NotFoundError"), null), "no_device");
  assert.equal(micErrorCode(error("NotReadableError"), null), "device_busy");
  assert.equal(micErrorCode(error("SecurityError"), null), "insecure_context");
  assert.equal(micErrorCode(new Error("?"), null), "mic_failed");
});

test("session only listens after explicit start, mic grant, and server ready", () => {
  let state = sessionReducer(IDLE, { type: "ready", at: 1 });
  assert.equal(state.status, "idle");
  state = sessionReducer(state, { type: "start" });
  state = sessionReducer(state, { type: "mic_ready" });
  state = sessionReducer(state, { type: "ready", at: 5 });
  assert.deepEqual([state.status, state.startedAt], ["listening", 5]);
  state = sessionReducer(state, { type: "stopped", reason: "tab_hidden" });
  assert.deepEqual([state.status, state.stopReason], ["idle", "tab_hidden"]);
  state = sessionReducer(sessionReducer(state, { type: "start" }), { type: "failed", code: "mic_blocked" });
  assert.deepEqual([state.status, state.error], ["error", "mic_blocked"]);
  assert.equal(sessionReducer(state, { type: "stopped", reason: "x" }).error, "mic_blocked");
  assert.equal(sessionReducer(state, { type: "start" }).status, "requesting");
});

test("history keeps the 20 newest entries and collapses repeated Silence", () => {
  const entry = (i, reason = "unknown") => ({ key: String(i), reason });
  let history = [];
  for (let i = 0; i < 25; i++) history = addHistory(history, entry(i));
  assert.equal(history.length, 20);
  assert.equal(history[0].key, "24");
  history = addHistory(addHistory(history, entry("s1", "silence")), entry("s2", "silence"));
  assert.deepEqual(history.slice(0, 2).map((h) => h.key), ["s2", "24"]);
});
