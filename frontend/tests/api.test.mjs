import assert from "node:assert/strict";
import { test } from "node:test";
import { api, ApiError, request } from "../lib/api.ts";
import { describeError } from "../lib/errors.ts";

test("requests map API error bodies, 204s, and network failure to stable codes", async (t) => {
  t.mock.method(globalThis, "fetch", async (url) => {
    assert.equal(url, "http://127.0.0.1:8000/api/recordings/x");
    return Response.json({ error: { code: "recording_not_found", message: "gone" } }, { status: 404 });
  });
  await assert.rejects(api.recording("x"), (error) =>
    error instanceof ApiError && error.code === "recording_not_found" && error.status === 404);

  t.mock.method(globalThis, "fetch", async () => new Response(null, { status: 204 }));
  assert.equal(await api.deleteRecording("x"), undefined);

  t.mock.method(globalThis, "fetch", async () => {
    throw new TypeError("Failed to fetch");
  });
  await assert.rejects(request("/api/health"), (error) => error.code === "offline");

  t.mock.method(globalThis, "fetch", async () => new Response("<html>", { status: 502 }));
  await assert.rejects(request("/api/health"), (error) => error.code === "http_502");
});

test("every backend and live error code has recovery copy", () => {
  for (const code of [
    "invalid_wav", "unsupported_wav", "upload_too_large", "recording_too_long", "queue_full",
    "worker_unavailable", "offline", "analysis_in_progress", "delete_failed", "evaluation_not_found",
    "mic_denied", "mic_blocked", "no_device", "device_busy", "live_busy", "disconnected",
  ]) {
    assert.notEqual(describeError(code).title, "Unexpected error", code);
  }
  assert.equal(describeError("mystery").recovery, "retry");
});
