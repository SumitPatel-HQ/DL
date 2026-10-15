import assert from "node:assert/strict";
import { afterEach, test } from "node:test";
import { getHealth } from "../lib/health.ts";

const originalBase = process.env.API_BASE_URL;
afterEach(() => {
  if (originalBase === undefined) delete process.env.API_BASE_URL;
  else process.env.API_BASE_URL = originalBase;
});

test("health client accepts both model states and rejects invalid contracts", async (t) => {
  process.env.API_BASE_URL = "http://127.0.0.1:8000";
  const payloads = [
    { status: "ok", model: { status: "unavailable", code: "worker_unavailable",
      setup_action: "Worker is waiting for metadata storage recovery." } },
    { status: "ok", model: { status: "ready", code: "model_ready" } },
    {
      status: "ok",
      model: {
        status: "unavailable",
        code: "model_incompatible",
        setup_action: "Install a matching checkpoint and frozen detector config.",
      },
    },
    {
      status: "ok",
      model: {
        status: "unavailable",
        code: "checkpoint_not_installed",
        setup_action: "Install a compatible checkpoint.",
      },
    },
  ];
  for (const payload of payloads) {
    t.mock.method(globalThis, "fetch", async (url) => {
      assert.equal(url.href, "http://127.0.0.1:8000/api/health");
      return Response.json(payload);
    });
    assert.deepEqual(await getHealth(), payload);
  }

  for (const payload of [
    null,
    {},
    { status: "ok", model: null },
    { status: "ok", model: { status: "ready", code: "checkpoint_not_installed" } },
    { status: "ok", model: { status: "unavailable", code: "model_ready" } },
    { status: "ok", model: { status: "unavailable", code: "checkpoint_not_installed" } },
  ]) {
    t.mock.method(globalThis, "fetch", async () => Response.json(payload));
    await assert.rejects(getHealth(), /Unexpected health response/);
  }
});

test("health client only connects to the endpoint served by the launcher", async (t) => {
  const fetch = t.mock.method(globalThis, "fetch", async () => {
    throw new Error("Unexpected network request");
  });
  for (const base of [
    "http://localhost:8000",
    "http://[::1]:8000",
    "http://127.0.0.1:8001",
    "https://127.0.0.1:8000",
    "http://example.com:8000",
  ]) {
    process.env.API_BASE_URL = base;
    await assert.rejects(getHealth(), /API_BASE_URL/);
  }
  assert.equal(fetch.mock.callCount(), 0);
});
