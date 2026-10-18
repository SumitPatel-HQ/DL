// The in-browser live engine must match the API's Python pipeline.
// Fixture: `uv run --project backend python scripts/live_fixture.py`.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { LiveDecider, LiveStream } from "../lib/live/decider.ts";
import { mfcc } from "../lib/live/dsp.ts";

const fixture = JSON.parse(readFileSync(new URL("./fixtures/live-parity.json", import.meta.url), "utf8"));

// Same integer-only signal as scripts/live_fixture.py.
function signal(length) {
  let state = 12345;
  const pcm = new Int16Array(length);
  for (let n = 0; n < length; n++) {
    state = (state * 48271) % 2147483647;
    const t = n % 37;
    pcm[n] = (t < 19 ? t : 37 - t) * 800 - 7000 + (state % 2001) - 1000;
  }
  return pcm;
}

function maxError(actual, expected) {
  return expected.reduce((worst, value, i) => Math.max(worst, Math.abs(actual[i] - value)), 0);
}

for (const stream of fixture.streams) {
  test(`live windows and MFCCs match Python at ${stream.rate} Hz`, () => {
    const pcm = signal(stream.length);
    const live = new LiveStream(stream.rate, fixture.policy);
    for (let first = 0; first < pcm.length; first += 2048) {
      live.push(pcm.slice(first, first + 2048).buffer);
    }
    const batch = live.takeWindows();
    assert.equal(batch.windows.length, stream.windows.length);
    stream.windows.forEach((expected, i) => {
      assert.ok(Math.abs(batch.ends_s[i] - expected.end_s) < 1e-9);
      const window = batch.windows[i];
      const samples = expected.samples.map((_, k) => window[k * fixture.sample_stride]);
      assert.ok(maxError(samples, expected.samples) < 1e-6, "resampled window");
      const features = mfcc(window, fixture.mean, fixture.std);
      const strided = expected.features.map((_, k) => features[k * fixture.feature_stride]);
      assert.ok(maxError(strided, expected.features) < 1e-4, "normalized MFCCs");
    });
  });
}

test("live decisions match Python on the same score rows", () => {
  const actions = new Map([
    ["right", "next_slide"], ["left", "previous_slide"], ["go", "play"],
    ["stop", "pause"], ["up", "volume_up"], ["down", "volume_down"],
  ]);
  const decider = new LiveDecider(fixture.policy, fixture.labels, actions);
  const decisions = [];
  fixture.scores.forEach((row, index) => {
    for (const decision of decider.observe(index * fixture.policy.hop_s, Float32Array.from(row))) {
      decisions.push({ ...decision, index });
    }
  });
  assert.equal(decisions.length, fixture.decisions.length);
  decisions.forEach((decision, i) => {
    const expected = fixture.decisions[i];
    assert.deepEqual(
      [decision.index, decision.label, decision.accepted, decision.reason, decision.action],
      [expected.index, expected.label, expected.accepted, expected.reason, expected.action],
    );
    assert.ok(Math.abs(decision.score - expected.score) < 1e-6);
    assert.ok(Math.abs(decision.t_s - expected.t_s) < 1e-9);
  });
});
