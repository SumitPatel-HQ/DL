import assert from "node:assert/strict";
import { test } from "node:test";
import {
  assignLanes, detectionAt, filterDetections, neighbor, segment, sortDetections,
} from "../lib/detections.ts";
import { phaseOf, pollDelay } from "../lib/analysis.ts";

const d = (id, keyword, start_s, end_s, confidence) => ({
  id, keyword, start_s, end_s, confidence, threshold: 0.5, windows: 1,
});
const list = [d(1, "go", 2, 3, 0.9), d(2, "stop", 2.4, 3.4, 0.6), d(3, "go", 5, 6.2, 0.75)];

test("sorting is numeric, stable, and supports Confidence triage", () => {
  assert.deepEqual(sortDetections(list, "confidence", "descending").map((x) => x.id), [1, 3, 2]);
  assert.deepEqual(sortDetections(list, "start", "descending").map((x) => x.id), [3, 2, 1]);
  assert.deepEqual(sortDetections(list, "duration", "ascending").map((x) => x.id), [1, 2, 3]);
  assert.deepEqual(sortDetections(list, "keyword", "ascending").map((x) => x.id), [1, 3, 2]);
});

test("Keyword filter keeps only selected Keywords", () => {
  assert.deepEqual(filterDetections(list, new Set(["stop"])).map((x) => x.id), [2]);
});

test("overlapping regions get separate lanes", () => {
  const { lanes, count } = assignLanes(list);
  assert.equal(count, 2);
  assert.deepEqual([lanes.get(1), lanes.get(2), lanes.get(3)], [0, 1, 0]);
});

test("clock, region, and row resolve to the same Detection", () => {
  assert.equal(detectionAt(list, 2.1)?.id, 1);
  assert.equal(detectionAt(list, 2.5)?.id, 2);
  assert.equal(detectionAt(list, 4), null);
  assert.equal(neighbor(list, 0, 1)?.id, 1);
  assert.equal(neighbor(list, 2, 1)?.id, 2);
  assert.equal(neighbor(list, 5.1, -1)?.id, 2);
  assert.equal(neighbor(list, 6, 1), null);
  assert.deepEqual(segment(list[0], 10), { from: 1.8, to: 3 });
  assert.deepEqual(segment(d(9, "up", 0.1, 1.1, 1), 1), { from: 0, to: 1 });
});

test("job phases distinguish completed, empty, and failed", () => {
  const recording = (status) => ({ status });
  const result = (detections) => ({ status: "completed", detections });
  assert.equal(phaseOf(recording("queued"), null), "queued");
  assert.equal(phaseOf(recording("completed"), result([])), "empty");
  assert.equal(phaseOf(recording("completed"), result(list)), "completed");
  assert.equal(phaseOf(recording("failed"), null), "failed");
  assert.ok(pollDelay(0) < pollDelay(5));
  assert.equal(pollDelay(50), 3000);
});
