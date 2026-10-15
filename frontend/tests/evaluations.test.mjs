import assert from "node:assert/strict";
import { test } from "node:test";
import { comparableRuns, comparisonKey, modelOptions, niceTicks, rowNormalize, runsFor, targetView } from "../lib/evaluations.ts";

const run = (id, kind, checkpoint, split, experiment = "E02") => ({
  id,
  kind,
  experiment_id: experiment,
  created_at: "2026-10-15T11:00:00+00:00",
  source: `${id}.json`,
  provenance: { checkpoint_sha256: checkpoint, split },
  target: {},
});
const runs = [
  run("e01-val", "isolated", "aaa", "validation", "E01"),
  run("e02-val", "isolated", "bbb", "validation"),
  run("e02-test", "isolated", "bbb", "test"),
  run("e02-cont", "continuous", "bbb", "test"),
];

test("models list the active checkpoint first and count its runs", () => {
  assert.deepEqual(
    modelOptions(runs, "bbb").map((m) => [m.experiment, m.active, m.runs]),
    [["E02", true, 3], ["E01", false, 1]],
  );
});

test("runs never mix kinds or checkpoints and put the test split first", () => {
  assert.deepEqual(runsFor(runs, "isolated", "bbb").map((r) => r.id), ["e02-test", "e02-val"]);
  assert.deepEqual(runsFor(runs, "continuous", "aaa"), []);
  assert.deepEqual(runsFor(runs, "continuous", "bbb").map((r) => r.id), ["e02-cont"]);
});

test("confusion rows normalize to recall shares", () => {
  assert.deepEqual(rowNormalize([[3, 1], [0, 0]]), [[0.75, 0.25], [0, 0]]);
});

test("unmet targets report the delta, never a redefined goal", () => {
  const view = targetView({ metric: "event_f1", goal: 0.7, achieved: 0.01038961038961039, met: false, note: "" });
  assert.equal(view.status, "unmet");
  assert.ok(Math.abs(view.delta + 0.6896103896103896) < 1e-9);
  assert.match(view.label, /unmet: event-F1 ≥ 0.70/);
});

test("comparison pairs share kind and dataset/split, never a lone run", () => {
  const full = (id, kind, experiment, dataset, splitVersion, split) => ({
    id,
    kind,
    experiment_id: experiment,
    created_at: "2026-10-15T11:00:00+00:00",
    source: `${id}.json`,
    provenance: {
      checkpoint_sha256: experiment,
      dataset_version: dataset,
      split_version: splitVersion,
      split,
    },
    target: {},
  });
  const dataset = "speech-commands-v0.02-p0-subset-v1";
  const splits = "speaker-sha256-80-10-10-noise-source-round-robin-v1";
  const pair = [
    full("e01-test", "isolated", "E01", dataset, splits, "test"),
    full("e02-test", "isolated", "E02", dataset, splits, "test"),
    full("e01-val", "isolated", "E01", dataset, splits, "validation"),
    full("e02-cont", "continuous", "E02", "ami-natural-p0-v1", splits, "test"),
  ];
  assert.equal(comparisonKey(pair[0]), comparisonKey(pair[1]));
  assert.notEqual(comparisonKey(pair[0]), comparisonKey(pair[2]));
  assert.deepEqual(comparableRuns(pair, "isolated").map((r) => r.id), ["e01-test", "e02-test"]);
  assert.deepEqual(comparableRuns(pair, "continuous"), []);
});

test("axis ticks are clean and cover the range", () => {
  assert.deepEqual(niceTicks(0, 1, 5), [0, 0.25, 0.5, 0.75, 1]);
  const ticks = niceTicks(0.27, 1.83, 5);
  assert.ok(ticks[0] <= 0.27 && ticks.at(-1) >= 1.83);
});
