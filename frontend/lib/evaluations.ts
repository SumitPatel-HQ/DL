// Dashboard view logic. Every value comes from one stored Evaluation Run;
// nothing here averages, merges, or falls back across runs.
import type { RunKind, RunSummary, Target } from "./api.ts";

export type ModelOption = {
  checkpoint: string;
  experiment: string;
  active: boolean;
  runs: number;
};

// Checkpoints that have at least one stored run, active model first.
export function modelOptions(runs: readonly RunSummary[], activeId: string | null): ModelOption[] {
  const byCheckpoint = new Map<string, ModelOption>();
  for (const run of runs) {
    const checkpoint = run.provenance.checkpoint_sha256;
    const option = byCheckpoint.get(checkpoint) ?? {
      checkpoint,
      experiment: run.experiment_id,
      active: checkpoint === activeId,
      runs: 0,
    };
    option.runs += 1;
    byCheckpoint.set(checkpoint, option);
  }
  return [...byCheckpoint.values()].sort(
    (a, b) => Number(b.active) - Number(a.active) || a.experiment.localeCompare(b.experiment),
  );
}

const SPLIT_ORDER = ["test", "validation"];

// Runs of one kind for one checkpoint, held-out test split first.
export function runsFor(
  runs: readonly RunSummary[],
  kind: RunKind,
  checkpoint: string | null,
): RunSummary[] {
  const rank = (split: string) => {
    const index = SPLIT_ORDER.indexOf(split);
    return index === -1 ? SPLIT_ORDER.length : index;
  };
  return runs
    .filter((run) => run.kind === kind && run.provenance.checkpoint_sha256 === checkpoint)
    .sort(
      (a, b) =>
        rank(a.provenance.split) - rank(b.provenance.split) ||
        b.created_at.localeCompare(a.created_at),
    );
}

// Comparison key: two runs compare only under identical kind + dataset/split.
// Cross-checkpoint pairs (E01 vs E02) share this key; kind or dataset drift does not.
export function comparisonKey(run: RunSummary): string {
  const p = run.provenance;
  return `${run.kind}::${p.dataset_version}::${p.split_version}::${p.split}::${conditionOf(run)}`;
}

// Runs of one kind that share a comparison key with at least one other run.
export function comparableRuns(runs: readonly RunSummary[], kind: RunKind): RunSummary[] {
  const counts = new Map<string, number>();
  for (const run of runs) {
    if (run.kind !== kind) continue;
    const key = comparisonKey(run);
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return runs
    .filter((run) => run.kind === kind && (counts.get(comparisonKey(run)) ?? 0) > 1)
    .sort(
      (a, b) =>
        a.experiment_id.localeCompare(b.experiment_id) || a.created_at.localeCompare(b.created_at),
    );
}

// Pairing key: comparisonKey without the condition, plus the checkpoint.
function noisePairKey(run: RunSummary): string {
  return `${run.provenance.checkpoint_sha256}::${comparisonKey(run).replace(/::(clean|noisy)$/, "")}`;
}

export function conditionOf(run: RunSummary): "clean" | "noisy" {
  return run.provenance.condition ?? "clean";
}

export type NoisePair = { clean: RunSummary; noisy: RunSummary };

// One clean + one noisy run of one kind that share checkpoint and dataset/split
// versions. A group missing either side yields no pair: the view reads Not
// evaluated, and nothing substitutes another checkpoint or split.
export function noisePairs(runs: readonly RunSummary[], kind: RunKind): NoisePair[] {
  const groups = new Map<string, { clean?: RunSummary; noisy?: RunSummary }>();
  for (const run of runs) {
    if (run.kind !== kind) continue;
    const key = noisePairKey(run);
    const group = groups.get(key) ?? {};
    const side = conditionOf(run);
    const current = group[side];
    if (!current || run.created_at > current.created_at) group[side] = run;
    groups.set(key, group);
  }
  return [...groups.values()]
    .filter((g): g is NoisePair => g.clean !== undefined && g.noisy !== undefined)
    .sort((a, b) => a.clean.experiment_id.localeCompare(b.clean.experiment_id));
}

// Each row becomes "of the actual class, where did detections go".
export function rowNormalize(values: readonly (readonly number[])[]): number[][] {
  return values.map((row) => {
    const total = row.reduce((sum, count) => sum + count, 0);
    return row.map((count) => (total === 0 ? 0 : count / total));
  });
}

export type TargetView = {
  status: "met" | "unmet";
  delta: number;
  label: string;
};

export function targetView(target: Target): TargetView {
  const delta = target.achieved - target.goal;
  const metric = target.metric === "macro_f1" ? "macro-F1" : "event-F1";
  return {
    status: target.met ? "met" : "unmet",
    delta,
    label: target.met
      ? `Target met: ${metric} ≥ ${target.goal.toFixed(2)}`
      : `Target unmet: ${metric} ≥ ${target.goal.toFixed(2)}`,
  };
}

// Clean axis ticks covering [min, max].
export function niceTicks(min: number, max: number, count = 5): number[] {
  if (!(max > min)) return [min];
  const raw = (max - min) / Math.max(1, count - 1);
  const power = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * power).find((s) => s >= raw) ?? raw;
  const first = Math.floor(min / step) * step;
  const ticks: number[] = [];
  for (let tick = first; tick <= max + step * 1e-9; tick += step) {
    ticks.push(Number(tick.toFixed(10)));
  }
  if (ticks[ticks.length - 1] < max) ticks.push(Number((ticks[ticks.length - 1] + step).toFixed(10)));
  return ticks;
}
