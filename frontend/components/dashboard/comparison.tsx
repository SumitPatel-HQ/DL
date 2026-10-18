"use client";

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  api,
  errorCode,
  type ContinuousRun,
  type IsolatedRun,
  type RunDetail,
  type RunKind,
  type RunSummary,
} from "../../lib/api.ts";
import { comparisonGroups, type ComparisonGroup } from "../../lib/evaluations.ts";
import { formatNumber, shortHash } from "../../lib/format.ts";
import { ChartIcon, DatabaseIcon, TrashIcon } from "../icons";
import { EmptyState, ErrorBanner } from "../ui";
import { ConfusionMatrix, Legend, LineChart } from "./charts";
import { RunTrace } from "./views";

const SERIES_COLORS = ["var(--series-train)", "var(--series-f1)", "var(--series-val)"];

function GroupPicker({
  groups,
  selected,
  onSelect,
}: {
  groups: readonly ComparisonGroup[];
  selected: string | null;
  onSelect: (key: string) => void;
}) {
  return (
    <div className="control">
      <span className="label">Evaluation split</span>
      <div className="seg" role="group" aria-label="Evaluation split">
        {groups.map((group) => (
          <button
            key={group.key}
            type="button"
            aria-pressed={group.key === selected}
            title={`${group.runs[0].provenance.dataset_version} · ${group.runs[0].provenance.split_version}`}
            onClick={() => onSelect(group.key)}
          >
            {group.label}
          </button>
        ))}
      </div>
    </div>
  );
}

function RunPicker({
  candidates,
  selected,
  onToggle,
}: {
  candidates: readonly RunSummary[];
  selected: readonly string[];
  onToggle: (id: string) => void;
}) {
  return (
    <div className="control">
      <span className="label">Models · first selected is baseline</span>
      <div className="seg" role="group" aria-label="Models to compare">
        {candidates.map((r) => (
          <button
            key={r.id}
            type="button"
            aria-pressed={selected.includes(r.id)}
            onClick={() => onToggle(r.id)}
          >
            <b>{r.experiment_id}</b> <span className="mono">{shortHash(r.provenance.checkpoint_sha256, 8)}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

function CompareTile({ run, children }: { run: RunDetail; children: ReactNode }) {
  return (
    <div className="card tile">
      <span className="k">
        {run.experiment_id} · <code>{shortHash(run.provenance.checkpoint_sha256, 8)}</code>
      </span>
      {children}
      {run.provenance.checkpoint_dataset_version ? (
        <span className="d">trained on {run.provenance.checkpoint_dataset_version}</span>
      ) : null}
      <RunTrace run={run} />
    </div>
  );
}

export function ComparisonSection({
  runs,
  kind,
  activeId,
  online,
  onImport,
  busy,
}: {
  runs: readonly RunSummary[];
  kind: RunKind;
  /** Checkpoint the service runs; its runs lead each group and become the baseline. */
  activeId: string | null;
  online: boolean;
  onImport: () => void;
  busy: boolean;
}) {
  const groups = useMemo(() => comparisonGroups(runs, kind, activeId), [runs, kind, activeId]);
  const [groupKey, setGroupKey] = useState<string | null>(null);
  // Selected run IDs in group order; the first is the baseline.
  const [picked, setPicked] = useState<string[]>([]);
  // The default (every run of the first group) is set once per kind. It never
  // follows later list changes, so deleting a compared run clears to Not
  // evaluated with no cross-run fallback instead of silently switching runs.
  // The active checkpoint can arrive after the runs, so it is part of the lock.
  const lockedKind = useRef<string | null>(null);
  useEffect(() => {
    const lock = `${kind}|${activeId}`;
    if (lockedKind.current !== lock && groups.length > 0) {
      lockedKind.current = lock;
      setGroupKey(groups[0].key);
      setPicked(groups[0].runs.map((r) => r.id));
    }
    // Runs arrive asynchronously; this syncs state with that external update.
  }, [kind, activeId, groups]);

  const group = groups.find((g) => g.key === groupKey) ?? null;
  const [result, setResult] = useState<RunDetail[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const storedIds = useMemo(() => new Set(runs.map((r) => r.id)), [runs]);
  const deletedPick = picked.some((id) => !storedIds.has(id));
  const stale = result !== null && result.some((run) => !storedIds.has(run.id));
  const pickKey = picked.join("|");

  useEffect(() => {
    if (picked.length < 2 || !online || deletedPick) return;
    // Skip refetch when the loaded comparison already matches the picks.
    if (result?.map((r) => r.id).join("|") === pickKey) return;
    let cancelled = false;
    const [baseline, ...challengers] = picked;
    // The backend validates each pair against the baseline (same dataset/split).
    Promise.all(challengers.map((id) => api.compareEvaluations(baseline, id)))
      .then((comparisons) => {
        if (!cancelled) {
          setResult([comparisons[0].runs[0], ...comparisons.map((c) => c.runs[1])]);
          setError(null);
        }
      })
      .catch((fetchError: unknown) => {
        if (!cancelled) {
          setResult(null);
          setError(errorCode(fetchError));
        }
      });
    return () => {
      cancelled = true;
    };
    // `result` is read above only to skip redundant fetches; depending on it
    // would refetch after every load, so it stays out of the list.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pickKey, online, deletedPick]);

  if (!online) return null;

  if (groups.length === 0) {
    return (
      <section className="card state-panel">
        <EmptyState
          icon={<ChartIcon size={22} />}
          title="Not evaluated"
          actions={
            <button type="button" className="btn primary" onClick={onImport} disabled={busy}>
              <DatabaseIcon size={15} /> Import committed runs
            </button>
          }
        >
          No stored {kind === "isolated" ? "isolated classification" : "continuous-event"} runs share an
          Evaluation dataset/split version, so there is nothing fair to compare. Import the committed runs
          (E01, E02, LIVE3) to compare the LSTM checkpoints.
        </EmptyState>
      </section>
    );
  }

  const chooseGroup = (key: string) => {
    const next = groups.find((g) => g.key === key);
    setGroupKey(key);
    setPicked(next ? next.runs.map((r) => r.id) : []);
  };
  const toggle = (id: string) => {
    const order = group?.runs.map((r) => r.id) ?? [];
    const next = picked.includes(id) ? picked.filter((p) => p !== id) : [...picked, id];
    setPicked(order.filter((o) => next.includes(o)));
  };

  const tooFew = picked.length < 2;
  const visible = stale || tooFew || result?.map((r) => r.id).join("|") !== pickKey ? null : result;
  const pending = !deletedPick && !stale && !tooFew && visible === null && error === null;

  return (
    <div style={{ display: "grid", gap: 16 }}>
      <div className="controls">
        <GroupPicker groups={groups} selected={groupKey} onSelect={chooseGroup} />
        <RunPicker candidates={group?.runs ?? []} selected={picked} onToggle={toggle} />
      </div>

      {deletedPick || stale ? (
        <section className="card state-panel">
          <EmptyState icon={<TrashIcon size={22} />} title="Run deleted · comparison cleared">
            One of the compared runs was deleted, so every dependent chart was cleared. Nothing falls back to
            another run. Pick remaining runs above, or re-import the committed runs.
          </EmptyState>
        </section>
      ) : tooFew ? (
        <section className="card state-panel">
          <EmptyState icon={<ChartIcon size={22} />} title="Pick at least two runs">
            A comparison needs two or more stored runs on the same split. Select another model above.
          </EmptyState>
        </section>
      ) : error ? (
        <ErrorBanner code={error} handlers={{ retry: () => setError(null) }} />
      ) : pending ? (
        <div style={{ display: "grid", gap: 16 }} aria-busy="true">
          <div className="skeleton" style={{ height: 120 }} />
          <div className="skeleton" style={{ height: 320 }} />
        </div>
      ) : visible ? (
        visible[0].kind === "isolated" ? (
          <IsolatedComparison runs={visible as IsolatedRun[]} />
        ) : (
          <ContinuousComparison runs={visible as ContinuousRun[]} />
        )
      ) : null}
    </div>
  );
}

function mean(values: number[]): number {
  return values.length === 0 ? 0 : values.reduce((sum, v) => sum + v, 0) / values.length;
}

// Union of every run's labels, first run's order first, so no class is hidden.
function unionLabels(lists: readonly (readonly string[])[]): string[] {
  const labels: string[] = [];
  for (const list of lists) for (const label of list) if (!labels.includes(label)) labels.push(label);
  return labels;
}

function IsolatedComparison({ runs }: { runs: IsolatedRun[] }) {
  const [hover, setHover] = useState<number | null>(null);
  const [baseline] = runs;
  // Union of epochs so no run's curve is truncated to another's range.
  const epochs = [...new Set(runs.flatMap((run) => run.training.curves.map((c) => c.epoch)))].sort(
    (p, q) => p - q,
  );
  const curveSeries = (metric: "validation_macro_f1" | "validation_accuracy") =>
    runs
      .map((run, i) => {
        const byEpoch = new Map(run.training.curves.map((c) => [c.epoch, c]));
        return {
          key: run.id,
          label: run.experiment_id,
          color: SERIES_COLORS[i % SERIES_COLORS.length],
          values: epochs.map((e) => byEpoch.get(e)?.[metric] ?? NaN),
        };
      })
      .filter((s) => s.values.some(Number.isFinite));
  // Live checkpoints log validation accuracy but not validation macro-F1.
  const f1Series = curveSeries("validation_macro_f1");
  const accuracySeries = curveSeries("validation_accuracy");
  const missingF1 = runs.filter((run) => !f1Series.some((s) => s.key === run.id)).map((r) => r.experiment_id);
  const labels = unionLabels(runs.map((run) => run.metrics.per_class.map((c) => c.label)));
  const classes = runs.map((run) => new Map(run.metrics.per_class.map((c) => [c.label, c])));
  // Macro precision/recall are unweighted means over the stored per-class
  // values, exactly as macro-F1 is the mean of the stored per-class F1.
  const macro = (run: IsolatedRun) => ({
    precision: mean(run.metrics.per_class.map((c) => c.precision)),
    recall: mean(run.metrics.per_class.map((c) => c.recall)),
  });
  const delta = (value: number, base: number) => {
    const d = value - base;
    return `${d >= 0 ? "+" : "−"}${Math.abs(d).toFixed(3)}`;
  };
  const bestF1 = Math.max(...runs.map((r) => r.metrics.macro_f1));
  return (
    <>
      <section className="card">
        <div className="card-head">
          <div>
            <h3>Macro precision, recall, F1 · {baseline.provenance.split} split</h3>
            <p className="sub">
              {baseline.provenance.dataset_version} · {baseline.provenance.split_version} · from stored runs
              alone · Δ is against baseline {baseline.experiment_id}
            </p>
          </div>
        </div>
        <div className="tiles" style={{ padding: "0 16px 16px" }}>
          {runs.map((run, i) => {
            const m = macro(run);
            return (
              <CompareTile key={run.id} run={run}>
                <span className="v num">
                  {run.metrics.macro_f1.toFixed(3)}
                  {i > 0 ? <small> {delta(run.metrics.macro_f1, baseline.metrics.macro_f1)}</small> : null}
                </span>
                <span className="d">
                  macro-F1{run.metrics.macro_f1 === bestF1 ? " · highest of the compared runs" : ""}
                  {i === 0 ? " · baseline" : ""}
                </span>
                <span className="d">
                  macro precision <b className="num">{m.precision.toFixed(3)}</b> · macro recall{" "}
                  <b className="num">{m.recall.toFixed(3)}</b>
                </span>
                <span className="d">
                  accuracy {run.metrics.accuracy.toFixed(3)} · {formatNumber(run.metrics.examples)} labeled 1 s
                  clips
                </span>
              </CompareTile>
            );
          })}
        </div>
      </section>

      <section className="card">
        <div className="card-head">
          <div>
            <h3>Per-class precision, recall, F1</h3>
            <p className="sub">Union of every run&apos;s labels; support is the denominator for recall</p>
          </div>
        </div>
        <div className="table-wrap">
          <table className="data">
            <thead>
              <tr>
                <th scope="col">Class</th>
                {runs.map((run) => (
                  <th key={run.id} scope="col" className="right">
                    {run.experiment_id} P / R / F1
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {labels.map((label) => (
                <tr key={label}>
                  <td>{label}</td>
                  {classes.map((byLabel, i) => {
                    const c = byLabel.get(label);
                    return (
                      <td key={runs[i].id} className="right mono">
                        {c ? `${c.precision.toFixed(3)} / ${c.recall.toFixed(3)} / ${c.f1.toFixed(3)}` : "—"}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <RunTrace run={baseline} />
      </section>

      <div className="grid-2">
        <section className="card">
          <div className="card-head">
            <div>
              <h3>Training curves · validation macro-F1</h3>
              <p className="sub">
                Each run&apos;s own training log; best epochs {runs.map((r) => r.training.best_epoch).join(" / ")}
                {missingF1.length > 0 ? ` · not logged by ${missingF1.join(", ")}` : ""}
              </p>
            </div>
            <Legend series={f1Series} />
          </div>
          <div className="card-body">
            <LineChart
              title={`Validation macro-F1 per epoch for ${f1Series.map((s) => s.label).join(", ")}`}
              epochs={epochs}
              series={f1Series}
              domain={[0, 1]}
              format={(v) => v.toFixed(2)}
              hover={hover}
              onHover={setHover}
              bestEpoch={runs.map((r) => r.training.best_epoch)}
            />
          </div>
        </section>
        <section className="card">
          <div className="card-head">
            <div>
              <h3>Training curves · validation accuracy</h3>
              <p className="sub">
                Each checkpoint validates on its own training-time validation set, so these curves are not
                on one shared set; the tiles above are.
              </p>
            </div>
            <Legend series={accuracySeries} />
          </div>
          <div className="card-body">
            <LineChart
              title={`Validation accuracy per epoch for ${accuracySeries.map((s) => s.label).join(", ")}`}
              epochs={epochs}
              series={accuracySeries}
              domain={[0, 1]}
              format={(v) => v.toFixed(2)}
              hover={hover}
              onHover={setHover}
              bestEpoch={runs.map((r) => r.training.best_epoch)}
            />
          </div>
        </section>
      </div>

      <div className="grid-2">
        {runs.map((run) => (
          <section key={run.id} className="card">
            <div className="card-head">
              <div>
                <h3>Confusion matrix · {run.experiment_id}</h3>
                <p className="sub">Rows are actual classes</p>
              </div>
            </div>
            <div className="card-body">
              <ConfusionMatrix
                labels={run.metrics.confusion_matrix.labels}
                values={run.metrics.confusion_matrix.values}
                mode="share"
              />
            </div>
            <RunTrace run={run} />
          </section>
        ))}
      </div>
    </>
  );
}

function ContinuousComparison({ runs }: { runs: ContinuousRun[] }) {
  const [baseline] = runs;
  const perKeyword = runs.map((run) => new Map(run.metrics.per_keyword.map((k) => [k.keyword, k])));
  const keywords = unionLabels(runs.map((run) => run.metrics.per_keyword.map((k) => k.keyword)));
  const regions = [...new Set(runs.flatMap((run) => Object.keys(run.metrics.false_alarms.by_region)))].sort();
  return (
    <>
      <section className="card">
        <div className="card-head">
          <div>
            <h3>Event precision, recall, F1 · {baseline.provenance.split} split</h3>
            <p className="sub">
              Continuous runs never report accuracy; from stored runs alone. Continuous runs store no
              training curves or confusion matrices, so event outcomes and false alarms are the comparison.
            </p>
          </div>
        </div>
        <div className="tiles" style={{ padding: "0 16px 16px" }}>
          {runs.map((run) => (
            <CompareTile key={run.id} run={run}>
              <span className="v num">{run.metrics.f1.toFixed(3)}</span>
              <span className="d">
                event precision <b className="num">{run.metrics.precision.toFixed(3)}</b> · event recall{" "}
                <b className="num">{run.metrics.recall.toFixed(3)}</b>
              </span>
              <span className="d">
                <b className="num">{run.metrics.true_positive}</b> hits of{" "}
                <b className="num">{run.metrics.detections}</b> Detections ·{" "}
                {formatNumber(run.metrics.false_alarms.per_hour, 0)} false alarms/hour
              </span>
            </CompareTile>
          ))}
        </div>
      </section>
      <div className="grid-2">
        <section className="card">
          <div className="card-head">
            <div>
              <h3>Per-Keyword events</h3>
              <p className="sub">Union of every run&apos;s Keywords</p>
            </div>
          </div>
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr>
                  <th scope="col">Keyword</th>
                  {runs.map((run) => (
                    <th key={run.id} scope="col" className="right">
                      {run.experiment_id} P / R / F1
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {keywords.map((keyword) => (
                  <tr key={keyword}>
                    <td>{keyword}</td>
                    {perKeyword.map((byKeyword, i) => {
                      const k = byKeyword.get(keyword);
                      return (
                        <td key={runs[i].id} className="right mono">
                          {k ? `${k.precision.toFixed(3)} / ${k.recall.toFixed(3)} / ${k.f1.toFixed(3)}` : "—"}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <RunTrace run={baseline} />
        </section>
        <section className="card">
          <div className="card-head">
            <div>
              <h3>False alarms</h3>
              <p className="sub">{baseline.metrics.false_alarms.definition}</p>
            </div>
          </div>
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr>
                  <th scope="col">Negative region</th>
                  {runs.map((run) => (
                    <th key={run.id} scope="col" className="right">
                      {run.experiment_id} alarms / hour
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {regions.map((region) => (
                  <tr key={region}>
                    <td style={{ color: "var(--text)" }}>{region}</td>
                    {runs.map((run) => {
                      const r = run.metrics.false_alarms.by_region[region];
                      return (
                        <td key={run.id} className="right mono">
                          {r ? `${r.count} / ${formatNumber(r.per_hour, 0)}` : "—"}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <RunTrace run={runs[runs.length - 1]} />
        </section>
      </div>
    </>
  );
}
