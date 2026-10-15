"use client";

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  api,
  errorCode,
  type Comparison,
  type ContinuousRun,
  type IsolatedRun,
  type RunDetail,
  type RunKind,
  type RunSummary,
} from "../../lib/api.ts";
import { comparableRuns, comparisonKey } from "../../lib/evaluations.ts";
import { formatNumber, shortHash } from "../../lib/format.ts";
import { ChartIcon, DatabaseIcon, TrashIcon } from "../icons";
import { EmptyState, ErrorBanner } from "../ui";
import { ConfusionMatrix, Legend, LineChart } from "./charts";
import { RunTrace } from "./views";

function RunPicker({
  label,
  candidates,
  selected,
  onSelect,
}: {
  label: string;
  candidates: readonly RunSummary[];
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="control">
      <span className="label">{label}</span>
      <div className="run-row" role="group" aria-label={label}>
        {candidates.map((r) => (
          <button
            key={r.id}
            type="button"
            className="run-chip"
            aria-pressed={r.id === selected}
            onClick={() => onSelect(r.id)}
          >
            <b>{r.experiment_id}</b>
            <span className="mono">{shortHash(r.provenance.checkpoint_sha256, 8)}</span>
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
      <RunTrace run={run} />
    </div>
  );
}

export function ComparisonSection({
  runs,
  kind,
  online,
  onImport,
  busy,
}: {
  runs: readonly RunSummary[];
  kind: RunKind;
  online: boolean;
  onImport: () => void;
  busy: boolean;
}) {
  const candidates = useMemo(() => comparableRuns(runs, kind), [runs, kind]);
  // Explicit user picks; null means "follow the default pair".
  const [pickA, setPickA] = useState<string | null>(null);
  const [pickB, setPickB] = useState<string | null>(null);
  // Default pair (E01 vs E02), locked once the runs arrive. The lock never
  // follows later list changes, so deleting a compared run clears to Not
  // evaluated with no cross-run fallback instead of silently switching pairs.
  const [locked, setLocked] = useState<{ a: string; b: string } | null>(null);
  const lockedKind = useRef<RunKind | null>(null);
  useEffect(() => {
    if (lockedKind.current !== kind && candidates.length >= 2) {
      lockedKind.current = kind;
      const first = candidates[0];
      const match =
        candidates.find((r) => r.id !== first.id && comparisonKey(r) === comparisonKey(first)) ??
        candidates[1];
      setPickA(first.id);
      setPickB(match.id);
      setLocked({ a: first.id, b: match.id });
    }
    // Runs arrive asynchronously; this syncs state with that external update.
  }, [kind, candidates]);

  const idA = pickA ?? locked?.a ?? null;
  const idB = pickB ?? locked?.b ?? null;
  const [result, setResult] = useState<Comparison | null>(null);
  const [error, setError] = useState<string | null>(null);

  const storedIds = useMemo(() => new Set(runs.map((r) => r.id)), [runs]);
  // A loaded comparison whose runs are no longer stored is stale: clear the
  // numbers without falling back to another run.
  const stale =
    result !== null && (!storedIds.has(result.runs[0].id) || !storedIds.has(result.runs[1].id));
  const deletedPick =
    (idA !== null && !storedIds.has(idA)) || (idB !== null && !storedIds.has(idB));

  useEffect(() => {
    if (!idA || !idB || !online || storedIds.size === 0) return;
    if (!storedIds.has(idA) || !storedIds.has(idB)) return;
    if (idA === idB) return;
    // Skip refetch when the loaded comparison already matches the picks.
    if (result?.runs[0].id === idA && result?.runs[1].id === idB) return;
    let cancelled = false;
    api
      .compareEvaluations(idA, idB)
      .then((comparison) => {
        if (!cancelled) {
          setResult(comparison);
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
  }, [idA, idB, online, storedIds]);

  if (!online) return null;

  const samePick = idA !== null && idA === idB;
  const matchesPicks = result?.runs[0].id === idA && result?.runs[1].id === idB;
  const visible = stale || samePick || !matchesPicks ? null : result;
  const pending =
    !deletedPick && !stale && !samePick && idA !== null && idB !== null && visible === null && error === null;

  if (candidates.length < 2) {
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
          No stored {kind === "isolated" ? "isolated classification" : "continuous-event"} pair shares an
          Evaluation dataset/split version, so there is nothing fair to compare. Import the committed runs
          (E01 vs E02) to compare baseline LSTM variants.
        </EmptyState>
      </section>
    );
  }

  return (
    <div style={{ display: "grid", gap: 16 }}>
      <div className="controls">
        <RunPicker label="Run A (baseline)" candidates={candidates} selected={idA} onSelect={setPickA} />
        <RunPicker label="Run B (challenger)" candidates={candidates} selected={idB} onSelect={setPickB} />
      </div>

      {deletedPick || stale ? (
        <section className="card state-panel">
          <EmptyState icon={<TrashIcon size={22} />} title="Run deleted · comparison cleared">
            One of the compared runs was deleted, so every dependent chart was cleared. Nothing falls back to
            another run. Pick two remaining runs above, or re-import the committed runs.
          </EmptyState>
        </section>
      ) : samePick ? (
        <section className="card state-panel">
          <EmptyState icon={<ChartIcon size={22} />} title="Pick two different runs">
            A comparison needs two stored run IDs. Choose a different Run B to compare against Run A.
          </EmptyState>
        </section>
      ) : error && !samePick ? (
        <ErrorBanner code={error} handlers={{ retry: () => setError(null) }} />
      ) : pending ? (
        <div style={{ display: "grid", gap: 16 }} aria-busy="true">
          <div className="skeleton" style={{ height: 120 }} />
          <div className="skeleton" style={{ height: 320 }} />
        </div>
      ) : visible ? (
        visible.kind === "isolated" ? (
          <IsolatedComparison a={visible.runs[0] as IsolatedRun} b={visible.runs[1] as IsolatedRun} />
        ) : (
          <ContinuousComparison a={visible.runs[0] as ContinuousRun} b={visible.runs[1] as ContinuousRun} />
        )
      ) : null}
    </div>
  );
}

function mean(values: number[]): number {
  return values.length === 0 ? 0 : values.reduce((sum, v) => sum + v, 0) / values.length;
}

function IsolatedComparison({ a, b }: { a: IsolatedRun; b: IsolatedRun }) {
  const [hover, setHover] = useState<number | null>(null);
  // Union of epochs so neither run's curve is truncated to the other's range.
  const epochs = [...new Set([...a.training.curves, ...b.training.curves].map((c) => c.epoch))].sort(
    (p, q) => p - q,
  );
  const aByEpoch = new Map(a.training.curves.map((c) => [c.epoch, c]));
  const bByEpoch = new Map(b.training.curves.map((c) => [c.epoch, c]));
  const f1Series = [
    {
      key: "a",
      label: `${a.experiment_id} validation macro-F1`,
      color: "var(--series-train)",
      values: epochs.map((e) => aByEpoch.get(e)?.validation_macro_f1 ?? NaN),
    },
    {
      key: "b",
      label: `${b.experiment_id} validation macro-F1`,
      color: "var(--series-f1)",
      values: epochs.map((e) => bByEpoch.get(e)?.validation_macro_f1 ?? NaN),
    },
  ].filter((s) => s.values.some((v) => !Number.isNaN(v)));
  // Union of labels so neither run's classes are hidden; A order first.
  const labels = [...a.metrics.per_class.map((c) => c.label)];
  for (const c of b.metrics.per_class) {
    if (!labels.includes(c.label)) labels.push(c.label);
  }
  const aClasses = new Map(a.metrics.per_class.map((c) => [c.label, c]));
  const bClasses = new Map(b.metrics.per_class.map((c) => [c.label, c]));
  // Macro precision/recall are unweighted means over the stored per-class
  // values, exactly as macro-F1 is the mean of the stored per-class F1.
  const macro = (run: IsolatedRun) => ({
    precision: mean(run.metrics.per_class.map((c) => c.precision)),
    recall: mean(run.metrics.per_class.map((c) => c.recall)),
  });
  const macroA = macro(a);
  const macroB = macro(b);
  return (
    <>
      <section className="card">
        <div className="card-head">
          <div>
            <h3>Macro precision, recall, F1 · {a.provenance.split} split</h3>
            <p className="sub">
              {a.provenance.dataset_version} · {a.provenance.split_version} · from stored runs alone
            </p>
          </div>
        </div>
        <div className="tiles" style={{ padding: "0 16px 16px" }}>
          {(
            [
              { run: a, macro: macroA },
              { run: b, macro: macroB },
            ] as const
          ).map(({ run, macro: m }) => (
            <CompareTile key={run.id} run={run}>
              <span className="v num">{run.metrics.macro_f1.toFixed(3)}</span>
              <span className="d">
                macro precision <b className="num">{m.precision.toFixed(3)}</b> · macro recall{" "}
                <b className="num">{m.recall.toFixed(3)}</b>
              </span>
              <span className="d">
                accuracy {run.metrics.accuracy.toFixed(3)} · {formatNumber(run.metrics.examples)} labeled
                1 s clips
              </span>
            </CompareTile>
          ))}
        </div>
      </section>

      <div className="grid-2">
        <section className="card">
          <div className="card-head">
            <div>
              <h3>Per-class precision, recall, F1 · A vs B</h3>
              <p className="sub">Union of both runs&apos; labels; support is the denominator for recall</p>
            </div>
          </div>
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr>
                  <th scope="col">Class</th>
                  <th scope="col" className="right">
                    {a.experiment_id} P / R / F1
                  </th>
                  <th scope="col" className="right">
                    {b.experiment_id} P / R / F1
                  </th>
                </tr>
              </thead>
              <tbody>
                {labels.map((label) => {
                  const left = aClasses.get(label);
                  const right = bClasses.get(label);
                  const cell = (c: typeof left) =>
                    c ? `${c.precision.toFixed(3)} / ${c.recall.toFixed(3)} / ${c.f1.toFixed(3)}` : "—";
                  return (
                    <tr key={label}>
                      <td>{label}</td>
                      <td className="right mono">{cell(left)}</td>
                      <td className="right mono">{cell(right)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <RunTrace run={a} />
        </section>
        <section className="card">
          <div className="card-head">
            <div>
              <h3>Training curves · validation macro-F1</h3>
              <p className="sub">
                Overlaid from stored runs; best epochs {a.training.best_epoch} vs {b.training.best_epoch}
              </p>
            </div>
            <Legend series={f1Series} />
          </div>
          <div className="card-body">
            <LineChart
              title={`Validation macro-F1 per epoch for ${a.experiment_id} and ${b.experiment_id}`}
              epochs={epochs}
              series={f1Series}
              domain={[0, 1]}
              format={(v) => v.toFixed(2)}
              hover={hover}
              onHover={setHover}
              bestEpoch={[a.training.best_epoch, b.training.best_epoch]}
            />
          </div>
          <RunTrace run={b} />
        </section>
      </div>

      <div className="grid-2">
        {([a, b] as IsolatedRun[]).map((run) => (
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

function ContinuousComparison({ a, b }: { a: ContinuousRun; b: ContinuousRun }) {
  const aKeywords = new Map(a.metrics.per_keyword.map((k) => [k.keyword, k]));
  const bKeywords = new Map(b.metrics.per_keyword.map((k) => [k.keyword, k]));
  const keywords = [...aKeywords.keys()];
  for (const keyword of bKeywords.keys()) {
    if (!keywords.includes(keyword)) keywords.push(keyword);
  }
  const regions = [...new Set([...Object.keys(a.metrics.false_alarms.by_region), ...Object.keys(b.metrics.false_alarms.by_region)])].sort();
  return (
    <>
      <section className="card">
        <div className="card-head">
          <div>
            <h3>Event precision, recall, F1 · {a.provenance.split} split</h3>
            <p className="sub">
              Continuous runs never report accuracy; from stored runs alone. Continuous runs store no
              training curves or confusion matrices, so event outcomes and false alarms are the comparison.
            </p>
          </div>
        </div>
        <div className="tiles" style={{ padding: "0 16px 16px" }}>
          {[a, b].map((run) => (
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
              <h3>Per-Keyword events · A vs B</h3>
              <p className="sub">Union of both runs&apos; Keywords</p>
            </div>
          </div>
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr>
                  <th scope="col">Keyword</th>
                  <th scope="col" className="right">
                    {a.experiment_id} P / R / F1
                  </th>
                  <th scope="col" className="right">
                    {b.experiment_id} P / R / F1
                  </th>
                </tr>
              </thead>
              <tbody>
                {keywords.map((keyword) => {
                  const left = aKeywords.get(keyword);
                  const right = bKeywords.get(keyword);
                  const cell = (k: typeof left) =>
                    k ? `${k.precision.toFixed(3)} / ${k.recall.toFixed(3)} / ${k.f1.toFixed(3)}` : "—";
                  return (
                    <tr key={keyword}>
                      <td>{keyword}</td>
                      <td className="right mono">{cell(left)}</td>
                      <td className="right mono">{cell(right)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <RunTrace run={a} />
        </section>
        <section className="card">
          <div className="card-head">
            <div>
              <h3>False alarms · A vs B</h3>
              <p className="sub">{a.metrics.false_alarms.definition}</p>
            </div>
          </div>
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr>
                  <th scope="col">Negative region</th>
                  <th scope="col" className="right">
                    {a.experiment_id} alarms / hour
                  </th>
                  <th scope="col" className="right">
                    {b.experiment_id} alarms / hour
                  </th>
                </tr>
              </thead>
              <tbody>
                {regions.map((region) => {
                  const left = a.metrics.false_alarms.by_region[region];
                  const right = b.metrics.false_alarms.by_region[region];
                  return (
                    <tr key={region}>
                      <td style={{ color: "var(--text)" }}>{region}</td>
                      <td className="right mono">
                        {left ? `${left.count} / ${formatNumber(left.per_hour, 0)}` : "—"}
                      </td>
                      <td className="right mono">
                        {right ? `${right.count} / ${formatNumber(right.per_hour, 0)}` : "—"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <RunTrace run={b} />
        </section>
      </div>
    </>
  );
}
