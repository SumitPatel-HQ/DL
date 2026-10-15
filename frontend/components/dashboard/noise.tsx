"use client";

import { useEffect, useMemo, useState, type ReactNode } from "react";
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
import { noisePairs, type NoisePair } from "../../lib/evaluations.ts";
import { formatNumber, shortHash } from "../../lib/format.ts";
import { ChartIcon, DatabaseIcon } from "../icons";
import { EmptyState, ErrorBanner } from "../ui";
import { RunTrace } from "./views";

function signed(value: number): string {
  return `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(3)}`;
}

function ConditionTile({ run, children }: { run: RunDetail; children: ReactNode }) {
  const noisy = run.provenance.condition === "noisy";
  return (
    <div className="card tile">
      <span className="k">
        {noisy ? "Noisy" : "Clean"} · {run.experiment_id} · <code>{shortHash(run.provenance.checkpoint_sha256, 8)}</code>
      </span>
      {children}
      <RunTrace run={run} />
    </div>
  );
}

function NoisePairView({ pair, online }: { pair: NoisePair; online: boolean }) {
  const [result, setResult] = useState<Comparison | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  const cleanId = pair.clean.id;
  const noisyId = pair.noisy.id;

  useEffect(() => {
    if (!online) return;
    let cancelled = false;
    api
      .compareEvaluations(cleanId, noisyId)
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
  }, [cleanId, noisyId, online, attempt]);

  if (error) return <ErrorBanner code={error} handlers={{ retry: () => setAttempt((n) => n + 1) }} />;
  const visible = result?.runs[0].id === cleanId && result?.runs[1].id === noisyId ? result : null;
  if (visible === null) {
    return (
      <div style={{ display: "grid", gap: 16 }} aria-busy="true">
        <div className="skeleton" style={{ height: 120 }} />
        <div className="skeleton" style={{ height: 240 }} />
      </div>
    );
  }
  const [clean, noisy] = visible.runs;
  const snr = noisy.provenance.snr_db;
  return (
    <>
      <section className="card">
        <div className="card-head">
          <div>
            <h3>Noise test · {visible.split} split</h3>
            <p className="sub">
              {visible.dataset_version} · {visible.split_version}
            </p>
            <p className="sub">
              Noise sources ({visible.split} split): {visible.noise_sources.join(", ") || "none recorded"}
              {snr != null ? ` · mixed at ${formatNumber(snr, 0)} dB SNR` : ""} · same checkpoint{" "}
              <code>{shortHash(clean.provenance.checkpoint_sha256, 8)}</code>, same protocol; only the dataset
              condition differs
            </p>
          </div>
        </div>
        {visible.kind === "isolated" ? (
          <IsolatedNoise clean={clean as IsolatedRun} noisy={noisy as IsolatedRun} />
        ) : (
          <ContinuousNoise clean={clean as ContinuousRun} noisy={noisy as ContinuousRun} />
        )}
      </section>
      {visible.kind === "isolated" ? (
        <IsolatedNoiseDetail clean={clean as IsolatedRun} noisy={noisy as IsolatedRun} />
      ) : (
        <ContinuousNoiseDetail clean={clean as ContinuousRun} noisy={noisy as ContinuousRun} />
      )}
    </>
  );
}

function IsolatedNoise({ clean, noisy }: { clean: IsolatedRun; noisy: IsolatedRun }) {
  return (
    <div className="tiles" style={{ padding: "0 16px 16px" }}>
      {[clean, noisy].map((run) => (
        <ConditionTile key={run.id} run={run}>
          <span className="v num">{run.metrics.macro_f1.toFixed(3)}</span>
          <span className="d">
            macro-F1
            {run === noisy ? (
              <>
                {" "}· Δ vs clean <b className="num">{signed(noisy.metrics.macro_f1 - clean.metrics.macro_f1)}</b>
              </>
            ) : null}
          </span>
          <span className="d">
            accuracy {run.metrics.accuracy.toFixed(3)} · {formatNumber(run.metrics.examples)} labeled 1 s clips
          </span>
        </ConditionTile>
      ))}
    </div>
  );
}

function IsolatedNoiseDetail({ clean, noisy }: { clean: IsolatedRun; noisy: IsolatedRun }) {
  const noisyClasses = new Map(noisy.metrics.per_class.map((c) => [c.label, c]));
  const cleanSilence = clean.metrics.silence_recall_by_source ?? {};
  const noisySilence = noisy.metrics.silence_recall_by_source ?? {};
  const sources = [...new Set([...Object.keys(cleanSilence), ...Object.keys(noisySilence)])].sort();
  const cell = (value: number | undefined) => (value === undefined ? "Not evaluated" : value.toFixed(3));
  return (
    <div className="grid-2">
      <section className="card">
        <div className="card-head">
          <div>
            <h3>Per-class precision, recall, F1 · clean vs noisy</h3>
            <p className="sub">Support is the denominator for recall</p>
          </div>
        </div>
        <div className="table-wrap">
          <table className="data">
            <thead>
              <tr>
                <th scope="col">Class</th>
                <th scope="col" className="right">Clean P / R / F1</th>
                <th scope="col" className="right">Noisy P / R / F1</th>
              </tr>
            </thead>
            <tbody>
              {clean.metrics.per_class.map((left) => {
                const right = noisyClasses.get(left.label);
                const fmt = (c: typeof left | undefined) =>
                  c ? `${c.precision.toFixed(3)} / ${c.recall.toFixed(3)} / ${c.f1.toFixed(3)}` : "—";
                return (
                  <tr key={left.label}>
                    <td>{left.label}</td>
                    <td className="right mono">{fmt(left)}</td>
                    <td className="right mono">{fmt(right)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <RunTrace run={clean} />
      </section>
      <section className="card">
        <div className="card-head">
          <div>
            <h3>Silence recall by noise source</h3>
            <p className="sub">
              Silence clips are the noise sources themselves, so they are not remixed in the noisy condition.
            </p>
          </div>
        </div>
        <div className="table-wrap">
          <table className="data">
            <thead>
              <tr>
                <th scope="col">Noise source</th>
                <th scope="col" className="right">Clean recall</th>
                <th scope="col" className="right">Noisy recall</th>
              </tr>
            </thead>
            <tbody>
              {sources.length === 0 ? (
                <tr>
                  <td colSpan={3}>Not evaluated: these runs store no per-source Silence recall.</td>
                </tr>
              ) : (
                sources.map((source) => (
                  <tr key={source}>
                    <td>{source}</td>
                    <td className="right mono">{cell(cleanSilence[source])}</td>
                    <td className="right mono">{cell(noisySilence[source])}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
        <RunTrace run={noisy} />
      </section>
    </div>
  );
}

function ContinuousNoise({ clean, noisy }: { clean: ContinuousRun; noisy: ContinuousRun }) {
  return (
    <div className="tiles" style={{ padding: "0 16px 16px" }}>
      {[clean, noisy].map((run) => (
        <ConditionTile key={run.id} run={run}>
          <span className="v num">{run.metrics.f1.toFixed(3)}</span>
          <span className="d">
            event F1 · precision <b className="num">{run.metrics.precision.toFixed(3)}</b> · recall{" "}
            <b className="num">{run.metrics.recall.toFixed(3)}</b>
          </span>
          <span className="d">
            <b className="num">{run.metrics.true_positive}</b> hits of{" "}
            <b className="num">{run.metrics.detections}</b> Detections ·{" "}
            {formatNumber(run.metrics.false_alarms.per_hour, 0)} false alarms/hour
          </span>
        </ConditionTile>
      ))}
    </div>
  );
}

function ContinuousNoiseDetail({ clean, noisy }: { clean: ContinuousRun; noisy: ContinuousRun }) {
  const noisyKeywords = new Map(noisy.metrics.per_keyword.map((k) => [k.keyword, k]));
  return (
    <section className="card">
      <div className="card-head">
        <div>
          <h3>Per-Keyword events · clean vs noisy</h3>
          <p className="sub">
            Experimental window Detector at the frozen ±{clean.detector.tolerance_s} s tolerance. Unlabeled gaps
            are excluded from negative exposure ({formatNumber(clean.composition.unlabeled_seconds, 0)} s here).
          </p>
        </div>
      </div>
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              <th scope="col">Keyword</th>
              <th scope="col" className="right">Clean P / R / F1</th>
              <th scope="col" className="right">Noisy P / R / F1</th>
            </tr>
          </thead>
          <tbody>
            {clean.metrics.per_keyword.map((left) => {
              const right = noisyKeywords.get(left.keyword);
              const fmt = (k: typeof left | undefined) =>
                k ? `${k.precision.toFixed(3)} / ${k.recall.toFixed(3)} / ${k.f1.toFixed(3)}` : "—";
              return (
                <tr key={left.keyword}>
                  <td>{left.keyword}</td>
                  <td className="right mono">{fmt(left)}</td>
                  <td className="right mono">{fmt(right)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <RunTrace run={noisy} />
    </section>
  );
}

export function NoiseComparisonSection({
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
  const pairs = useMemo(() => noisePairs(runs, kind), [runs, kind]);
  if (!online) return null;
  if (pairs.length === 0) {
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
          No stored {kind === "isolated" ? "isolated classification" : "continuous-event"} clean and noisy pair
          shares a checkpoint and dataset/split version. Import the committed noise runs, or rerun both
          conditions; nothing falls back to another run.
        </EmptyState>
      </section>
    );
  }
  return (
    <div style={{ display: "grid", gap: 16 }}>
      {pairs.map((pair) => (
        <NoisePairView key={`${pair.clean.id}::${pair.noisy.id}`} pair={pair} online={online} />
      ))}
    </div>
  );
}
