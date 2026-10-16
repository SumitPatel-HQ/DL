"use client";

import { useState, type CSSProperties, type ReactNode } from "react";
import type { ContinuousRun, IsolatedRun, RunSummary, Target } from "../../lib/api.ts";
import { targetView } from "../../lib/evaluations.ts";
import { formatNumber, formatPercent, formatTimestamp, shortHash } from "../../lib/format.ts";
import { AlertIcon, CheckIcon } from "../icons";
import { Banner, KeywordName } from "../ui";
import { ConfusionMatrix, Legend, LineChart, type Series } from "./charts";

function Copy({ value }: { value: string }) {
  const [done, setDone] = useState(false);
  return (
    <button
      type="button"
      className="copy-btn"
      onClick={() => {
        void navigator.clipboard?.writeText(value).then(() => {
          setDone(true);
          window.setTimeout(() => setDone(false), 1200);
        });
      }}
      aria-label={`Copy ${value}`}
    >
      {done ? "copied" : "copy"}
    </button>
  );
}

export function ProvenancePanel({ run }: { run: RunSummary }) {
  const p = run.provenance;
  return (
    <section className="prov" aria-label="Run provenance">
      <div>
        <span className="k">Evaluation Run</span>
        <span className="v"><code title={run.id}>{run.id}</code><Copy value={run.id} /></span>
      </div>
      <div>
        <span className="k">Checkpoint SHA-256</span>
        <span className="v"><code title={p.checkpoint_sha256}>{shortHash(p.checkpoint_sha256, 16)}…</code><Copy value={p.checkpoint_sha256} /></span>
      </div>
      <div>
        <span className="k">Dataset version</span>
        <span className="v" title={p.dataset_version}>{p.dataset_version}</span>
      </div>
      <div>
        <span className="k">Split</span>
        <span className="v" title={p.split_version}><b>{p.split}</b> · {p.split_version}</span>
      </div>
      <div>
        <span className="k">{p.config_role === "detector" ? "Frozen detector config" : "Training config"}</span>
        <span className="v"><code title={p.config_sha256}>{shortHash(p.config_sha256, 16)}…</code><Copy value={p.config_sha256} /></span>
      </div>
      <div>
        <span className="k">Code revision</span>
        <span className="v">
          <code>{p.code_revision.commit ? shortHash(p.code_revision.commit, 10) : "unknown"}</code>
          {p.code_revision.dirty ? <span className="tag warn" title="Uncommitted changes existed when this run was produced">dirty tree</span> : null}
        </span>
      </div>
      <div>
        <span className="k">Created</span>
        <span className="v">{formatTimestamp(run.created_at)}</span>
      </div>
      <div>
        <span className="k">Source file</span>
        <span className="v" title={run.source}><code>{run.source}</code></span>
      </div>
    </section>
  );
}

export function RunTrace({ run }: { run: RunSummary }) {
  return (
    <div className="card-foot">
      <span>From run <code>{run.id}</code></span>
      <span>ckpt <code>{shortHash(run.provenance.checkpoint_sha256, 8)}</code></span>
      <span>{run.provenance.split} split · <code>{run.provenance.split_version}</code></span>
      <span>cfg <code>{shortHash(run.provenance.config_sha256, 8)}</code></span>
      <span>{formatTimestamp(run.created_at)}</span>
    </div>
  );
}

function TargetTile({ target, title, denominator }: { target: Target; title: string; denominator: ReactNode }) {
  const view = targetView(target);
  return (
    <div className="card tile hero-tile">
      <span className="k">
        {title}
        <span className={`tag ${view.status === "met" ? "ok" : "warn"}`}>
          {view.status === "met" ? <CheckIcon size={12} /> : <AlertIcon size={12} />}
          {view.status === "met" ? "target met" : "target unmet"}
        </span>
      </span>
      <span className="v num">{target.achieved.toFixed(3)}</span>
      <div className="target-track" aria-hidden="true">
        <div className={`fill${target.met ? " met" : ""}`} style={{ width: `${Math.min(1, target.achieved) * 100}%` }} />
        <div className="goal" style={{ left: `${target.goal * 100}%` }} data-label={`goal ${target.goal.toFixed(2)}`} />
      </div>
      <span className="d">
        {view.label} · Δ <b className="num">{view.delta >= 0 ? "+" : "−"}{Math.abs(view.delta).toFixed(3)}</b>
      </span>
      <span className="d">{denominator}</span>
      <span className="d" style={{ color: "var(--muted)" }}>{target.note}</span>
    </div>
  );
}

function Tile({ title, value, detail }: { title: string; value: ReactNode; detail: ReactNode }) {
  return (
    <div className="card tile">
      <span className="k">{title}</span>
      <span className="v num">{value}</span>
      <span className="d">{detail}</span>
    </div>
  );
}

export function IsolatedView({ run }: { run: IsolatedRun }) {
  const [hover, setHover] = useState<number | null>(null);
  const [mode, setMode] = useState<"share" | "count">("share");
  const curves = run.training.curves;
  const epochs = curves.map((c) => c.epoch);
  const loss: Series[] = [
    { key: "train_loss", label: "train loss", color: "var(--series-train)", values: curves.map((c) => c.train_loss) },
    { key: "val_loss", label: "validation loss", color: "var(--series-val)", values: curves.map((c) => c.validation_loss) },
  ];
  const quality: Series[] = [
    { key: "train_acc", label: "train accuracy", color: "var(--series-train)", values: curves.map((c) => c.train_accuracy) },
    { key: "val_acc", label: "validation accuracy", color: "var(--series-val)", values: curves.map((c) => c.validation_accuracy) },
    { key: "val_f1", label: "validation macro-F1", color: "var(--series-f1)", values: curves.map((c) => c.validation_macro_f1) },
  ];
  const maxLoss = Math.max(...loss.flatMap((s) => s.values));
  const m = run.metrics;
  const best = curves.find((c) => c.epoch === run.training.best_epoch);
  const lastEpoch = curves[curves.length - 1];
  return (
    <>
      <div className="tiles">
        <TargetTile
          target={run.target}
          title={`Macro-F1 · ${run.provenance.split}`}
          denominator={<>over <b className="num">{formatNumber(m.examples)}</b> labeled 1 s clips, {run.labels?.length ?? m.confusion_matrix.labels.length} classes</>}
        />
        <Tile
          title={`Accuracy · ${run.provenance.split}`}
          value={formatPercent(m.accuracy)}
          detail={<>{formatNumber(Math.round(m.accuracy * m.examples))} of {formatNumber(m.examples)} clips correct</>}
        />
        <Tile
          title="Best epoch"
          value={<>{run.training.best_epoch}<small> / {run.training.epochs_run}</small></>}
          detail={<>min validation loss {run.training.best_validation_loss.toFixed(3)}{run.training.stopped_early ? " · stopped early" : ""}</>}
        />
        <Tile
          title="Train vs validation gap"
          value={lastEpoch ? <>{((lastEpoch.train_accuracy - lastEpoch.validation_accuracy) * 100).toFixed(1)}<small> pp</small></> : "—"}
          detail={
            lastEpoch && best ? (
              <>final train acc {formatPercent(lastEpoch.train_accuracy)} vs validation {formatPercent(lastEpoch.validation_accuracy)}; at best epoch {formatPercent(best.validation_accuracy)}</>
            ) : "No curves in this run"
          }
        />
      </div>
      <div className="grid-2" style={{ marginBottom: 16 }}>
        <section className="card">
          <div className="card-head">
            <div>
              <h3>Loss per epoch</h3>
              <p className="sub">Training log saved with the checkpoint · {run.training.train_examples} train / {run.training.validation_examples} validation clips</p>
            </div>
            <Legend series={loss} />
          </div>
          <div className="card-body">
            <LineChart
              title="Train and validation loss per epoch"
              epochs={epochs}
              series={loss}
              domain={[0, maxLoss]}
              format={(v) => v.toFixed(2)}
              hover={hover}
              onHover={setHover}
              bestEpoch={run.training.best_epoch}
            />
          </div>
          <RunTrace run={run} />
        </section>
        <section className="card">
          <div className="card-head">
            <div>
              <h3>Accuracy and macro-F1 per epoch</h3>
              <p className="sub">Validation loss rising while training keeps improving is the overfitting signature</p>
            </div>
            <Legend series={quality} />
          </div>
          <div className="card-body">
            <LineChart
              title="Train accuracy, validation accuracy, and validation macro-F1 per epoch"
              epochs={epochs}
              series={quality}
              domain={[0, 1]}
              format={(v) => v.toFixed(2)}
              hover={hover}
              onHover={setHover}
              bestEpoch={run.training.best_epoch}
              reference={{ value: run.target.goal, label: `macro-F1 target ${run.target.goal.toFixed(2)}` }}
            />
          </div>
          <RunTrace run={run} />
        </section>
      </div>
      <div className="grid-2">
        <section className="card">
          <div className="card-head">
            <div>
              <h3>Confusion matrix · {run.provenance.split}</h3>
              <p className="sub">Rows are actual classes, including Unknown and Silence</p>
            </div>
            <div className="seg" role="group" aria-label="Matrix values">
              <button type="button" aria-pressed={mode === "share"} onClick={() => setMode("share")}>Row %</button>
              <button type="button" aria-pressed={mode === "count"} onClick={() => setMode("count")}>Counts</button>
            </div>
          </div>
          <div className="card-body">
            <ConfusionMatrix labels={m.confusion_matrix.labels} values={m.confusion_matrix.values} mode={mode} />
          </div>
          <RunTrace run={run} />
        </section>
        <section className="card">
          <div className="card-head">
            <div>
              <h3>Per-class precision, recall, F1</h3>
              <p className="sub">Averages hide weak classes; support is the denominator for recall</p>
            </div>
          </div>
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr>
                  <th scope="col">Class</th>
                  <th scope="col" className="right">Precision</th>
                  <th scope="col" className="right">Recall</th>
                  <th scope="col">F1</th>
                  <th scope="col" className="right">Support</th>
                </tr>
              </thead>
              <tbody>
                {m.per_class.map((c) => (
                  <tr key={c.label}>
                    <td>
                      {c.label === "unknown" || c.label === "silence" ? (
                        <span className="kw-name" style={{ "--kw": "var(--muted)" } as CSSProperties}>
                          {c.label} <span className="tag">reject class</span>
                        </span>
                      ) : (
                        <KeywordName keyword={c.label} />
                      )}
                    </td>
                    <td className="right mono">{c.precision.toFixed(3)}</td>
                    <td className="right mono">{c.recall.toFixed(3)}</td>
                    <td>
                      <span className="f1bar" aria-hidden="true"><span style={{ width: `${c.f1 * 100}%` }} /></span>
                      <span className="mono" style={{ color: "var(--text)" }}>{c.f1.toFixed(3)}</span>
                    </td>
                    <td className="right mono">{c.support}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="card-foot">
            <span>
              Macro-F1 <b className="mono" style={{ color: "var(--text)" }}>{m.macro_f1.toFixed(3)}</b> is the unweighted mean
              of the {run.labels?.length ?? m.per_class.length} F1 values above.
            </span>
          </div>
          <RunTrace run={run} />
        </section>
      </div>
    </>
  );
}

const HIT = "#5bd0d6";
const FALSE = "#fb414a";
const MISS = "#fe9a00";

export function ContinuousView({ run }: { run: ContinuousRun }) {
  const m = run.metrics;
  const fa = m.false_alarms;
  const events = m.true_positive + m.false_negative;
  const c = run.composition;
  const config = run.provenance.config;
  const detectorFile = shortHash(run.provenance.config_sha256, 12);
  return (
    <>
      <div style={{ marginBottom: 16 }}>
        {"smooth_frames" in config ? (
          <Banner tone="experimental" title="Experimental: temporal BiLSTM prototype, frozen before test">
            Run <code>{run.id}</code> (<code>docs/evaluations/temporal-v1</code>). These numbers measure the frozen
            operating point · smoothing <b>{config.smooth_frames} frames</b> · threshold <b>{config.threshold}</b> ·
            min frames <b>{config.min_frames}</b> · suppression <b>{config.suppression_s} s</b> · detector config{" "}
            <code>{detectorFile}</code>, tuned on the <b>{run.detector.tuned_on.split}</b> split only. Confidence is a
            smoothed frame score, not correctness. Continuous runs never report accuracy. This is not reliable search of
            natural speech. It does not replace the isolated classifier, and the {run.target.goal} event-F1 target is
            reported, never redefined. See ADR-0007.
          </Banner>
        ) : (
          <Banner tone="experimental" title="Experimental: sliding-window detector, frozen before test">
            These numbers measure the frozen operating point · stride <b>{config.stride_s} s</b> · threshold{" "}
            <b>{config.threshold}</b> · min consecutive <b>{config.min_consecutive}</b> · suppression{" "}
            <b>{config.suppression_s} s</b> · detector config <code>{detectorFile}</code>,
            tuned on the <b>{run.detector.tuned_on.split}</b> split only. {run.confidence_note} Continuous runs never
            report accuracy. Per ADR-0006 this is not a working continuous spotting feature.
          </Banner>
        )}
      </div>
      <div className="tiles">
        <TargetTile
          target={run.target}
          title={`Event F1 · ${run.provenance.split}`}
          denominator={<>{events} labeled Keyword events in {formatNumber(c.duration_s)} s of natural speech</>}
        />
        <Tile
          title="Event precision"
          value={m.precision.toFixed(3)}
          detail={<><b className="num">{m.true_positive}</b> hit of <b className="num">{m.detections}</b> Detections</>}
        />
        <Tile
          title="Event recall"
          value={m.recall.toFixed(3)}
          detail={<><b className="num">{m.true_positive}</b> hit of <b className="num">{events}</b> Keyword events</>}
        />
        <Tile
          title="False alarms per hour"
          value={formatNumber(fa.per_hour, 0)}
          detail={<><b className="num">{fa.count}</b> false Detections over <b className="num">{fa.negative_seconds.toFixed(1)} s</b> of annotated Unknown + Silence</>}
        />
      </div>
      <div className="grid-2" style={{ marginBottom: 16 }}>
        <section className="card">
          <div className="card-head">
            <div>
              <h3>Event outcomes</h3>
              <p className="sub">Match rule: {run.detector.match_rule}, tolerance ±{run.detector.tolerance_s} s</p>
            </div>
          </div>
          <div className="card-body outcome">
            <div>
              <div className="row-head"><span>Detections</span><b className="num">{m.detections}</b></div>
              <div className="bar" role="img" aria-label={`${m.true_positive} hits and ${m.false_positive} false Detections`}>
                <div style={{ flex: m.true_positive, background: HIT }} />
                <div style={{ flex: m.false_positive, background: FALSE }} />
              </div>
            </div>
            <div>
              <div className="row-head"><span>Labeled Keyword events</span><b className="num">{events}</b></div>
              <div className="bar" role="img" aria-label={`${m.true_positive} hits and ${m.false_negative} misses`}>
                <div style={{ flex: m.true_positive, background: HIT }} />
                <div style={{ flex: m.false_negative, background: MISS }} />
              </div>
            </div>
            <div className="legend-row">
              <span><i style={{ background: HIT }} />Hit {m.true_positive}</span>
              <span><i style={{ background: FALSE }} />False Detection {m.false_positive}</span>
              <span><i style={{ background: MISS }} />Missed event {m.false_negative}</span>
              {m.ignored_matches ? <span>Ignored matches {m.ignored_matches}</span> : null}
            </div>
            <table className="data">
              <thead>
                <tr>
                  <th scope="col">Negative region</th>
                  <th scope="col" className="right">False alarms</th>
                  <th scope="col" className="right">Seconds</th>
                  <th scope="col" className="right">Per hour</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(fa.by_region).map(([region, value]) => (
                  <tr key={region}>
                    <td style={{ color: "var(--text)" }}>{region}</td>
                    <td className="right mono">{value.count}</td>
                    <td className="right mono">{value.seconds.toFixed(1)}</td>
                    <td className="right mono">{formatNumber(value.per_hour, 0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="table-note">{fa.definition}</p>
          </div>
          <RunTrace run={run} />
        </section>
        <section className="card">
          <div className="card-head">
            <div>
              <h3>Per-Keyword events</h3>
              <p className="sub">Keyword counts per split are small; per-Keyword numbers are not reliable</p>
            </div>
          </div>
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr>
                  <th scope="col">Keyword</th>
                  <th scope="col" className="right">Events</th>
                  <th scope="col" className="right">Hits</th>
                  <th scope="col" className="right">False</th>
                  <th scope="col" className="right">Missed</th>
                  <th scope="col" className="right">Precision</th>
                  <th scope="col" className="right">Recall</th>
                  <th scope="col" className="right">F1</th>
                </tr>
              </thead>
              <tbody>
                {m.per_keyword.map((k) => (
                  <tr key={k.keyword}>
                    <td><KeywordName keyword={k.keyword} /></td>
                    <td className="right mono">{k.true_positive + k.false_negative}</td>
                    <td className="right mono">{k.true_positive}</td>
                    <td className="right mono">{k.false_positive}</td>
                    <td className="right mono">{k.false_negative}</td>
                    <td className="right mono">{k.precision.toFixed(3)}</td>
                    <td className="right mono">{k.recall.toFixed(3)}</td>
                    <td className="right mono">{k.f1.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <RunTrace run={run} />
        </section>
      </div>
      <div className="grid-2">
        <section className="card">
          <div className="card-head"><h3>Test material</h3></div>
          <div className="card-body">
            <dl className="kv">
              <dt>Recordings</dt><dd>{c.recordings} · {formatNumber(c.duration_s)} s</dd>
              <dt>Speakers</dt><dd>{c.speaker_ids.join(", ")}</dd>
              <dt>Annotated Unknown</dt><dd>{(c.annotated_seconds.unknown ?? 0).toFixed(1)} s</dd>
              <dt>Annotated Silence</dt><dd>{(c.annotated_seconds.silence ?? 0).toFixed(1)} s</dd>
              <dt>Unlabeled</dt><dd>{c.unlabeled_seconds.toFixed(1)} s (not counted as negative exposure)</dd>
              <dt>Keyword events</dt>
              <dd>{Object.entries(c.keyword_events).map(([k, n]) => `${k} ${n}`).join(" · ")}</dd>
              <dt>Annotation quality</dt><dd>{c.annotation_quality}</dd>
            </dl>
          </div>
          <RunTrace run={run} />
        </section>
        <section className="card">
          <div className="card-head"><h3>Limitations recorded with this run</h3></div>
          <div className="card-body" style={{ display: "grid", gap: 14 }}>
            <ul className="list-plain">
              {run.limitations.map((l) => <li key={l}>{l}</li>)}
            </ul>
            <dl className="kv">
              <dt>Detector created</dt><dd>{formatTimestamp(run.detector.detector_created_at)}</dd>
              <dt>Tuned on</dt><dd>{run.detector.tuned_on.dataset_version} · {run.detector.tuned_on.split}</dd>
            </dl>
          </div>
          <RunTrace run={run} />
        </section>
      </div>
    </>
  );
}
