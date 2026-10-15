"use client";

import { useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { niceTicks, rowNormalize } from "../../lib/evaluations.ts";
import { formatPercent } from "../../lib/format.ts";

export function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(0);
  useLayoutEffect(() => {
    const element = ref.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.floor(entry.contentRect.width)));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  return [ref, width] as const;
}

export type Series = { key: string; label: string; color: string; values: number[] };

const M = { top: 16, right: 92, bottom: 30, left: 44 };

// Multi-series line chart on one y-axis, with a crosshair shared across charts.
export function LineChart({
  title,
  epochs,
  series,
  domain,
  format,
  hover,
  onHover,
  bestEpoch,
  reference,
  height = 230,
}: {
  title: string;
  epochs: number[];
  series: Series[];
  domain: [number, number];
  format: (value: number) => string;
  hover: number | null;
  onHover: (index: number | null) => void;
  bestEpoch: number;
  reference?: { value: number; label: string };
  height?: number;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const plotW = Math.max(10, width - M.left - M.right);
  const plotH = height - M.top - M.bottom;
  const ticks = niceTicks(domain[0], domain[1], 5);
  const yMin = ticks[0];
  const yMax = ticks[ticks.length - 1];
  const first = epochs[0] ?? 1;
  const last = epochs[epochs.length - 1] ?? 1;
  const x = (epoch: number) => M.left + ((epoch - first) / Math.max(1, last - first)) * plotW;
  const y = (value: number) => M.top + (1 - (value - yMin) / (yMax - yMin)) * plotH;
  const xTicks = epochs.filter((e) => e === first || e === last || e % Math.max(1, Math.ceil(epochs.length / 8)) === 0);

  // End labels nudged apart so they never overlap.
  const ends = series
    .map((s) => ({ s, y: y(s.values[s.values.length - 1]) }))
    .sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + 14);

  const index = hover;
  const pointer = (clientX: number, element: SVGSVGElement) => {
    const rect = element.getBoundingClientRect();
    const epoch = first + ((clientX - rect.left - M.left) / plotW) * (last - first);
    const nearest = Math.round(epoch) - first;
    onHover(Math.min(epochs.length - 1, Math.max(0, nearest)));
  };

  return (
    <div className="chart" ref={ref} style={{ position: "relative" }}>
      {width > 0 ? (
        <svg
          width={width}
          height={height}
          role="img"
          aria-label={`${title}. ${series.map((s) => `${s.label} ends at ${format(s.values[s.values.length - 1])}`).join(". ")}. Best epoch ${bestEpoch}.`}
          onPointerMove={(e) => pointer(e.clientX, e.currentTarget)}
          onPointerLeave={() => onHover(null)}
        >
          {ticks.map((tick) => (
            <g key={tick}>
              <line x1={M.left} x2={M.left + plotW} y1={y(tick)} y2={y(tick)} stroke="var(--grid)" />
              <text x={M.left - 8} y={y(tick)} textAnchor="end" dominantBaseline="middle" className="tick-label">
                {format(tick)}
              </text>
            </g>
          ))}
          <line x1={M.left} x2={M.left + plotW} y1={M.top + plotH} y2={M.top + plotH} stroke="var(--axis)" />
          {xTicks.map((epoch) => (
            <text key={epoch} x={x(epoch)} y={M.top + plotH + 18} textAnchor="middle" className="tick-label">
              {epoch}
            </text>
          ))}
          {reference ? (
            <g>
              <line x1={M.left} x2={M.left + plotW} y1={y(reference.value)} y2={y(reference.value)} stroke="var(--text-2)" strokeDasharray="4 4" />
              <text x={M.left + 6} y={y(reference.value) - 6} style={{ fill: "var(--text-2)" }}>{reference.label}</text>
            </g>
          ) : null}
          <line x1={x(bestEpoch)} x2={x(bestEpoch)} y1={M.top} y2={M.top + plotH} stroke="var(--text)" strokeOpacity={0.5} />
          <text x={x(bestEpoch) + 5} y={M.top + 10} style={{ fill: "var(--text)", fontWeight: 600 }}>
            best epoch {bestEpoch}
          </text>
          {series.map((s) => (
            <path
              key={s.key}
              d={s.values.map((v, i) => `${i ? "L" : "M"}${x(epochs[i]).toFixed(1)},${y(v).toFixed(1)}`).join("")}
              fill="none"
              stroke={s.color}
              strokeWidth={2}
              strokeLinejoin="round"
              strokeLinecap="round"
            />
          ))}
          {ends.map(({ s, y: labelY }) => (
            <g key={s.key}>
              <circle cx={x(last)} cy={y(s.values[s.values.length - 1])} r={4} fill={s.color} stroke="var(--surface-1)" strokeWidth={2} />
              <text x={x(last) + 9} y={labelY} dominantBaseline="middle" style={{ fill: "var(--text-2)" }}>
                {s.label}
              </text>
            </g>
          ))}
          {index !== null ? (
            <g>
              <line x1={x(epochs[index])} x2={x(epochs[index])} y1={M.top} y2={M.top + plotH} stroke="var(--text-2)" />
              {series.map((s) => (
                <circle key={s.key} cx={x(epochs[index])} cy={y(s.values[index])} r={4.5} fill={s.color} stroke="var(--surface-1)" strokeWidth={2} />
              ))}
            </g>
          ) : null}
        </svg>
      ) : (
        <div className="skeleton" style={{ height }} />
      )}
      {index !== null && width > 0 ? (
        <div
          className="tooltip"
          style={{
            top: M.top,
            left: Math.min(x(epochs[index]) + 12, width - 190),
          }}
        >
          <span className="t-title">Epoch {epochs[index]}{epochs[index] === bestEpoch ? " · best" : ""}</span>
          {series.map((s) => (
            <span key={s.key} className="t-row">
              <span><i style={{ width: 10, height: 2, background: s.color, display: "inline-block" }} />{s.label}</span>
              <b>{format(s.values[index])}</b>
            </span>
          ))}
        </div>
      ) : null}
    </div>
  );
}

export function Legend({ series }: { series: Series[] }) {
  return (
    <div className="chart-legend">
      {series.map((s) => (
        <span key={s.key} className="item">
          <span className="line" style={{ background: s.color }} />
          {s.label}
        </span>
      ))}
    </div>
  );
}

// Rows are actual classes, columns predicted. Ink is the row share (recall view).
export function ConfusionMatrix({
  labels,
  values,
  mode,
}: {
  labels: string[];
  values: number[][];
  mode: "share" | "count";
}) {
  const shares = rowNormalize(values);
  const [focus, setFocus] = useState<{ row: number; col: number } | null>(null);
  const readout: ReactNode = focus
    ? (() => {
        const total = values[focus.row].reduce((a, b) => a + b, 0);
        const count = values[focus.row][focus.col];
        return (
          <>
            Actual <b>{labels[focus.row]}</b> predicted as <b>{labels[focus.col]}</b>: {count} of {total} (
            {formatPercent(total ? count / total : 0)} of actual {labels[focus.row]})
          </>
        );
      })()
    : "Hover or focus a cell for its count and share. Diagonal cells (outlined) are correct predictions.";
  return (
    <div style={{ display: "grid", gap: 10 }}>
      <div className="table-wrap">
        <table className="cm-table">
          <caption className="sr-only">
            Confusion matrix. Rows are actual classes, columns are predicted classes. Values are {mode === "share" ? "row shares" : "counts"}.
          </caption>
          <thead>
            <tr>
              <th scope="col" className="corner">
                <span className="sr-only">Actual class</span><span aria-hidden="true">actual ↓</span>
              </th>
              {labels.map((label) => (
                <th key={label} scope="col">{label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {values.map((row, r) => (
              <tr key={labels[r]}>
                <th scope="row">{labels[r]}</th>
                {row.map((count, c) => {
                  const share = shares[r][c];
                  return (
                    <td
                      key={labels[c]}
                      tabIndex={0}
                      className={r === c ? "diag" : undefined}
                      style={{
                        background: `color-mix(in srgb, var(--seq-1) ${Math.round(share * 100)}%, var(--seq-0))`,
                        color: share > 0.5 ? "#fff" : share > 0.12 ? "var(--text)" : "var(--muted)",
                      }}
                      onMouseEnter={() => setFocus({ row: r, col: c })}
                      onFocus={() => setFocus({ row: r, col: c })}
                      onMouseLeave={() => setFocus(null)}
                      aria-label={`Actual ${labels[r]}, predicted ${labels[c]}: ${count}, ${formatPercent(share)} of row`}
                    >
                      {mode === "share" ? (share === 0 ? "·" : Math.round(share * 100)) : count}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="table-note" aria-live="polite" style={{ minHeight: 20 }}>{readout}</p>
      <div className="cm-key">
        <span>0%</span>
        <span className="ramp" />
        <span>100% of actual class</span>
        <span style={{ marginLeft: 12 }}>{mode === "share" ? "Cells show row % (recall view)" : "Cells show counts; ink stays row share"}</span>
      </div>
    </div>
  );
}
