"use client";

import { useEffect, useRef } from "react";
import type { Detection } from "../../lib/api.ts";
import type { SortDirection, SortKey } from "../../lib/detections.ts";
import { formatClock } from "../../lib/format.ts";
import { keywordColor } from "../../lib/keywords.ts";
import { PlayIcon } from "../icons";
import { KeywordName, kwStyle, ScoreBar } from "../ui";

const COLUMNS: { key: SortKey | null; label: string; right?: boolean }[] = [
  { key: "keyword", label: "Keyword" },
  { key: "start", label: "Start" },
  { key: "end", label: "End" },
  { key: "duration", label: "Span", right: true },
  { key: "confidence", label: "Confidence (score)" },
  { key: null, label: "Threshold", right: true },
  { key: null, label: "Windows", right: true },
  { key: null, label: "" },
];

export function DetectionTable({
  detections,
  sort,
  onSort,
  activeId,
  hoverId,
  onHover,
  onPlay,
  onSeek,
}: {
  detections: readonly Detection[];
  sort: { key: SortKey; direction: SortDirection };
  onSort: (key: SortKey) => void;
  activeId: number | null;
  hoverId: number | null;
  onHover: (id: number | null) => void;
  onPlay: (detection: Detection) => void;
  onSeek: (seconds: number) => void;
}) {
  const wrapRef = useRef<HTMLDivElement>(null);

  // Keep the active row visible inside the table without scrolling the page.
  useEffect(() => {
    const wrap = wrapRef.current;
    if (activeId === null || !wrap) return;
    const row = wrap.querySelector<HTMLTableRowElement>(`[data-detection="${activeId}"]`);
    if (!row) return;
    const header = 38;
    if (row.offsetTop - header < wrap.scrollTop) wrap.scrollTop = row.offsetTop - header;
    else if (row.offsetTop + row.offsetHeight > wrap.scrollTop + wrap.clientHeight) {
      wrap.scrollTop = row.offsetTop + row.offsetHeight - wrap.clientHeight;
    }
  }, [activeId]);

  return (
    <div className="table-wrap" ref={wrapRef} style={{ maxHeight: 440 }}>
      <table className="data">
        <caption className="sr-only">
          Detections. Confidence is a classifier score, not a measure of correctness.
        </caption>
        <thead>
          <tr>
            {COLUMNS.map((column) => (
              <th
                key={column.label || "play"}
                className={column.right ? "right" : undefined}
                aria-sort={column.key && sort.key === column.key ? sort.direction : undefined}
                scope="col"
              >
                {column.key ? (
                  <button type="button" className="sort-btn" onClick={() => onSort(column.key as SortKey)}>
                    {column.label}
                    <span className="arrow" aria-hidden="true">
                      {sort.key === column.key && sort.direction === "descending" ? "↓" : "↑"}
                    </span>
                  </button>
                ) : (
                  <span className={column.label ? undefined : "sr-only"}>{column.label || "Play"}</span>
                )}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {detections.map((d) => (
            <tr
              key={d.id}
              data-detection={d.id}
              className={`det-row${d.id === hoverId ? " hover" : ""}`}
              style={kwStyle(d.keyword)}
              aria-current={d.id === activeId ? "true" : undefined}
              onMouseEnter={() => onHover(d.id)}
              onMouseLeave={() => onHover(null)}
              onClick={() => onPlay(d)}
            >
              <td><KeywordName keyword={d.keyword} /></td>
              <td>
                <button
                  type="button"
                  className="time-btn"
                  aria-label={`Seek to ${formatClock(d.start_s)}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    onSeek(d.start_s);
                  }}
                >
                  {formatClock(d.start_s)}
                </button>
              </td>
              <td>
                <button
                  type="button"
                  className="time-btn"
                  aria-label={`Seek to ${formatClock(d.end_s)}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    onSeek(d.end_s);
                  }}
                >
                  {formatClock(d.end_s)}
                </button>
              </td>
              <td className="right mono">{(d.end_s - d.start_s).toFixed(2)} s</td>
              <td><ScoreBar value={d.confidence} threshold={d.threshold} color={keywordColor(d.keyword)} /></td>
              <td className="right mono">{d.threshold.toFixed(2)}</td>
              <td className="right mono">{d.windows}</td>
              <td className="right">
                <button
                  type="button"
                  className="btn small icon ghost"
                  aria-label={`Play ${d.keyword} Detection at ${formatClock(d.start_s)}`}
                  onFocus={() => onHover(d.id)}
                  onBlur={() => onHover(null)}
                  onClick={(e) => {
                    e.stopPropagation();
                    onPlay(d);
                  }}
                >
                  <PlayIcon size={14} />
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
