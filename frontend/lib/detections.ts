// Detection review logic shared by the waveform regions and the table.
import type { Detection } from "./api.ts";

export type SortKey = "keyword" | "start" | "end" | "duration" | "confidence";
export type SortDirection = "ascending" | "descending";

const value: Record<SortKey, (d: Detection) => number | string> = {
  keyword: (d) => d.keyword,
  start: (d) => d.start_s,
  end: (d) => d.end_s,
  duration: (d) => d.end_s - d.start_s,
  confidence: (d) => d.confidence,
};

export function sortDetections(
  detections: readonly Detection[],
  key: SortKey,
  direction: SortDirection,
): Detection[] {
  const sign = direction === "ascending" ? 1 : -1;
  return [...detections].sort((a, b) => {
    const left = value[key](a);
    const right = value[key](b);
    const order = left < right ? -1 : left > right ? 1 : 0;
    return order * sign || a.start_s - b.start_s || a.id - b.id;
  });
}

export function filterDetections(
  detections: readonly Detection[],
  keywords: ReadonlySet<string>,
): Detection[] {
  return detections.filter((d) => keywords.has(d.keyword));
}

// Overlapping spans go to separate lanes so no region hides another.
export function assignLanes(detections: readonly Detection[]): {
  lanes: Map<number, number>;
  count: number;
} {
  const ends: number[] = [];
  const lanes = new Map<number, number>();
  for (const d of [...detections].sort((a, b) => a.start_s - b.start_s || a.id - b.id)) {
    let lane = ends.findIndex((end) => end <= d.start_s + 1e-9);
    if (lane === -1) {
      lane = ends.length;
      ends.push(d.end_s);
    } else ends[lane] = d.end_s;
    lanes.set(d.id, lane);
  }
  return { lanes, count: Math.max(1, ends.length) };
}

// The Detection under the playhead; the latest start wins when spans overlap.
export function detectionAt(detections: readonly Detection[], time: number): Detection | null {
  let found: Detection | null = null;
  for (const d of detections) {
    if (d.start_s <= time && time < d.end_s && (found === null || d.start_s > found.start_s)) {
      found = d;
    }
  }
  return found;
}

// Previous/next Detection by start time, relative to the clock.
export function neighbor(
  detections: readonly Detection[],
  time: number,
  direction: 1 | -1,
): Detection | null {
  const ordered = sortDetections(detections, "start", "ascending");
  if (direction === 1) return ordered.find((d) => d.start_s > time + 0.05) ?? null;
  return ordered.findLast((d) => d.start_s < time - 0.3) ?? null;
}

export const PRE_ROLL_S = 0.2;

export function segment(d: Detection, duration: number): { from: number; to: number } {
  return { from: Math.max(0, d.start_s - PRE_ROLL_S), to: Math.min(duration, d.end_s) };
}
