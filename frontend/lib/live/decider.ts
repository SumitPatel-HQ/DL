// Browser port of LiveStream and LiveDecider (backend/app/services/live.py).
// Keep both in step with the Python originals; tests/decider.test.mjs replays
// Python-generated decisions.
import { canonicalWindow } from "./dsp.ts";

export type LivePolicy = {
  hop_s: number;
  threshold: number;
  margin: number;
  smooth_windows: number;
  rms_gate_dbfs: number;
  min_consecutive: number;
  cooldown_s: number;
  max_lag_s: number;
  max_frame_bytes: number;
  window_s: number;
};

export type Decision = {
  t_s: number;
  label: string;
  score: number;
  accepted: boolean;
  reason: string;
  action: string | null;
};

function peakMargin(scores: ArrayLike<number>): number {
  let first = -Infinity;
  let second = -Infinity;
  for (let i = 0; i < scores.length; i++) {
    const value = scores[i];
    if (value > first) {
      second = first;
      first = value;
    } else if (value > second) {
      second = value;
    }
  }
  return scores.length < 2 ? first : first - second;
}

export class LiveDecider {
  private readonly policy: LivePolicy;
  private readonly labels: readonly string[];
  private readonly actions: ReadonlyMap<string, string>;
  private lastAccept: { t_s: number; label: string } | null = null;
  private history: Float32Array[] = [];
  private runLabel: string | null = null;
  private decided = false;
  private streak = 0;
  private best = 0;
  private lastS = 0;

  constructor(policy: LivePolicy, labels: readonly string[], actions: ReadonlyMap<string, string>) {
    this.policy = policy;
    this.labels = labels;
    this.actions = actions;
  }

  // End the current run without a decision. Cooldown is kept.
  reset(keepHistory = false): void {
    if (!keepHistory) this.history = [];
    this.runLabel = null;
    this.decided = false;
    this.streak = 0;
    this.best = 0;
    this.lastS = 0;
  }

  observe(t_s: number, scores: Float32Array): Decision[] {
    this.history.push(scores);
    const depth = Math.max(1, this.policy.smooth_windows);
    if (this.history.length > depth) this.history.splice(0, this.history.length - depth);
    const smoothed = new Float32Array(scores.length);
    for (const row of this.history) for (let i = 0; i < row.length; i++) smoothed[i] += row[i];
    for (let i = 0; i < smoothed.length; i++) smoothed[i] /= this.history.length;
    let index = 0;
    for (let i = 1; i < smoothed.length; i++) if (smoothed[i] > smoothed[index]) index = i;
    const label = this.labels[index];
    const score = smoothed[index];
    const decisions: Decision[] = [];
    if (label !== this.runLabel) {
      if (this.runLabel !== null && this.actions.has(this.runLabel) && !this.decided) {
        decisions.push({
          t_s: this.lastS, label: this.runLabel, score: this.best, accepted: false, reason: "low_confidence", action: null,
        });
      }
      this.reset(true);
      this.runLabel = label;
    }
    this.lastS = t_s;
    this.best = Math.max(this.best, score);
    if (this.decided) return decisions;
    const action = this.actions.get(label);
    if (action === undefined) {
      this.decided = true;
      decisions.push({ t_s, label, score, accepted: false, reason: label, action: null });
      return decisions;
    }
    const confident = score >= this.policy.threshold && peakMargin(smoothed) >= this.policy.margin;
    this.streak = confident ? this.streak + 1 : 0;
    if (this.streak >= this.policy.min_consecutive) {
      this.decided = true;
      const last = this.lastAccept;
      if (last !== null && last.label === label && t_s - last.t_s < this.policy.cooldown_s) {
        decisions.push({ t_s, label, score: this.best, accepted: false, reason: "cooldown", action: null });
      } else {
        this.lastAccept = { t_s, label };
        decisions.push({ t_s, label, score: this.best, accepted: true, reason: "accepted", action });
      }
    }
    return decisions;
  }
}

export type WindowBatch = { windows: Float32Array[]; ends_s: number[]; dropped_s: number };

// Bounded mic tail at the browser's native rate, cut into 16 kHz windows.
export class LiveStream {
  private readonly rate: number;
  private readonly policy: LivePolicy;
  private readonly window: number;
  private readonly hop: number;
  private samples = new Float32Array(0);
  private offset = 0;
  private received = 0;
  private nextEnd: number;

  constructor(sampleRate: number, policy: LivePolicy) {
    this.rate = sampleRate;
    this.policy = policy;
    this.window = Math.round(sampleRate * policy.window_s);
    this.hop = Math.round(sampleRate * policy.hop_s);
    this.nextEnd = this.window;
  }

  push(frame: ArrayBuffer): void {
    if (frame.byteLength === 0 || frame.byteLength % 2 || frame.byteLength > this.policy.max_frame_bytes) {
      throw new Error("Frames must be non-empty mono PCM_s16le within the byte limit");
    }
    const pcm = new Int16Array(frame);
    const grown = new Float32Array(this.samples.length + pcm.length);
    grown.set(this.samples);
    for (let i = 0; i < pcm.length; i++) grown[this.samples.length + i] = pcm[i] / 32768;
    this.samples = grown;
    this.received += pcm.length;
  }

  // Every complete window since the last call, skipping audio that is too stale.
  takeWindows(): WindowBatch {
    let dropped = 0;
    const lag = this.received - this.nextEnd;
    if (lag > Math.round(this.rate * this.policy.max_lag_s)) {
      dropped = Math.floor(lag / this.hop) * this.hop;
      this.nextEnd += dropped;
    }
    const windows: Float32Array[] = [];
    const ends: number[] = [];
    while (this.nextEnd <= this.received) {
      const first = this.nextEnd - this.window - this.offset;
      windows.push(canonicalWindow(this.samples.subarray(first, first + this.window), this.rate));
      ends.push(this.nextEnd / this.rate);
      this.nextEnd += this.hop;
    }
    const keep = this.nextEnd - this.window - this.offset;
    if (keep > 0) {
      this.samples = this.samples.slice(keep);
      this.offset += keep;
    }
    return { windows, ends_s: ends, dropped_s: dropped / this.rate };
  }
}
