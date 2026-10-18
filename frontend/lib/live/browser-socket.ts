// In-browser stand-in for the /api/live WebSocket. A serverless deployment
// cannot hold a socket open, so the deployed site scores the mic here with the
// same ONNX model, features, windows and decision policy as the API. It speaks
// the server's message protocol, so LiveSession drives both the same way.
// Mic audio never leaves the page; decisions are not stored anywhere.
import type { InferenceSession, Tensor } from "onnxruntime-web";
import { LiveDecider, LiveStream, type LivePolicy } from "./decider.ts";
import { mfcc, windowRms, WINDOW_FRAMES } from "./dsp.ts";

const MODEL_URL = "/models/kws.onnx";
const METADATA_URL = "/models/kws.json";
const ORT_WASM_PATH = "/ort/";
const PROTOCOL_VERSION = 1;
const N_MFCC = 40;

type Metadata = {
  version: number;
  checkpoint_sha256: string;
  labels: string[];
  keywords: string[];
  mean: number[];
  std: number[];
  policy: LivePolicy;
  actions: { keyword: string; action: string; label: string }[];
};

type LoadedModel = {
  metadata: Metadata;
  session: InferenceSession;
  tensor: typeof Tensor;
};

let loading: Promise<LoadedModel> | null = null;

function loadModel(): Promise<LoadedModel> {
  loading ??= (async () => {
    const ort = await import("onnxruntime-web/wasm");
    ort.env.wasm.wasmPaths = ORT_WASM_PATH;
    // Threads need cross-origin isolation; one thread scores 10 windows/s easily.
    ort.env.wasm.numThreads = 1;
    const response = await fetch(METADATA_URL, { cache: "force-cache" });
    if (!response.ok) throw new Error("Model metadata is unavailable.");
    const metadata = (await response.json()) as Metadata;
    if (metadata.version !== PROTOCOL_VERSION) throw new Error("Model metadata version mismatch.");
    const session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ["wasm"] });
    return { metadata, session, tensor: ort.Tensor };
  })();
  loading.catch(() => {
    loading = null;
  });
  return loading;
}

const CONNECTING = 0;
const OPEN = 1;
const CLOSING = 2;
const CLOSED = 3;

export class BrowserLiveSocket {
  readyState = CONNECTING;
  bufferedAmount = 0;
  binaryType: BinaryType = "arraybuffer";
  onopen: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;

  private model: LoadedModel | null = null;
  private stream: LiveStream | null = null;
  private decider: LiveDecider | null = null;
  private busy = false;
  private arrived = 0;
  private nextId = 1;

  constructor() {
    loadModel().then(
      (model) => {
        if (this.readyState !== CONNECTING) return;
        this.model = model;
        this.readyState = OPEN;
        this.onopen?.(new Event("open"));
      },
      () => {
        if (this.readyState !== CONNECTING) return;
        this.emit({ type: "error", code: "model_unavailable", message: "The in-browser model failed to load." });
        this.finish(1011);
      },
    );
  }

  send(data: string | ArrayBuffer): void {
    if (this.readyState !== OPEN) return;
    if (typeof data !== "string") {
      this.audio(data);
      return;
    }
    type Control = { type?: unknown; sample_rate?: unknown; version?: unknown };
    let control: Control | null;
    try {
      control = JSON.parse(data) as Control | null;
    } catch {
      control = null;
    }
    const kind = typeof control?.type === "string" ? control.type : "";
    if (kind === "start") this.start(control?.version, control?.sample_rate);
    else if (kind === "stop") {
      this.emit({ type: "stopped" });
      this.finish(1000);
    } else if (kind.startsWith("external_") || kind.startsWith("negative_")) {
      // Windows Actions need the local API; the deployed site never offers them.
      this.emit(this.externalState());
    }
    // action_result needs no answer: nothing is stored in the browser.
  }

  close(code = 1000): void {
    if (this.readyState === CLOSED) return;
    this.finish(code);
  }

  private start(version: unknown, rate: unknown): void {
    const model = this.model;
    if (!model) return;
    if (version !== PROTOCOL_VERSION) {
      this.fail("unsupported_version", "Reload the page to match the protocol.");
      return;
    }
    if (typeof rate !== "number" || !Number.isInteger(rate) || rate < 8000 || rate > 96000) {
      this.fail("unsupported_audio", "Send mono PCM_s16le at 8-96 kHz.");
      return;
    }
    const { metadata } = model;
    this.stream = new LiveStream(rate, metadata.policy);
    this.decider = new LiveDecider(
      metadata.policy,
      metadata.labels,
      new Map(metadata.actions.map((item) => [item.keyword, item.action])),
    );
    this.emit({
      type: "ready",
      version: PROTOCOL_VERSION,
      session_id: crypto.randomUUID().replaceAll("-", ""),
      keywords: metadata.keywords,
      actions: metadata.actions,
      commands: metadata.actions,
      policy: metadata.policy,
    });
    this.emit(this.externalState());
  }

  private audio(frame: ArrayBuffer): void {
    if (!this.stream) return;
    try {
      this.stream.push(frame);
    } catch (error) {
      this.fail("invalid_frame", error instanceof Error ? error.message : "Invalid frame.");
      return;
    }
    this.arrived = performance.now();
    if (!this.busy) void this.process();
  }

  // Score every complete window; frames that arrive meanwhile wait for the next pass.
  private async process(): Promise<void> {
    this.busy = true;
    try {
      while (this.readyState === OPEN && this.stream && this.decider && this.model) {
        const batch = this.stream.takeWindows();
        if (batch.dropped_s) {
          this.decider.reset();
          this.emit({ type: "backpressure", dropped_s: batch.dropped_s });
        }
        if (!batch.windows.length) return;
        const scores = await this.score(batch.windows);
        if (this.readyState !== OPEN) return;
        const labels = this.model.metadata.labels;
        batch.ends_s.forEach((end, i) => {
          const row = scores[i];
          let top = 0;
          for (let k = 1; k < row.length; k++) if (row[k] > row[top]) top = k;
          this.emit({ type: "frame", t_s: round(end, 3), label: labels[top], score: row[top] });
          for (const decision of this.decider?.observe(end, row) ?? []) {
            this.emit({
              type: "decision",
              id: this.nextId++,
              t_s: round(decision.t_s, 3),
              label: decision.label,
              score: decision.score,
              accepted: decision.accepted,
              reason: decision.reason,
              action: decision.action,
              processing_ms: round(performance.now() - this.arrived, 1),
            });
          }
        });
      }
    } catch {
      this.fail("live_failed", "Live scoring failed.");
    } finally {
      this.busy = false;
    }
  }

  private async score(windows: Float32Array[]): Promise<Float32Array[]> {
    const model = this.model;
    if (!model) throw new Error("Model is unavailable");
    const { metadata, session, tensor } = model;
    const silence = metadata.labels.indexOf("silence");
    const gate = 10 ** (metadata.policy.rms_gate_dbfs / 20);
    const result = windows.map(() => {
      const row = new Float32Array(metadata.labels.length);
      row[silence] = 1;
      return row;
    });
    const loud = windows.flatMap((window, i) => (windowRms(window) >= gate ? [i] : []));
    if (!loud.length) return result;
    const size = WINDOW_FRAMES * N_MFCC;
    const features = new Float32Array(loud.length * size);
    loud.forEach((index, i) => features.set(mfcc(windows[index], metadata.mean, metadata.std), i * size));
    const output = await session.run({ features: new tensor("float32", features, [loud.length, WINDOW_FRAMES, N_MFCC]) });
    const probabilities = output.probabilities.data as Float32Array;
    const classes = metadata.labels.length;
    loud.forEach((index, i) => result[index].set(probabilities.subarray(i * classes, (i + 1) * classes)));
    return result;
  }

  private externalState() {
    return {
      type: "external_state",
      supported: false,
      enabled: false,
      target_id: null,
      allowed_actions: [],
      targets: [],
      experimental: true,
      error: "Windows Actions need the local API.",
      negative_running: false,
    };
  }

  private fail(code: string, message: string): void {
    this.emit({ type: "error", code, message });
    this.finish(1011);
  }

  private emit(message: Record<string, unknown>): void {
    if (this.readyState === CLOSED) return;
    this.onmessage?.(new MessageEvent("message", { data: JSON.stringify(message) }));
  }

  private finish(code: number): void {
    if (this.readyState === CLOSED) return;
    this.readyState = CLOSING;
    this.stream = null;
    this.decider = null;
    this.readyState = CLOSED;
    this.onclose?.(new CloseEvent("close", { code }));
  }
}

function round(value: number, digits: number): number {
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}
