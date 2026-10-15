// Browser client for the loopback API. The launcher serves only this endpoint,
// and the API only accepts browser origins on 127.0.0.1:3000 or localhost:3000.
export const API_BASE = "http://127.0.0.1:8000";
export const LIVE_URL = "ws://127.0.0.1:8000/api/live";

export class ApiError extends Error {
  readonly code: string;
  readonly status: number;

  constructor(code: string, status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
  }
}

export function errorCode(error: unknown): string {
  return error instanceof ApiError ? error.code : "unexpected";
}

export type RecordingStatus = "uploaded" | "queued" | "running" | "completed" | "failed";

export type DetectorConfig = {
  stride_s: number;
  threshold: number;
  min_consecutive: number;
  suppression_s: number;
  window_s: number;
};

/** Post-processing of the temporal prototype (ADR-0007); no sliding window. */
export type TemporalConfig = {
  smooth_frames: number;
  threshold: number;
  min_frames: number;
  suppression_s: number;
};

export type Recording = {
  id: string;
  filename: string;
  duration_s: number;
  sample_rate: number;
  channels: number;
  status: RecordingStatus;
  error_code: string | null;
  model_id: string | null;
  keywords: string[];
  detector: DetectorConfig | null;
  created_at: string;
};

export type Detection = {
  id: number;
  keyword: string;
  start_s: number;
  end_s: number;
  confidence: number;
  threshold: number;
  windows: number;
};

export type DetectionResult = {
  recording_id: string;
  model_id: string | null;
  status: RecordingStatus;
  code: string;
  detections: Detection[];
  detector: DetectorConfig | null;
  timestamp_quality: string;
  experimental: boolean;
  limitation: string;
  evaluation_evidence: string;
};

export type ActiveModel = {
  id: string;
  version: string | null;
  labels: string[];
  keywords: string[];
  detector: DetectorConfig;
  experimental: boolean;
  limitation: string;
  evaluation_evidence: string;
};

export type HealthModel =
  | { status: "ready"; code: "model_ready"; id: string }
  | { status: "unavailable"; code: string; setup_action: string };

export type Provenance = {
  checkpoint_sha256: string;
  dataset_version: string;
  split_version: string;
  split: string;
  config_role: "training" | "detector";
  config_sha256: string;
  code_revision: { commit: string | null; dirty: boolean | null };
  /** Present on noise-condition runs; a missing condition means clean. */
  condition?: "clean" | "noisy";
  noise_sources?: string[];
  snr_db?: number | null;
  noise_sources_by_split?: Record<string, string[]>;
  /** Set when the checkpoint trained on a different dataset version than the run used. */
  checkpoint_dataset_version?: string | null;
};

export type Target = {
  metric: "macro_f1" | "event_f1";
  goal: number;
  achieved: number;
  met: boolean;
  note: string;
};

export type RunKind = "isolated" | "continuous";

export type RunSummary = {
  id: string;
  kind: RunKind;
  experiment_id: string;
  created_at: string;
  source: string;
  provenance: Provenance;
  target: Target;
};

export type ClassMetrics = {
  label: string;
  precision: number;
  recall: number;
  f1: number;
  support: number;
};

export type EpochCurve = {
  epoch: number;
  train_loss: number;
  train_accuracy: number;
  validation_loss: number;
  validation_accuracy: number;
  validation_macro_f1: number;
};

export type IsolatedRun = RunSummary & {
  kind: "isolated";
  experimental: false;
  provenance: Provenance & { config: Record<string, unknown> };
  labels: string[];
  metrics: {
    examples: number;
    accuracy: number;
    macro_f1: number;
    per_class: ClassMetrics[];
    confusion_matrix: { labels: string[]; rows: "true"; columns: "detected"; values: number[][] };
    silence_recall_by_source?: Record<string, number>;
  };
  training: {
    train_examples: number;
    validation_examples: number;
    epochs_run: number;
    best_epoch: number;
    best_validation_loss: number;
    stopped_early: boolean;
    curves: EpochCurve[];
  };
};

export type EventCounts = {
  true_positive: number;
  false_positive: number;
  false_negative: number;
  precision: number;
  recall: number;
  f1: number;
};

export type ContinuousRun = RunSummary & {
  kind: "continuous";
  experimental: true;
  model_family: "sliding-window" | "temporal-bilstm-frame";
  provenance: Provenance & { config: DetectorConfig | TemporalConfig };
  metrics: EventCounts & {
    detections: number;
    false_alarms: {
      definition: string;
      count: number;
      negative_seconds: number;
      per_hour: number;
      by_region: Record<string, { count: number; seconds: number; per_hour: number }>;
    };
    ignored_matches: number;
    per_keyword: (EventCounts & { keyword: string })[];
  };
  detector: {
    tolerance_s: number;
    match_rule: string;
    tuned_on: { dataset_version: string; split_version: string; split: string };
    detector_created_at: string;
  };
  composition: {
    recordings: number;
    duration_s: number;
    keyword_events: Record<string, number>;
    ignored_keyword_events: number;
    annotated_seconds: Record<string, number>;
    unlabeled_seconds: number;
    annotation_quality: string;
    speaker_ids: string[];
  };
  limitations: string[];
  confidence_note: string;
};

export type RunDetail = IsolatedRun | ContinuousRun;

export type Comparison = {
  code: "comparison_available";
  comparison_id: string;
  comparison_type: "model" | "noise";
  noise_sources: string[];
  noise_sources_by_split: Record<string, string[]>;
  kind: RunKind;
  dataset_version: string;
  split_version: string;
  split: string;
  runs: [RunDetail, RunDetail];
};

export type ActionMap = {
  version: number;
  actions: { keyword: string; action: string; label: string }[];
  /** Deprecated alias for `actions`, returned by GET /api/commands and the WS ready message. */
  commands?: { keyword: string; action: string; label: string }[];
  rejected_labels: string[];
  policy: {
    hop_s: number;
    threshold: number;
    min_consecutive: number;
    cooldown_s: number;
    max_lag_s: number;
    max_frame_bytes: number;
    window_s: number;
  };
  retention: string;
};

/** Deprecated alias. Use ActionMap with the `actions` field. */
export type CommandMap = ActionMap;

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, { cache: "no-store", ...init });
  } catch {
    throw new ApiError("offline", 0, "The local API is not reachable.");
  }
  if (response.status === 204) return undefined as T;
  let body: unknown = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }
  if (!response.ok) {
    const error =
      typeof body === "object" && body !== null && "error" in body &&
      typeof body.error === "object" && body.error !== null
        ? (body.error as { code?: unknown; message?: unknown })
        : null;
    throw new ApiError(
      typeof error?.code === "string" ? error.code : `http_${response.status}`,
      response.status,
      typeof error?.message === "string" ? error.message : `Request failed with ${response.status}.`,
    );
  }
  return body as T;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => request<{ status: "ok"; model: HealthModel }>("/api/health"),
  activeModel: () => request<ActiveModel>("/api/models/active"),
  recordings: () => request<Recording[]>("/api/recordings"),
  recording: (id: string) => request<Recording>(`/api/recordings/${id}`),
  detections: (id: string) => request<DetectionResult>(`/api/recordings/${id}/detections`),
  analyze: (id: string, keywords: string[]) =>
    request<Recording>(`/api/recordings/${id}/analyze`, json({ keywords })),
  deleteRecording: (id: string) =>
    request<void>(`/api/recordings/${id}`, { method: "DELETE" }),
  audioUrl: (id: string) => `${API_BASE}/api/recordings/${id}/audio`,
  csvUrl: (id: string) => `${API_BASE}/api/recordings/${id}/detections.csv`,
  evaluations: () =>
    request<{ code: "runs_available" | "not_evaluated"; runs: RunSummary[] }>("/api/evaluations"),
  evaluation: (id: string) => request<RunDetail>(`/api/evaluations/${encodeURIComponent(id)}`),
  importEvaluations: () =>
    request<{ imported: string[]; rejected: { source: string; code: string }[] }>(
      "/api/evaluations/import",
      { method: "POST" },
    ),
  deleteEvaluation: (id: string) =>
    request<void>(`/api/evaluations/${encodeURIComponent(id)}`, { method: "DELETE" }),
  compareEvaluations: (runA: string, runB: string) =>
    request<Comparison>(
      `/api/evaluations/compare?run_a=${encodeURIComponent(runA)}&run_b=${encodeURIComponent(runB)}`,
    ),
  actions: () => request<ActionMap>("/api/actions"),
  /** Deprecated alias for actions(). Kept for backward compatibility. */
  commands: () => request<CommandMap>("/api/commands"),
};

// fetch cannot report upload progress, so uploads use XMLHttpRequest.
export function uploadRecording(
  file: File,
  onProgress: (fraction: number) => void,
  signal?: AbortSignal,
): Promise<Recording> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}/api/recordings`);
    xhr.responseType = "json";
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(event.loaded / event.total);
    };
    xhr.onload = () => {
      const body = xhr.response as { error?: { code?: string; message?: string } } | null;
      if (xhr.status === 201) resolve(xhr.response as Recording);
      else
        reject(
          new ApiError(
            body?.error?.code ?? `http_${xhr.status}`,
            xhr.status,
            body?.error?.message ?? "Upload failed.",
          ),
        );
    };
    xhr.onerror = () => reject(new ApiError("offline", 0, "The local API is not reachable."));
    xhr.onabort = () => reject(new ApiError("upload_cancelled", 0, "Upload cancelled."));
    signal?.addEventListener("abort", () => xhr.abort(), { once: true });
    const form = new FormData();
    form.append("file", file, file.name);
    xhr.send(form);
  });
}
