// Live session protocol and state. The mic opens only on Start and is released
// on Stop, disconnect, error, or tab-hide; no state keeps listening.
import { isAction, type Action } from "./stage.ts";

export type ServerMessage =
  | {
      type: "ready";
      session_id: string;
      keywords: string[];
      commands: { keyword: string; action: string; label: string }[];
      policy: { threshold: number; min_consecutive: number; cooldown_s: number; hop_s: number };
    }
  | { type: "frame"; t_s: number; label: string; score: number }
  | {
      type: "decision";
      id: number;
      t_s: number;
      label: string;
      score: number;
      accepted: boolean;
      reason: "accepted" | "cooldown" | "low_confidence" | "unknown" | "silence";
      action: Action | null;
      processing_ms: number;
    }
  | { type: "backpressure"; dropped_s: number }
  | { type: "error"; code: string; message: string }
  | { type: "stopped" };

const REASONS = ["accepted", "cooldown", "low_confidence", "unknown", "silence"];

export function parseServerMessage(text: string): ServerMessage | null {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    return null;
  }
  if (typeof value !== "object" || value === null || !("type" in value)) return null;
  const message = value as Record<string, unknown>;
  switch (message.type) {
    case "ready":
      return typeof message.session_id === "string" && Array.isArray(message.keywords)
        ? (message as ServerMessage)
        : null;
    case "frame":
      return typeof message.label === "string" && typeof message.score === "number"
        ? (message as ServerMessage)
        : null;
    case "decision":
      return typeof message.id === "number" &&
        typeof message.label === "string" &&
        typeof message.score === "number" &&
        typeof message.accepted === "boolean" &&
        REASONS.includes(message.reason as string) &&
        (message.action === null || isAction(message.action)) &&
        // An action only ever rides on an accepted decision.
        (message.accepted === (message.action !== null))
        ? (message as ServerMessage)
        : null;
    case "backpressure":
      return typeof message.dropped_s === "number" ? (message as ServerMessage) : null;
    case "error":
      return typeof message.code === "string" ? (message as ServerMessage) : null;
    case "stopped":
      return { type: "stopped" };
    default:
      return null;
  }
}

// getUserMedia failures branch on the DOMException name, never on its message.
export function micErrorCode(error: unknown, permission: PermissionState | null): string {
  const name = error instanceof Error || error instanceof DOMException ? error.name : "";
  switch (name) {
    case "NotAllowedError":
      return permission === "denied" ? "mic_blocked" : "mic_denied";
    case "SecurityError":
      return "insecure_context";
    case "NotFoundError":
    case "OverconstrainedError":
      return "no_device";
    case "NotReadableError":
    case "AbortError":
      return "device_busy";
    default:
      return "mic_failed";
  }
}

export type SessionStatus = "idle" | "requesting" | "connecting" | "listening" | "error";

export type SessionState = {
  status: SessionStatus;
  error: string | null;
  stopReason: string | null;
  startedAt: number | null;
};

export type SessionEvent =
  | { type: "start" }
  | { type: "mic_ready" }
  | { type: "ready"; at: number }
  | { type: "stopped"; reason: string }
  | { type: "failed"; code: string };

export const IDLE: SessionState = { status: "idle", error: null, stopReason: null, startedAt: null };

export function sessionReducer(state: SessionState, event: SessionEvent): SessionState {
  switch (event.type) {
    case "start":
      return state.status === "idle" || state.status === "error"
        ? { status: "requesting", error: null, stopReason: null, startedAt: null }
        : state;
    case "mic_ready":
      return state.status === "requesting" ? { ...state, status: "connecting" } : state;
    case "ready":
      return state.status === "connecting" ? { ...state, status: "listening", startedAt: event.at } : state;
    case "stopped":
      return state.status === "error" ? state : { ...IDLE, stopReason: event.reason };
    case "failed":
      return { status: "error", error: event.code, stopReason: null, startedAt: null };
  }
}

export const REASON_COPY: Record<string, string> = {
  accepted: "Accepted",
  cooldown: "Cooldown: one action per utterance",
  low_confidence: "Below threshold or not consistent",
  unknown: "Unknown speech",
  silence: "Silence",
};

export type HistoryEntry = {
  key: string;
  source: "voice" | "manual" | "system";
  label: string;
  score: number | null;
  accepted: boolean;
  reason: string;
  result: string | null;
  processingMs: number | null;
  at: number;
};

export const HISTORY_LIMIT = 20;

// Newest first; consecutive Silence entries collapse so pauses don't flood the log.
export function addHistory(history: readonly HistoryEntry[], entry: HistoryEntry): HistoryEntry[] {
  if (entry.reason === "silence" && history[0]?.reason === "silence") {
    return [entry, ...history.slice(1)];
  }
  return [entry, ...history].slice(0, HISTORY_LIMIT);
}
