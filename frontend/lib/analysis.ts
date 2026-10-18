// Analysis job state as the reviewer sees it, derived from durable backend state.
import type { DetectionResult, Recording } from "./api.ts";
import { DEPLOYED } from "./deploy.ts";
import { isWavProblem, type WavInfo, type WavProblem } from "./wav.ts";

// The hosted API sits behind Vercel's 4.5 MB request body limit.
export const MAX_UPLOAD_BYTES = (DEPLOYED ? 4 : 40) * 1024 * 1024;
export const MAX_DURATION_S = 600;
const FILENAME = /^[\p{L}\p{N}_ .()-]+\.wav$/iu;

export type Phase = "uploaded" | "queued" | "running" | "completed" | "empty" | "failed";

export function phaseOf(recording: Recording, result: DetectionResult | null): Phase {
  if (recording.status !== "completed") return recording.status;
  if (result !== null && result.status === "completed" && result.detections.length === 0) {
    return "empty";
  }
  return "completed";
}

export function isActive(recording: Recording): boolean {
  return recording.status === "queued" || recording.status === "running";
}

export const PHASE_COPY: Record<Phase, { label: string; detail: string }> = {
  uploaded: { label: "Ready to analyze", detail: "Pick Keywords and run analysis." },
  queued: {
    label: "Queued",
    detail: "Waiting for the single analysis worker. Leaving this page does not cancel the job.",
  },
  running: {
    label: "Running",
    detail: "Scoring 1 s windows. Leaving this page does not cancel the job.",
  },
  completed: { label: "Completed", detail: "Detections are ready to review." },
  empty: { label: "No Detections", detail: "No selected Keyword reached the frozen threshold." },
  failed: { label: "Failed", detail: "The job stopped. See the error below." },
};

// Poll quickly while a job is fresh, then back off.
export function pollDelay(attempt: number): number {
  return Math.min(3000, Math.round(750 * 1.25 ** attempt));
}

export type FileCheck = { code: string; message: string };

// Mirrors the API limits so the reviewer is told before an upload is wasted.
export function precheck(
  file: { name: string; size: number },
  header: WavInfo | WavProblem,
): FileCheck[] {
  const problems: FileCheck[] = [];
  if (file.name.length > 200 || !FILENAME.test(file.name)) {
    problems.push({
      code: "invalid_filename",
      message: "Use a plain filename ending in .wav (letters, numbers, spaces, . - _ ( )).",
    });
  }
  if (file.size === 0) problems.push({ code: "invalid_wav", message: "The file is empty." });
  if (file.size > MAX_UPLOAD_BYTES) {
    problems.push({
      code: "upload_too_large",
      message: `The file is larger than ${MAX_UPLOAD_BYTES / 1024 / 1024} MiB.`,
    });
  }
  if (file.size > 0 && isWavProblem(header)) {
    problems.push({ code: header.code, message: header.message });
  } else if (!isWavProblem(header) && header.durationS > MAX_DURATION_S) {
    problems.push({ code: "recording_too_long", message: "The recording is longer than 10 minutes." });
  }
  return problems;
}
