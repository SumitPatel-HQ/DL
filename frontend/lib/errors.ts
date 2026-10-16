// One line of cause plus one recovery for every stable error code the UI can meet.
export type Recovery =
  | "retry"
  | "choose_file"
  | "wait_retry"
  | "start_api"
  | "install_model"
  | "reanalyze"
  | "refresh"
  | "reload"
  | "enable_mic"
  | "mic_settings"
  | "start_again"
  | "import_runs"
  | "none";

export type ErrorCopy = { title: string; detail: string; recovery: Recovery };

const COPY: Record<string, ErrorCopy> = {
  offline: {
    title: "Backend offline",
    detail: "The local API on localhost:8000 is not answering. Start it with npm run api.",
    recovery: "retry",
  },
  invalid_wav: {
    title: "Corrupt or non-PCM WAV",
    detail: "The file is not valid uncompressed integer PCM, or its sample data is truncated.",
    recovery: "choose_file",
  },
  unsupported_wav: {
    title: "Unsupported WAV format",
    detail: "Use mono or stereo PCM WAV recorded at 8-96 kHz.",
    recovery: "choose_file",
  },
  invalid_filename: {
    title: "Filename not allowed",
    detail: "Rename the file to plain letters, numbers, spaces, dots, dashes or brackets ending in .wav.",
    recovery: "choose_file",
  },
  invalid_upload: {
    title: "Upload rejected",
    detail: "Send exactly one .wav file.",
    recovery: "choose_file",
  },
  upload_too_large: {
    title: "File too large",
    detail: "Recordings are limited to 40 MiB. Trim or downsample the WAV.",
    recovery: "choose_file",
  },
  recording_too_long: {
    title: "Recording too long",
    detail: "Recordings are limited to 10 minutes. Split the audio into shorter files.",
    recovery: "choose_file",
  },
  upload_disconnected: {
    title: "Upload interrupted",
    detail: "The connection dropped mid-upload and partial data was discarded.",
    recovery: "choose_file",
  },
  upload_cancelled: {
    title: "Upload cancelled",
    detail: "Nothing was stored.",
    recovery: "choose_file",
  },
  uploads_busy: {
    title: "Upload slots busy",
    detail: "Two uploads are already in progress. Wait for one to finish.",
    recovery: "wait_retry",
  },
  rate_limited: {
    title: "Too many requests",
    detail: "The API allows 60 uploads, analyses and deletions per minute.",
    recovery: "wait_retry",
  },
  queue_full: {
    title: "Analysis queue full",
    detail: "One Recording is running and two are waiting. Retry in a moment.",
    recovery: "wait_retry",
  },
  analysis_in_progress: {
    title: "Already analyzing",
    detail: "This Recording is queued or running. Wait for it to finish.",
    recovery: "none",
  },
  unsupported_keyword: {
    title: "Keyword not trained",
    detail: "Pick Keywords offered by the active model.",
    recovery: "none",
  },
  invalid_request: {
    title: "Request rejected",
    detail: "The request did not match the API contract. Reload the page.",
    recovery: "reload",
  },
  recording_not_found: {
    title: "Recording no longer exists",
    detail: "It was deleted, possibly from another tab.",
    recovery: "refresh",
  },
  audio_unavailable: {
    title: "Audio missing",
    detail: "Recording metadata exists but its audio bytes are gone. Delete the Recording.",
    recovery: "none",
  },
  delete_failed: {
    title: "Delete not finished",
    detail: "The audio file could not be removed yet. Retrying deletion is safe.",
    recovery: "retry",
  },
  analysis_failed: {
    title: "Analysis failed",
    detail: "Inference stopped with an error. The Recording is intact; run analysis again.",
    recovery: "reanalyze",
  },
  service_interrupted: {
    title: "Analysis interrupted",
    detail: "The API stopped before this job finished. Run analysis again.",
    recovery: "reanalyze",
  },
  persistence_error: {
    title: "Metadata write failed",
    detail: "Local storage refused a write. Free disk space, then run analysis again.",
    recovery: "reanalyze",
  },
  storage_error: {
    title: "Storage unavailable",
    detail: "A local storage operation failed. Check disk space and permissions.",
    recovery: "retry",
  },
  checkpoint_not_installed: {
    title: "No model installed",
    detail: "Train a checkpoint and set KWS_CHECKPOINT and KWS_DETECTOR, then restart the API.",
    recovery: "install_model",
  },
  model_incompatible: {
    title: "Model incompatible",
    detail: "The checkpoint and frozen detector config do not match. Install a matching pair.",
    recovery: "install_model",
  },
  vocabulary_not_reviewed: {
    title: "Vocabulary awaiting review",
    detail: "The expanded vocabulary needs reviewed Evaluation Runs. Restore the P0 checkpoint and its detector.",
    recovery: "install_model",
  },
  worker_unavailable: {
    title: "Worker waiting for storage",
    detail: "Metadata writes failed and the analysis worker is retrying.",
    recovery: "wait_retry",
  },
  evaluations_unavailable: {
    title: "Evaluation store unavailable",
    detail: "Stored Evaluation Runs could not be read.",
    recovery: "retry",
  },
  evaluation_not_found: {
    title: "Run deleted",
    detail: "This Evaluation Run no longer exists. Its views were cleared; nothing falls back.",
    recovery: "none",
  },
  comparison_kind_mismatch: {
    title: "Runs cannot be compared",
    detail: "Isolated classification and continuous-event numbers never share a view. Pick two isolated or two continuous runs.",
    recovery: "none",
  },
  comparison_dataset_mismatch: {
    title: "Runs use different datasets",
    detail: "Comparison needs identical dataset version, split version, and split. Pick two runs from the same Evaluation dataset.",
    recovery: "none",
  },
  comparison_provenance_mismatch: {
    title: "Runs are not a clean/noisy pair",
    detail: "A noise comparison needs the same checkpoint hash and the same dataset version, split version, and split. Only the dataset condition may differ.",
    recovery: "none",
  },
  comparison_protocol_mismatch: {
    title: "Detector protocol differs",
    detail: "Clean and noisy continuous runs must share the frozen detector configuration and tolerance.",
    recovery: "none",
  },
  mic_denied: {
    title: "Microphone permission denied",
    detail: "The browser did not get mic access. Press Start again and choose Allow.",
    recovery: "enable_mic",
  },
  mic_blocked: {
    title: "Microphone blocked for this site",
    detail:
      "The browser remembers the block and will not ask again. Click the lock or site icon in the address bar, set Microphone to Allow, then press Start.",
    recovery: "mic_settings",
  },
  no_device: {
    title: "No microphone found",
    detail: "Connect or enable an input device, then press Start.",
    recovery: "enable_mic",
  },
  device_busy: {
    title: "Microphone in use",
    detail: "Another app or the OS holds the device. Close it, then press Start.",
    recovery: "enable_mic",
  },
  insecure_context: {
    title: "Mic needs a secure page",
    detail: "Open the app at http://127.0.0.1:3000 or http://localhost:3000.",
    recovery: "reload",
  },
  mic_failed: {
    title: "Microphone failed to start",
    detail: "The audio pipeline could not start in this browser.",
    recovery: "enable_mic",
  },
  live_busy: {
    title: "Another live session is running",
    detail: "Only one mic session can run. Stop it in the other tab, then press Start.",
    recovery: "start_again",
  },
  disconnected: {
    title: "Live connection lost",
    detail: "The mic was released and the session closed. Nothing kept listening.",
    recovery: "start_again",
  },
  unsupported_version: {
    title: "Page out of date",
    detail: "The live protocol changed. Reload the page.",
    recovery: "reload",
  },
  unsupported_audio: {
    title: "Mic format not accepted",
    detail: "The browser mic rate is outside 8-96 kHz.",
    recovery: "reload",
  },
  invalid_frame: {
    title: "Audio frame rejected",
    detail: "The session closed after a malformed audio frame.",
    recovery: "start_again",
  },
  invalid_message: {
    title: "Live protocol error",
    detail: "The session closed after an unexpected message.",
    recovery: "start_again",
  },
  live_failed: {
    title: "Live scoring failed",
    detail: "The model raised an error during scoring. The session closed.",
    recovery: "start_again",
  },
};

export function describeError(code: string): ErrorCopy {
  return (
    COPY[code] ?? {
      title: "Unexpected error",
      detail: `The API returned ${code}.`,
      recovery: "retry",
    }
  );
}
