"use client";

import { useCallback, useEffect, useRef, useState, type DragEvent } from "react";
import { api, errorCode, uploadRecording, type Recording } from "../../lib/api.ts";
import { isActive, MAX_DURATION_S, PHASE_COPY, precheck, type FileCheck } from "../../lib/analysis.ts";
import { describeError } from "../../lib/errors.ts";
import { formatBytes, formatDuration } from "../../lib/format.ts";
import { inspectWav } from "../../lib/wav.ts";
import { UploadIcon, WaveIcon, XIcon } from "../icons";
import { useBackend } from "../shell/backend";
import { EmptyState, ErrorBanner, ExperimentalBanner } from "../ui";
import { RecordingReview } from "./review";

type Upload = {
  key: number;
  name: string;
  size: number;
  progress: number;
  error: string | null;
  problems: FileCheck[];
};

const STATUS_DOT: Record<Recording["status"], string> = {
  uploaded: "",
  queued: "warn",
  running: "busy",
  completed: "ok",
  failed: "critical",
};

export function AnalysisWorkbench() {
  const { backend, refresh } = useBackend();
  const [recordings, setRecordings] = useState<Recording[] | null>(null);
  const [listError, setListError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [uploads, setUploads] = useState<Upload[]>([]);
  const [over, setOver] = useState(false);
  const [reload, setReload] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const uploadKey = useRef(0);
  const online = backend.status !== "offline";

  useEffect(() => {
    if (!online) return;
    let cancelled = false;
    api
      .recordings()
      .then((list) => {
        if (cancelled) return;
        setRecordings([...list].reverse());
        setListError(null);
      })
      .catch((error: unknown) => {
        if (!cancelled) setListError(errorCode(error));
      });
    return () => {
      cancelled = true;
    };
  }, [online, reload]);

  // Background jobs on other Recordings keep their list status fresh.
  const othersActive = (recordings ?? []).some((r) => r.id !== selectedId && isActive(r));
  useEffect(() => {
    if (!othersActive) return;
    const timer = window.setTimeout(() => setReload((n) => n + 1), 2500);
    return () => window.clearTimeout(timer);
  }, [othersActive, recordings]);

  const selected = recordings?.find((r) => r.id === selectedId) ?? recordings?.[0] ?? null;

  const onChanged = useCallback((next: Recording) => {
    setRecordings((list) => list?.map((r) => (r.id === next.id ? next : r)) ?? list);
  }, []);

  const onDeleted = useCallback((id: string) => {
    setRecordings((list) => list?.filter((r) => r.id !== id) ?? list);
    setSelectedId(null);
  }, []);

  const start = async (file: File) => {
    const key = ++uploadKey.current;
    const header = inspectWav(await file.slice(0, 65536).arrayBuffer());
    const problems = precheck(file, header);
    const entry: Upload = { key, name: file.name, size: file.size, progress: 0, error: null, problems };
    setUploads((list) => [entry, ...list.filter((u) => u.error === null && u.problems.length === 0)]);
    if (problems.length) return;
    try {
      const recording = await uploadRecording(file, (progress) =>
        setUploads((list) => list.map((u) => (u.key === key ? { ...u, progress } : u))),
      );
      setUploads((list) => list.filter((u) => u.key !== key));
      setRecordings((list) => [recording, ...(list ?? [])]);
      setSelectedId(recording.id);
    } catch (error) {
      setUploads((list) => list.map((u) => (u.key === key ? { ...u, error: errorCode(error) } : u)));
    }
  };

  const onFiles = (files: FileList | null) => {
    if (files?.[0]) void start(files[0]);
    if (inputRef.current) inputRef.current.value = "";
  };

  const onDrop = (event: DragEvent) => {
    event.preventDefault();
    setOver(false);
    onFiles(event.dataTransfer.files);
  };

  const dismiss = (key: number) => setUploads((list) => list.filter((u) => u.key !== key));

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <p className="eyebrow">Audio Analysis</p>
          <h1>Review Detections in a Recording</h1>
          <p>
            Upload a PCM WAV, choose trained Keywords, and verify each Detection by listening. Waveform regions,
            table rows, and the playback clock stay linked.
          </p>
        </div>
      </div>
      <div style={{ display: "grid", gap: 12, marginBottom: 18 }}>
        <ExperimentalBanner />
        {backend.status === "offline" ? (
          <ErrorBanner code="offline" handlers={{ retry: () => { refresh(); setReload((n) => n + 1); } }} />
        ) : listError ? (
          <ErrorBanner code={listError} handlers={{ retry: () => setReload((n) => n + 1) }} />
        ) : null}
      </div>
      <div className="workbench">
        <aside className="sidebar" aria-label="Recordings">
          <div
            className={`drop${over ? " over" : ""}`}
            role="button"
            tabIndex={0}
            aria-disabled={!online}
            onClick={() => online && inputRef.current?.click()}
            onKeyDown={(e) => {
              if ((e.key === "Enter" || e.key === " ") && online) {
                e.preventDefault();
                inputRef.current?.click();
              }
            }}
            onDragOver={(e) => {
              e.preventDefault();
              setOver(true);
            }}
            onDragLeave={() => setOver(false)}
            onDrop={onDrop}
          >
            <UploadIcon size={22} />
            <strong>Drop a WAV or browse</strong>
            <span>PCM 8-96 kHz · mono/stereo · ≤ 40 MiB · ≤ {MAX_DURATION_S / 60} min</span>
            <input
              ref={inputRef}
              type="file"
              accept=".wav,audio/wav,audio/x-wav"
              hidden
              onChange={(e) => onFiles(e.target.files)}
            />
          </div>
          {uploads.map((upload) => (
            <div key={upload.key} className="card" style={{ padding: 14, display: "grid", gap: 8 }} role="status">
              <div style={{ display: "flex", justifyContent: "space-between", gap: 8, alignItems: "center" }}>
                <strong style={{ fontSize: "0.88rem", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {upload.name}
                </strong>
                {upload.error || upload.problems.length ? (
                  <button type="button" className="btn small icon ghost" aria-label="Dismiss" onClick={() => dismiss(upload.key)}>
                    <XIcon size={14} />
                  </button>
                ) : null}
              </div>
              {upload.problems.length ? (
                <>
                  <span className="tag critical">Not uploaded · fails local checks</span>
                  <ul className="list-plain" style={{ fontSize: "0.82rem" }}>
                    {upload.problems.map((p) => <li key={p.code}>{p.message}</li>)}
                  </ul>
                  <button type="button" className="btn small" onClick={() => inputRef.current?.click()}>Choose another file</button>
                </>
              ) : upload.error ? (
                <>
                  <span className="tag critical">{describeError(upload.error).title}</span>
                  <p style={{ fontSize: "0.82rem", color: "var(--text-2)" }}>{describeError(upload.error).detail}</p>
                  <button type="button" className="btn small" onClick={() => inputRef.current?.click()}>Choose another file</button>
                </>
              ) : (
                <>
                  <div className="progress"><div style={{ width: `${upload.progress * 100}%` }} /></div>
                  <span className="table-note">
                    Uploading · {Math.round(upload.progress * 100)}% of {formatBytes(upload.size)}
                    {upload.progress >= 1 ? " · validating on server" : ""}
                  </span>
                </>
              )}
            </div>
          ))}
          <div className="card">
            <div className="card-head">
              <h3>Recordings</h3>
              <span className="sub">{recordings ? recordings.length : "…"} stored locally</span>
            </div>
            <div className="rec-list">
              {recordings === null && !online ? (
                <p className="table-note" style={{ padding: "14px 10px" }}>Recordings load when the API is back.</p>
              ) : recordings === null ? (
                [0, 1, 2].map((i) => <div key={i} className="skeleton" style={{ height: 52, margin: 4 }} />)
              ) : recordings.length === 0 ? (
                <p className="table-note" style={{ padding: "14px 10px" }}>No Recordings yet. Upload a WAV to begin.</p>
              ) : (
                recordings.map((r) => (
                  <button
                    key={r.id}
                    type="button"
                    className="rec-item"
                    aria-current={r.id === selected?.id ? "true" : undefined}
                    onClick={() => setSelectedId(r.id)}
                  >
                    <span className="name">{r.filename}</span>
                    <span className="meta">
                      <span>{formatDuration(r.duration_s)}</span>
                      <span style={{ display: "inline-flex", gap: 6, alignItems: "center" }}>
                        <span className={`dot ${STATUS_DOT[r.status]}`} />
                        {r.status === "uploaded" ? "Not analyzed" : PHASE_COPY[r.status].label}
                      </span>
                    </span>
                  </button>
                ))
              )}
            </div>
          </div>
        </aside>
        {selected ? (
          <RecordingReview
            key={selected.id}
            recording={selected}
            backend={backend}
            onChanged={onChanged}
            onDeleted={onDeleted}
          />
        ) : (
          <section className="card state-panel">
            <EmptyState
              icon={<WaveIcon size={24} />}
              title={!online ? "Backend offline" : recordings === null ? "Loading Recordings…" : "No Recording selected"}
              actions={
                online ? (
                  <button type="button" className="btn primary" onClick={() => inputRef.current?.click()}>
                    <UploadIcon size={15} /> Upload WAV
                  </button>
                ) : null
              }
            >
              {!online
                ? "Start the API with npm run api, then press Retry above. Stored Recordings are kept on disk."
                : "Audio stays on this machine. The API stores the WAV under local data storage until you delete it."}
            </EmptyState>
          </section>
        )}
      </div>
    </div>
  );
}
