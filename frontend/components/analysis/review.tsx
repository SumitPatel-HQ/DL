"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, errorCode, type Detection, type DetectionResult, type Recording } from "../../lib/api.ts";
import { isActive, PHASE_COPY, phaseOf, pollDelay, type Phase } from "../../lib/analysis.ts";
import {
  detectionAt, filterDetections, neighbor, segment, sortDetections, type SortDirection, type SortKey,
} from "../../lib/detections.ts";
import { formatClock, formatDuration, formatTimestamp, shortHash } from "../../lib/format.ts";
import {
  CheckIcon, NextIcon, PauseIcon, PlayIcon, PrevIcon, SearchIcon, TrashIcon, WaveIcon, XIcon,
} from "../icons";
import type { Backend } from "../shell/backend";
import { Banner, EmptyState, ErrorBanner, kwStyle } from "../ui";
import { DetectionTable } from "./detection-table";
import { Waveform } from "./waveform";

const STEPS = ["Uploaded", "Queued", "Running", "Completed"];
const STEP_INDEX: Record<Phase, number> = { uploaded: 0, queued: 1, running: 2, completed: 3, empty: 3, failed: 3 };

function JobStepper({ phase, analyzed }: { phase: Phase; analyzed: boolean }) {
  const current = STEP_INDEX[phase];
  const last = phase === "failed" ? "Failed" : phase === "empty" ? "No Detections" : "Completed";
  return (
    <div className="stepper" aria-hidden="true">
      {STEPS.map((label, index) => {
        const done = analyzed ? index < current : index === 0 && phase !== "uploaded";
        const isCurrent = analyzed ? index === current : index === 0;
        const tone = isCurrent && index === 3 ? (phase === "failed" ? " fail" : " ok") : "";
        return (
          <span key={label} style={{ display: "inline-flex", alignItems: "center" }}>
            {index > 0 ? <span className={`step-line${done || isCurrent ? " done" : ""}`} /> : null}
            <span className={`step${done ? " done" : ""}${isCurrent ? ` current${tone}` : ""}`}>
              <span className="bullet">
                {phase === "failed" && index === 3 ? <XIcon size={12} /> : done || (isCurrent && index === 3) ? <CheckIcon size={12} /> : index + 1}
              </span>
              {index === 3 ? last : label}
            </span>
          </span>
        );
      })}
    </div>
  );
}

export function RecordingReview({
  recording,
  backend,
  onChanged,
  onDeleted,
}: {
  recording: Recording;
  backend: Backend;
  onChanged: (recording: Recording) => void;
  onDeleted: (id: string) => void;
}) {
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [resultError, setResultError] = useState<string | null>(null);
  const [reload, setReload] = useState(0);
  const [picked, setPicked] = useState<string[] | null>(null);
  const [filter, setFilter] = useState<string[] | null>(null);
  const [sort, setSort] = useState<{ key: SortKey; direction: SortDirection }>({ key: "start", direction: "ascending" });
  const [actionError, setActionError] = useState<{ code: string; retry?: () => void } | null>(null);
  const [pollError, setPollError] = useState<string | null>(null);
  const [busy, setBusy] = useState<"analyze" | "delete" | null>(null);
  const [activeId, setActiveId] = useState<number | null>(null);
  const [hoverId, setHoverId] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const audioRef = useRef<HTMLAudioElement>(null);
  const clockRef = useRef<HTMLSpanElement>(null);
  const segmentRef = useRef<{ id: number; from: number; to: number } | null>(null);
  const dialogRef = useRef<HTMLDialogElement>(null);

  const ready = backend.status === "ready";
  // No fallback to a static list: without the active model there are no
  // keywords to offer. Never default to stale recording.keywords either;
  // the default is always the current model's keywords.
  const modelKeywords: string[] | null = ready ? backend.model.keywords : null;
  const chosen = picked ?? modelKeywords ?? [];
  const active = isActive(recording);
  const shownResult = recording.status === "completed" && result?.status === "completed" ? result : null;
  const phase = phaseOf(recording, shownResult);
  const analyzed = recording.model_id !== null || recording.status !== "uploaded";
  const detections = useMemo(() => shownResult?.detections ?? [], [shownResult]);
  const visible = useMemo(() => new Set(filter ?? recording.keywords), [filter, recording.keywords]);
  const rows = useMemo(
    () => sortDetections(filterDetections(detections, visible), sort.key, sort.direction),
    [detections, visible, sort],
  );
  const shownDetections = useMemo(() => filterDetections(detections, visible), [detections, visible]);

  // Poll durable job state until it is terminal; navigation never cancels the job.
  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    let attempt = 0;
    let timer = 0;
    const schedule = () => {
      timer = window.setTimeout(poll, pollDelay(attempt++));
    };
    const poll = async () => {
      try {
        const next = await api.recording(recording.id);
        if (cancelled) return;
        setPollError(null);
        onChanged(next);
        if (isActive(next)) schedule();
      } catch (error) {
        if (cancelled) return;
        const code = errorCode(error);
        if (code === "recording_not_found") onDeleted(recording.id);
        else {
          setPollError(code);
          schedule();
        }
      }
    };
    schedule();
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [active, recording.id, recording.status, onChanged, onDeleted]);

  useEffect(() => {
    if (recording.status !== "completed") return;
    let cancelled = false;
    api
      .detections(recording.id)
      .then((next) => {
        if (cancelled) return;
        setResult(next);
        setResultError(null);
      })
      .catch((error: unknown) => {
        if (!cancelled) setResultError(errorCode(error));
      });
    return () => {
      cancelled = true;
    };
  }, [recording.id, recording.status, reload]);

  // One clock drives the readout, segment stop, and the linked region/row.
  useEffect(() => {
    let frame = 0;
    const loop = () => {
      const element = audioRef.current;
      if (element) {
        const t = element.currentTime;
        const seg = segmentRef.current;
        if (seg && t >= seg.to) {
          element.pause();
          segmentRef.current = null;
        }
        if (clockRef.current) clockRef.current.textContent = formatClock(t);
        const current = segmentRef.current;
        const id = current && t >= current.from && t < current.to ? current.id : detectionAt(shownDetections, t)?.id ?? null;
        setActiveId((previous) => (previous === id ? previous : id));
      }
      frame = requestAnimationFrame(loop);
    };
    frame = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(frame);
  }, [shownDetections]);

  const play = useCallback(
    (detection: Detection) => {
      const element = audioRef.current;
      if (!element) return;
      const bounds = segment(detection, recording.duration_s);
      segmentRef.current = { id: detection.id, ...bounds };
      element.currentTime = bounds.from;
      void element.play().catch(() => setActionError({ code: "audio_unavailable" }));
    },
    [recording.duration_s],
  );

  const seek = useCallback(
    (seconds: number) => {
      const element = audioRef.current;
      if (!element) return;
      segmentRef.current = null;
      element.currentTime = Math.min(Math.max(0, seconds), recording.duration_s);
    },
    [recording.duration_s],
  );

  const toggle = useCallback(() => {
    const element = audioRef.current;
    if (!element) return;
    if (element.paused) {
      segmentRef.current = null;
      void element.play().catch(() => setActionError({ code: "audio_unavailable" }));
    } else element.pause();
  }, []);

  const jump = useCallback(
    (direction: 1 | -1) => {
      const element = audioRef.current;
      const target = neighbor(shownDetections, element?.currentTime ?? 0, direction);
      if (target) play(target);
    },
    [shownDetections, play],
  );

  // Keyboard parity with the waveform: Space, arrows, and J/K.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (event.altKey || event.ctrlKey || event.metaKey) return;
      if (target?.closest("input, textarea, select, dialog")) return;
      const element = audioRef.current;
      if (!element) return;
      if (event.key === " " && !target?.closest("button, a")) {
        event.preventDefault();
        toggle();
      } else if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
        if (target?.closest("button, a, [role=tab]")) return;
        event.preventDefault();
        seek(element.currentTime + (event.key === "ArrowLeft" ? -1 : 1) * (event.shiftKey ? 5 : 1));
      } else if (event.key === "j" || event.key === "J") jump(-1);
      else if (event.key === "k" || event.key === "K") jump(1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggle, seek, jump]);

  const analyze = async () => {
    setBusy("analyze");
    setActionError(null);
    try {
      onChanged(await api.analyze(recording.id, chosen));
      setFilter(null);
    } catch (error) {
      setActionError({ code: errorCode(error), retry: analyze });
    } finally {
      setBusy(null);
    }
  };

  const remove = async () => {
    setBusy("delete");
    setActionError(null);
    try {
      audioRef.current?.pause();
      await api.deleteRecording(recording.id);
      dialogRef.current?.close();
      onDeleted(recording.id);
    } catch (error) {
      const code = errorCode(error);
      dialogRef.current?.close();
      if (code === "recording_not_found") onDeleted(recording.id);
      else setActionError({ code, retry: remove });
    } finally {
      setBusy(null);
    }
  };

  const toggleKeyword = (keyword: string) => {
    const order = modelKeywords ?? chosen;
    return setPicked(
      chosen.includes(keyword)
        ? chosen.filter((k) => k !== keyword)
        : order.filter((k) => k === keyword || chosen.includes(k)),
    );
  };

  const toggleFilter = (keyword: string) => {
    const current = filter ?? recording.keywords;
    setFilter(current.includes(keyword) ? current.filter((k) => k !== keyword) : [...current, keyword]);
  };

  const counts = useMemo(() => {
    const map = new Map<string, number>();
    for (const d of detections) map.set(d.keyword, (map.get(d.keyword) ?? 0) + 1);
    return map;
  }, [detections]);

  const detector = recording.detector;
  const otherModel = ready && recording.model_id !== null && recording.model_id !== backend.model.id;

  return (
    <div className="main-col">
      <section className="card" aria-labelledby="rec-title">
        <div className="rec-head">
          <div style={{ minWidth: 0 }}>
            <p className="eyebrow">Recording</p>
            <h2 id="rec-title">{recording.filename}</h2>
            <div className="meta">
              <span><b>{formatDuration(recording.duration_s)}</b></span>
              <span><b>{(recording.sample_rate / 1000).toLocaleString()} kHz</b> · {recording.channels === 1 ? "mono" : "stereo"}</span>
              <span>Uploaded {formatTimestamp(recording.created_at)}</span>
              {recording.model_id ? <span>Checkpoint <code>{shortHash(recording.model_id, 8)}</code></span> : null}
            </div>
          </div>
          <button
            type="button"
            className="btn danger"
            onClick={() => dialogRef.current?.showModal()}
            disabled={busy !== null}
            title={
              active
                ? "Delete this recording now; queued or running analysis is cancelled"
                : undefined
            }
            aria-label={
              active
                ? "Delete this recording and cancel its queued or running analysis"
                : "Delete this recording"
            }
          >
            <TrashIcon size={15} /> Delete
          </button>
        </div>
        <div className="analyze-row" style={{ borderTop: "1px solid var(--border)" }}>
          <div className="row" role="group" aria-label="Keywords to detect">
            <span className="label">Keywords{ready && backend.model.version ? ` · ${backend.model.version}` : ""}</span>
            {modelKeywords === null || modelKeywords.length === 0 ? (
              <span className="table-note" role="status">
                {ready
                  ? "The active model reports no keywords, so there is nothing to select."
                  : "No model is active, so no keywords are available. Connect the local API and model to select keywords."}
              </span>
            ) : (
              modelKeywords.map((keyword) => (
              <button
                key={keyword}
                type="button"
                className="chip"
                style={kwStyle(keyword)}
                aria-pressed={chosen.includes(keyword)}
                onClick={() => toggleKeyword(keyword)}
                disabled={!ready || active}
              >
                <span className="swatch" />
                {keyword}
              </button>
              ))
            )}
            <span className="shell-spacer" />
            <button
              type="button"
              className="btn primary"
              onClick={analyze}
              disabled={!ready || active || busy !== null || chosen.length === 0}
            >
              <SearchIcon size={15} />
              {busy === "analyze" ? "Queuing…" : analyzed ? "Analyze again" : "Analyze"}
            </button>
          </div>
          <div className="row" style={{ justifyContent: "space-between", gap: 16 }}>
            <JobStepper phase={phase} analyzed={analyzed} />
            <p className="job-detail" role="status" aria-live="polite">
              <span className="sr-only">Analysis status: {PHASE_COPY[phase].label}. </span>
              {PHASE_COPY[phase].detail}
            </p>
          </div>
          {active ? (
            <div className="progress indeterminate" aria-hidden="true"><div /></div>
          ) : null}
          {!ready && backend.status === "unavailable" ? (
            <ErrorBanner code={backend.code} />
          ) : null}
          {pollError ? <ErrorBanner code={pollError} /> : null}
          {actionError ? (
            <ErrorBanner
              code={actionError.code}
              handlers={{ wait_retry: actionError.retry, retry: actionError.retry, refresh: () => onDeleted(recording.id) }}
            />
          ) : null}
          {recording.status === "failed" && recording.error_code ? (
            <ErrorBanner code={recording.error_code} handlers={{ reanalyze: analyze }} />
          ) : null}
          {otherModel ? (
            <Banner tone="info" title="Detections came from a different checkpoint">
              These Detections used checkpoint <code>{shortHash(recording.model_id ?? "", 8)}</code>, not the
              active model. Analyze again to use the active model.
            </Banner>
          ) : null}
        </div>
      </section>

      <section className="card" aria-label="Waveform">
        <div className="wave-toolbar">
          <button type="button" className="btn icon" onClick={() => jump(-1)} aria-label="Previous Detection (J)" disabled={!shownDetections.length}>
            <PrevIcon size={15} />
          </button>
          <button type="button" className="btn icon primary" onClick={toggle} aria-label={playing ? "Pause (Space)" : "Play (Space)"}>
            {playing ? <PauseIcon size={15} /> : <PlayIcon size={15} />}
          </button>
          <button type="button" className="btn icon" onClick={() => jump(1)} aria-label="Next Detection (K)" disabled={!shownDetections.length}>
            <NextIcon size={15} />
          </button>
          <span className="clock" aria-label="Playback position">
            <span ref={clockRef}>0:00.000</span> <span className="total">/ {formatClock(recording.duration_s)}</span>
          </span>
          <span className="shell-spacer" />
          <div className="shortcuts" aria-label="Keyboard shortcuts">
            <span><span className="kbd">Space</span> play</span>
            <span><span className="kbd">←</span><span className="kbd">→</span> ±1 s</span>
            <span><span className="kbd">⇧</span>+arrows ±5 s</span>
            <span><span className="kbd">J</span><span className="kbd">K</span> prev/next</span>
          </div>
        </div>
        <Waveform
          recordingId={recording.id}
          duration={recording.duration_s}
          detections={detections}
          visible={visible}
          activeId={activeId}
          hoverId={hoverId}
          audioRef={audioRef}
          onRegion={play}
          onHover={setHoverId}
          onSeek={seek}
        />
        <audio
          ref={audioRef}
          src={api.audioUrl(recording.id)}
          preload="metadata"
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onEnded={() => setPlaying(false)}
        />
      </section>

      <section className="card" aria-labelledby="det-title">
        <div className="card-head">
          <div>
            <h3 id="det-title">
              Detections{shownResult ? <span className="sub"> · {rows.length} of {detections.length} shown</span> : null}
            </h3>
            {detector ? (
              <p className="sub">
                Frozen detector · stride {detector.stride_s.toFixed(2)} s · threshold {detector.threshold.toFixed(2)} ·
                min consecutive {detector.min_consecutive} · suppression {detector.suppression_s.toFixed(1)} s
              </p>
            ) : null}
          </div>
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            {shownResult && detections.length ? (
              <div className="filters" role="group" aria-label="Filter by Keyword">
                {recording.keywords.map((keyword) => (
                  <button
                    key={keyword}
                    type="button"
                    className="chip"
                    style={kwStyle(keyword)}
                    aria-pressed={visible.has(keyword)}
                    onClick={() => toggleFilter(keyword)}
                  >
                    <span className="swatch" />
                    {keyword} <span className="mono" style={{ color: "var(--muted)" }}>{counts.get(keyword) ?? 0}</span>
                  </button>
                ))}
              </div>
            ) : null}
            {recording.status === "completed" ? (
              <a
                className="btn small"
                href={api.csvUrl(recording.id)}
                download={`${recording.id}.csv`}
              >
                Export CSV
              </a>
            ) : null}
          </div>
        </div>
        {resultError ? (
          <div className="card-body"><ErrorBanner code={resultError} handlers={{ retry: () => setReload((n) => n + 1) }} /></div>
        ) : phase === "completed" && shownResult ? (
          rows.length ? (
            <DetectionTable
              detections={rows}
              sort={sort}
              onSort={(key) =>
                setSort((current) => ({
                  key,
                  direction: current.key === key && current.direction === "ascending" ? "descending" : "ascending",
                }))
              }
              activeId={activeId}
              hoverId={hoverId}
              onHover={setHoverId}
              onPlay={play}
              onSeek={seek}
            />
          ) : (
            <EmptyState icon={<SearchIcon size={22} />} title="All Detections filtered out">
              Turn a Keyword filter back on to see its Detections.
            </EmptyState>
          )
        ) : phase === "empty" ? (
          <EmptyState icon={<WaveIcon size={22} />} title="No Detections">
            No selected Keyword ({recording.keywords.join(", ")}) reached the frozen threshold
            {detector ? ` of ${detector.threshold.toFixed(2)}` : ""}. This is a measured negative result, not a
            guessed label. Try other Keywords, or check that the speech is audible. <code>no_detections</code>
          </EmptyState>
        ) : active ? (
          <div className="card-body" style={{ display: "grid", gap: 10 }} aria-hidden="true">
            {[0, 1, 2].map((i) => <div key={i} className="skeleton" style={{ height: 34, opacity: 1 - i * 0.25 }} />)}
          </div>
        ) : recording.status === "failed" ? (
          <EmptyState icon={<XIcon size={22} />} title="No Detections from a failed job">
            The last analysis did not finish, so there is nothing to review. Run analysis again.
          </EmptyState>
        ) : phase === "completed" ? (
          <div className="card-body"><div className="skeleton" style={{ height: 34 }} /></div>
        ) : (
          <EmptyState icon={<SearchIcon size={22} />} title="Not analyzed yet">
            Pick one or more trained Keywords above and press Analyze. Detections appear here and on the
            waveform.
          </EmptyState>
        )}
        <div className="card-foot">
          Confidence is the classifier&apos;s score for the window, not a measure of correctness. The white tick on
          each bar marks the threshold. Times are approximate window spans.
        </div>
      </section>

      <dialog ref={dialogRef} className="modal" aria-labelledby="delete-title">
        <div className="modal-body">
          <h2 id="delete-title">Delete this Recording?</h2>
          <p>
            <strong>{recording.filename}</strong> will be removed from local storage. This cannot be undone.
            {active ? " Its queued or running analysis is cancelled first." : ""}
          </p>
          <ul>
            <li>Audio bytes ({formatDuration(recording.duration_s)} WAV)</li>
            <li>Recording metadata and analysis settings</li>
            <li>{detections.length} child Detection{detections.length === 1 ? "" : "s"}</li>
          </ul>
        </div>
        <div className="modal-actions">
          <button type="button" className="btn ghost" onClick={() => dialogRef.current?.close()} disabled={busy === "delete"}>
            Cancel
          </button>
          <button type="button" className="btn solid-danger" onClick={remove} disabled={busy === "delete"}>
            <TrashIcon size={15} /> {busy === "delete" ? "Deleting…" : "Delete permanently"}
          </button>
        </div>
      </dialog>
    </div>
  );
}
