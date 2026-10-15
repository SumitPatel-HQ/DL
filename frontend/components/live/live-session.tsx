"use client";

import { useCallback, useEffect, useReducer, useRef, useState, type CSSProperties } from "react";
import { api, LIVE_URL, type CommandMap } from "../../lib/api.ts";
import { formatScore } from "../../lib/format.ts";
import { keywordColor, REJECTED_COLOR } from "../../lib/keywords.ts";
import { micPermission, openMic, type MicPipeline } from "../../lib/live/mic.ts";
import {
  addHistory, IDLE, micErrorCode, parseServerMessage, REASON_COPY, sessionReducer,
  type HistoryEntry, type ServerMessage,
} from "../../lib/live/session.ts";
import { applyAction, initialStage, KEY_BINDINGS, KEY_LABELS, type Action, type Stage } from "../../lib/live/stage.ts";
import { CheckIcon, LockIcon, MicIcon, StopIcon, XIcon } from "../icons";
import { useBackend } from "../shell/backend";
import { Banner, ErrorBanner } from "../ui";
import { CooldownBar, Elapsed, LevelMeter } from "./meters";
import { Player, SLIDES, SlideDeck } from "./stage";

type Policy = CommandMap["policy"];
type Decision = Extract<ServerMessage, { type: "decision" }>;

const DEFAULT_COMMANDS: CommandMap["commands"] = [
  { keyword: "right", action: "next_slide", label: "Next slide" },
  { keyword: "left", action: "previous_slide", label: "Previous slide" },
  { keyword: "go", action: "play", label: "Play" },
  { keyword: "stop", action: "pause", label: "Pause" },
  { keyword: "up", action: "volume_up", label: "Volume up" },
  { keyword: "down", action: "volume_down", label: "Volume down" },
];
const BACKPRESSURE_BYTES = 256 * 1024;
const STOP_REASONS: Record<string, string> = {
  user: "Stopped. The mic is released.",
  tab_hidden: "Stopped because the tab was hidden. The mic is released.",
  page_exit: "Stopped on page exit.",
};

let entryKey = 0;

export function LiveSession() {
  const { backend, refresh } = useBackend();
  const [session, dispatch] = useReducer(sessionReducer, IDLE);
  const [commands, setCommands] = useState<CommandMap | null>(null);
  const [policy, setPolicy] = useState<Policy | null>(null);
  const [stage, setStage] = useState<Stage>(() => initialStage(SLIDES.length));
  const [direction, setDirection] = useState<1 | -1>(1);
  const [flash, setFlash] = useState<{ action: Action; keyword: string | null } | null>(null);
  const [last, setLast] = useState<(Decision & { result: string | null }) | null>(null);
  const [hearing, setHearing] = useState<{ label: string; score: number } | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [cooldownUntil, setCooldownUntil] = useState(0);
  const [locked, setLocked] = useState(false);
  const [analyser, setAnalyser] = useState<AnalyserNode | null>(null);
  const [slowNotice, setSlowNotice] = useState<string | null>(null);

  const stageRef = useRef(stage);
  const micRef = useRef<MicPipeline | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const readyRef = useRef(false);
  const serverErrorRef = useRef<string | null>(null);
  const droppedRef = useRef(0);
  const flashTimer = useRef(0);
  const lockTimer = useRef(0);
  const attemptRef = useRef(0);

  useEffect(() => {
    let cancelled = false;
    api.commands().then((map) => {
      if (cancelled) return;
      setCommands(map);
      setPolicy(map.policy);
    }).catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [backend.status]);

  // Release everything: tracks, audio context, socket. Safe to call repeatedly.
  const teardown = useCallback(() => {
    attemptRef.current += 1;
    readyRef.current = false;
    const socket = socketRef.current;
    socketRef.current = null;
    if (socket) {
      socket.onclose = null;
      socket.onmessage = null;
      socket.onerror = null;
      if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ type: "stop" }));
      if (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING) socket.close(1000);
    }
    micRef.current?.stop();
    micRef.current = null;
    setAnalyser(null);
    setHearing(null);
  }, []);

  const stop = useCallback(
    (reason: string) => {
      teardown();
      dispatch({ type: "stopped", reason });
    },
    [teardown],
  );

  const fail = useCallback(
    (code: string) => {
      teardown();
      dispatch({ type: "failed", code });
    },
    [teardown],
  );

  // The single action path shared by voice, keyboard, and clicks (ADR-0004).
  const perform = useCallback((action: Action, keyword: string | null) => {
    const outcome = applyAction(stageRef.current, action);
    stageRef.current = outcome.stage;
    setStage(outcome.stage);
    if (action === "next_slide" || action === "previous_slide") setDirection(action === "next_slide" ? 1 : -1);
    window.clearTimeout(flashTimer.current);
    setFlash({ action, keyword });
    flashTimer.current = window.setTimeout(() => setFlash(null), 520);
    return outcome;
  }, []);

  const manual = useCallback(
    (action: Action) => {
      const outcome = perform(action, null);
      setHistory((list) =>
        addHistory(list, {
          key: `m${++entryKey}`,
          source: "manual",
          label: `${KEY_LABELS[action]} key`,
          score: null,
          accepted: true,
          reason: "manual",
          result: outcome.result,
          processingMs: null,
          at: performance.now(),
        }),
      );
    },
    [perform],
  );

  const onMessage = useCallback(
    (message: ServerMessage) => {
      switch (message.type) {
        case "ready":
          readyRef.current = true;
          setPolicy((current) => (current ? { ...current, ...message.policy } : (message.policy as Policy)));
          dispatch({ type: "ready", at: performance.now() });
          return;
        case "frame":
          setHearing({ label: message.label, score: message.score });
          return;
        case "decision": {
          let result: string | null = null;
          if (message.accepted && message.action) {
            const outcome = perform(message.action, message.label);
            result = outcome.result;
            socketRef.current?.send(
              JSON.stringify({ type: "action_result", decision_id: message.id, status: outcome.changed ? "applied" : "no_change" }),
            );
            const cooldown = (policy?.cooldown_s ?? 1.2) * 1000;
            setCooldownUntil(performance.now() + cooldown);
            setLocked(true);
            window.clearTimeout(lockTimer.current);
            lockTimer.current = window.setTimeout(() => setLocked(false), cooldown);
          }
          setLast({ ...message, result });
          setHistory((list) =>
            addHistory(list, {
              key: `d${message.id}`,
              source: "voice",
              label: message.label,
              score: message.score,
              accepted: message.accepted,
              reason: message.reason,
              result,
              processingMs: message.processing_ms,
              at: performance.now(),
            }),
          );
          return;
        }
        case "backpressure":
          setSlowNotice(`The API fell behind and dropped ${message.dropped_s.toFixed(2)} s of stale audio. The current utterance restarted, so it cannot double-trigger.`);
          setHistory((list) =>
            addHistory(list, {
              key: `b${++entryKey}`,
              source: "system",
              label: "stale audio",
              score: null,
              accepted: false,
              reason: "backpressure",
              result: `dropped ${message.dropped_s.toFixed(2)} s`,
              processingMs: null,
              at: performance.now(),
            }),
          );
          return;
        case "error":
          serverErrorRef.current = message.code;
          return;
        case "stopped":
          return;
      }
    },
    [perform, policy],
  );

  const start = useCallback(async () => {
    dispatch({ type: "start" });
    const attempt = ++attemptRef.current;
    serverErrorRef.current = null;
    droppedRef.current = 0;
    setSlowNotice(null);
    const permission = await micPermission();
    let mic: MicPipeline;
    try {
      mic = await openMic((frame) => {
        const socket = socketRef.current;
        if (!socket || socket.readyState !== WebSocket.OPEN || !readyRef.current) return;
        if (socket.bufferedAmount > BACKPRESSURE_BYTES) {
          droppedRef.current += frame.byteLength;
          setSlowNotice("The connection is slow, so stale mic frames are being dropped instead of queued.");
          return;
        }
        socket.send(frame);
      });
    } catch (error) {
      if (attempt === attemptRef.current) fail(micErrorCode(error, permission));
      return;
    }
    if (attempt !== attemptRef.current) {
      // Stopped while the permission prompt was open: release at once.
      mic.stop();
      return;
    }
    micRef.current = mic;
    setAnalyser(mic.analyser);
    dispatch({ type: "mic_ready" });
    const socket = new WebSocket(LIVE_URL);
    socket.binaryType = "arraybuffer";
    socketRef.current = socket;
    socket.onopen = () =>
      socket.send(JSON.stringify({ type: "start", version: 1, sample_rate: mic.sampleRate, channels: 1, encoding: "pcm_s16le" }));
    socket.onmessage = (event: MessageEvent) => {
      if (typeof event.data !== "string") return;
      const message = parseServerMessage(event.data);
      if (message) onMessage(message);
    };
    socket.onclose = () => {
      const code = serverErrorRef.current ?? (readyRef.current ? "disconnected" : "offline");
      fail(code);
    };
  }, [fail, onMessage]);

  // Never listen in the background: hiding or leaving the tab stops the session.
  useEffect(() => {
    const onHide = () => {
      if (document.hidden && micRef.current) stop("tab_hidden");
    };
    const onExit = () => {
      if (micRef.current) stop("page_exit");
    };
    document.addEventListener("visibilitychange", onHide);
    window.addEventListener("pagehide", onExit);
    return () => {
      document.removeEventListener("visibilitychange", onHide);
      window.removeEventListener("pagehide", onExit);
    };
  }, [stop]);

  useEffect(() => () => teardown(), [teardown]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.altKey || event.ctrlKey || event.metaKey) return;
      const target = event.target as HTMLElement | null;
      if (target?.closest("input, textarea, select")) return;
      const action = KEY_BINDINGS[event.key.length === 1 ? event.key.toLowerCase() : event.key];
      if (!action) return;
      event.preventDefault();
      manual(action);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [manual]);

  const status = session.status;
  const listening = status === "listening";
  const pending = status === "requesting" || status === "connecting";
  const canStart = backend.status === "ready" && !listening && !pending;
  const commandList = commands?.commands ?? DEFAULT_COMMANDS;
  const lastColor = last ? (last.accepted || last.reason === "cooldown" || last.reason === "low_confidence" ? keywordColor(last.label) : REJECTED_COLOR) : REJECTED_COLOR;

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <p className="eyebrow">Voice Control</p>
          <h1>Drive the demo with six Keywords</h1>
          <p>
            Press Start to open the mic. Accepted Keywords trigger in-app slide and player actions only; Unknown,
            Silence, and uncertain input are rejected and shown. Keyboard keys run the same actions.
          </p>
        </div>
      </div>

      <section className="card" aria-label="Mic session">
        <div className="session-bar">
          <div className="status">
            <span className={`session-state${listening ? " listening" : ""}`} role="status">
              <span className={`dot ${listening ? "live" : pending ? "busy" : status === "error" ? "critical" : ""}`} />
              {listening ? "Listening" : status === "requesting" ? "Requesting mic" : status === "connecting" ? "Connecting" : status === "error" ? "Stopped · error" : "Mic off"}
              {listening && session.startedAt !== null ? <Elapsed since={session.startedAt} /> : null}
            </span>
            <span className="session-sub">
              {listening ? "Streaming to 127.0.0.1 only" : session.stopReason ? STOP_REASONS[session.stopReason] ?? "Stopped." : "Nothing is listening"}
            </span>
          </div>
          {listening || pending ? (
            <button type="button" className="btn solid-danger lg" onClick={() => stop("user")}>
              <StopIcon size={16} /> Stop
            </button>
          ) : (
            <button type="button" className="btn primary lg" onClick={start} disabled={!canStart}>
              <MicIcon size={18} /> Start listening
            </button>
          )}
          <LevelMeter analyser={analyser} />
          <div className="hearing">
            <span className="label">Model hears now</span>
            <span className="now" style={{ "--kw": keywordColor(hearing?.label ?? "") } as CSSProperties}>
              {hearing ? (
                <>
                  <span className="kw-name" style={{ "--kw": keywordColor(hearing.label) } as CSSProperties}>{hearing.label}</span>
                  <span className="mono" style={{ color: "var(--muted)" }}>{formatScore(hearing.score)}</span>
                </>
              ) : (
                <span style={{ color: "var(--muted)" }}>—</span>
              )}
            </span>
          </div>
        </div>
        <div className="privacy">
          <LockIcon size={14} />
          Mic audio is scored in memory in 1 s windows and discarded. No raw mic audio is recorded or stored; only
          decision metadata (label, Confidence, action) is logged. Stop, a lost connection, or hiding this tab releases
          the mic at once.
        </div>
      </section>

      <div style={{ display: "grid", gap: 12, marginTop: 16 }}>
        {backend.status === "offline" ? (
          <ErrorBanner code="offline" handlers={{ retry: refresh }} />
        ) : backend.status === "unavailable" ? (
          <ErrorBanner code={backend.code} />
        ) : null}
        {session.error ? (
          <ErrorBanner
            code={session.error}
            handlers={{ enable_mic: start, mic_settings: start, start_again: start, retry: start }}
          />
        ) : null}
        {slowNotice && listening ? <Banner tone="info" live title="Backpressure">{slowNotice}</Banner> : null}
      </div>

      <div className="live-grid">
        <div className="stage">
          <SlideDeck stage={stage} flash={flash?.action ?? null} direction={direction} onAction={manual} />
          <Player stage={stage} flash={flash?.action ?? null} onAction={manual} />
          <section className="card" aria-labelledby="commands-title">
            <div className="card-head">
              <h3 id="commands-title">Command map</h3>
              <span className="sub">
                Allowlist · {policy ? `threshold ${policy.threshold} · ${policy.min_consecutive} windows in a row · cooldown ${policy.cooldown_s} s` : "loading policy"}
              </span>
            </div>
            <div className="commands">
              {commandList.map((command) => {
                const action = command.action as Action;
                return (
                  <button
                    key={command.keyword}
                    type="button"
                    className={`command${flash?.keyword === command.keyword ? " flash" : ""}${locked ? " locked" : ""}`}
                    style={{ "--kw": keywordColor(command.keyword) } as CSSProperties}
                    onClick={() => manual(action)}
                    aria-label={`${command.label}: say ${command.keyword} or press ${KEY_LABELS[action]}`}
                  >
                    <span className="kw-name">“{command.keyword}”</span>
                    <span className="act">{command.label}</span>
                    <span className="kbd">{KEY_LABELS[action]}</span>
                  </button>
                );
              })}
            </div>
            <div className="card-foot">
              Unknown speech and Silence never map to an action. Nothing here can reach the OS, other apps, or a shell.
            </div>
          </section>
        </div>

        <div className="stage">
          <section className="card decision" aria-labelledby="last-title" aria-live="polite">
            <div className="big">
              <div style={{ display: "grid", gap: 6 }}>
                <span className="eyebrow" id="last-title">Last decision</span>
                {last ? (
                  <span className="word" style={{ "--kw": lastColor } as CSSProperties}>{last.label}</span>
                ) : (
                  <span className="word" style={{ "--kw": REJECTED_COLOR, color: "var(--muted)" } as CSSProperties}>none yet</span>
                )}
              </div>
              {last ? (
                <span className={`verdict ${last.accepted ? "accepted" : "rejected"}`}>
                  {last.accepted ? <CheckIcon size={14} /> : <XIcon size={14} />}
                  {last.accepted ? "ACCEPTED" : "REJECTED"}
                </span>
              ) : null}
            </div>
            <div style={{ display: "grid", gap: 6 }}>
              <span className="table-note">Confidence (classifier score, not correctness)</span>
              <span className="scorebar">
                <span className="track" style={{ width: "100%", maxWidth: 360 }} aria-hidden="true">
                  <span className="fill" style={{ width: `${(last?.score ?? 0) * 100}%`, background: lastColor }} />
                  {policy ? <span className="threshold" style={{ left: `${policy.threshold * 100}%` }} title="Threshold" /> : null}
                </span>
                <span className="value">{last ? formatScore(last.score) : "—"}</span>
              </span>
            </div>
            <p className="reason">{last ? REASON_COPY[last.reason] : listening ? "Say one of the six Keywords." : "Start the mic, or use the keys below."}</p>
            {last?.result ? (
              <p className="result"><CheckIcon size={16} /> {last.result} <span className="tag">{last.processing_ms.toFixed(0)} ms window → decision</span></p>
            ) : null}
            <CooldownBar until={cooldownUntil} duration={(policy?.cooldown_s ?? 1.2) * 1000} />
          </section>

          <section className="card" aria-labelledby="history-title">
            <div className="card-head">
              <h3 id="history-title">Decision history</h3>
              <span className="sub">Newest first · last 20 · accepted and rejected</span>
            </div>
            {history.length === 0 ? (
              <p className="table-note" style={{ padding: 18 }}>No decisions yet. Voice decisions and keyboard actions appear here.</p>
            ) : (
              <ol className="history">
                {history.map((entry) => (
                  <li
                    key={entry.key}
                    className={entry.accepted && entry.source === "voice" ? "accepted" : undefined}
                    style={{ "--kw": entry.source === "voice" && (entry.accepted || entry.reason === "cooldown" || entry.reason === "low_confidence") ? keywordColor(entry.label) : REJECTED_COLOR } as CSSProperties}
                  >
                    <span className="h-label">{entry.label}</span>
                    <span className="h-detail">
                      <span className="src">{entry.source === "voice" ? (entry.accepted ? "accepted" : "rejected") : entry.source === "manual" ? "manual" : "system"}</span>{" "}
                      {entry.result ?? REASON_COPY[entry.reason] ?? entry.reason}
                    </span>
                    <span className="h-meta">
                      {entry.score !== null ? formatScore(entry.score) : ""}
                      {entry.processingMs !== null ? ` · ${entry.processingMs.toFixed(0)}ms` : ""}
                    </span>
                  </li>
                ))}
              </ol>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}
