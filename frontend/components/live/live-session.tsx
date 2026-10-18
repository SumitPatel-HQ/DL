"use client";

import { useCallback, useEffect, useReducer, useRef, useState, type CSSProperties } from "react";
import { api, LIVE_URL, type ActionMap } from "../../lib/api.ts";
import { DEPLOYED } from "../../lib/deploy.ts";
import { BrowserLiveSocket } from "../../lib/live/browser-socket.ts";
import { formatScore } from "../../lib/format.ts";
import { keywordColor, REJECTED_COLOR } from "../../lib/keywords.ts";
import { micPermission, openMic, type MicPipeline } from "../../lib/live/mic.ts";
import {
  addHistory, IDLE, micErrorCode, parseServerMessage, REASON_COPY, sessionReducer,
  type HistoryEntry, type ServerMessage, type ExternalState, type ExternalTarget, type NegativeReport,
} from "../../lib/live/session.ts";
import { applyAction, initialStage, KEY_BINDINGS, KEY_LABELS, type Action, type Stage } from "../../lib/live/stage.ts";
import { Settings } from "lucide-react";
import { CheckIcon, XIcon } from "../icons";
import { useBackend } from "../shell/backend";
import { Banner, ErrorBanner } from "../ui";
import { CooldownBar } from "./meters";
import { Player, SLIDES, SlideDeck } from "./stage";
import { ExternalControls } from "./external-controls";
import { FloatingDock } from "./floating-dock";

type Policy = ActionMap["policy"];
type Decision = Extract<ServerMessage, { type: "decision" }>;

const DEFAULT_ACTIONS: ActionMap["actions"] = [
  { keyword: "right", action: "next_slide", label: "Next slide" },
  { keyword: "left", action: "previous_slide", label: "Previous slide" },
  { keyword: "go", action: "play", label: "Play" },
  { keyword: "stop", action: "pause", label: "Pause" },
  { keyword: "up", action: "volume_up", label: "Volume up" },
  { keyword: "down", action: "volume_down", label: "Volume down" },
];

const DEFAULT_EXTERNAL_TARGETS: ExternalTarget[] = [
  {
    id: "volume_default",
    kind: "volume",
    label: "Windows volume: Speakers / Audio Output",
    actions: ["volume_up", "volume_down"],
  },
  {
    id: "powerpoint_default",
    kind: "powerpoint",
    label: "PowerPoint: Active slideshow presentation",
    actions: ["next_slide", "previous_slide"],
  },
  {
    id: "media_default",
    kind: "media",
    label: "Media: Windows media player session",
    actions: ["play", "pause"],
  },
];

const DEFAULT_EXTERNAL: ExternalState = {
  type: "external_state",
  supported: !DEPLOYED,
  enabled: false,
  target_id: null,
  allowed_actions: [],
  targets: DEFAULT_EXTERNAL_TARGETS,
  experimental: true,
  error: null,
  negative_running: false,
};

const BACKPRESSURE_BYTES = 256 * 1024;

// Deployed there is no socket server, so the page scores the mic itself.
// `?engine=browser` tries the in-browser engine against a local API.
function openLiveSocket(): WebSocket | BrowserLiveSocket {
  if (DEPLOYED || new URLSearchParams(window.location.search).get("engine") === "browser") {
    return new BrowserLiveSocket();
  }
  return new WebSocket(LIVE_URL);
}
const LAST_DECISION_HOLD_MS = 2000;
const STOP_REASONS: Record<string, string> = {
  user: "Stopped. The mic is released.",
  tab_hidden: "Stopped because the tab was hidden. The mic is released.",
  page_exit: "Stopped on page exit.",
};

let entryKey = 0;

export function LiveSession() {
  const { backend, refresh } = useBackend();
  const [session, dispatch] = useReducer(sessionReducer, IDLE);
  const [actionMap, setActionMap] = useState<ActionMap | null>(null);
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
  const [external, setExternal] = useState<ExternalState>(DEFAULT_EXTERNAL);
  const [externalBusy, setExternalBusy] = useState(false);
  const [negativeReport, setNegativeReport] = useState<NegativeReport | null>(null);
  const [showExternal, setShowExternal] = useState(false);
  const pendingGrantRef = useRef<{
    target_id: string;
    kind?: string;
    actions: Action[];
    acknowledge_experimental: boolean;
  } | null>(null);
  const replayRef = useRef<"idle" | "discovering" | "done">("idle");

  const stageRef = useRef(stage);
  const micRef = useRef<MicPipeline | null>(null);
  const socketRef = useRef<WebSocket | BrowserLiveSocket | null>(null);
  const readyRef = useRef(false);
  const serverErrorRef = useRef<string | null>(null);
  const droppedRef = useRef(0);
  const flashTimer = useRef(0);
  const lockTimer = useRef(0);
  const lastAcceptedAt = useRef(-Infinity);
  const attemptRef = useRef(0);

  useEffect(() => {
    let cancelled = false;
    api.actions().then((map) => {
      if (cancelled) return;
      setActionMap(map);
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
    setExternal((prev) => ({
      ...prev,
      enabled: false,
      target_id: null,
      allowed_actions: [],
      error: null,
      negative_running: false,
    }));
    setExternalBusy(false);
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

  // In-app actions only. External voice results are executed by the backend.
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
        case "external_state": {
          const serverState = message;
          const mergedTargets = serverState.targets.length > 0 ? serverState.targets : DEFAULT_EXTERNAL_TARGETS;

          // Replay a grant saved while idle exactly once per socket. Every grant
          // bumps the server generation and discards buffered audio, so
          // answering each external_state with another grant starves the model.
          const socket = socketRef.current;
          const pending = pendingGrantRef.current;
          if (pending && socket?.readyState === WebSocket.OPEN) {
            if (replayRef.current === "idle") {
              // The server knows no targets until discovery runs.
              replayRef.current = "discovering";
              socket.send(JSON.stringify({ type: "external_targets" }));
            } else if (replayRef.current === "discovering" && serverState.targets.length > 0) {
              replayRef.current = "done";
              const matched = serverState.targets.find((t) => t.id === pending.target_id)
                ?? serverState.targets.find((t) => t.kind === pending.kind);
              const actions = matched ? pending.actions.filter((a) => matched.actions.includes(a)) : [];
              if (matched && actions.length > 0) {
                socket.send(JSON.stringify({
                  type: "external_grant",
                  target_id: matched.id,
                  actions,
                  acknowledge_experimental: true,
                }));
              }
            } else if (replayRef.current === "discovering") {
              replayRef.current = "done";
            }
          }

          setExternal({ ...serverState, targets: mergedTargets });
          setExternalBusy(false);
          return;
        }
        case "negative_report":
          setNegativeReport(message.report);
          setExternal((current) => current ? { ...current, negative_running: false } : current);
          setExternalBusy(false);
          return;
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
            if (message.external) {
              result = `External ${message.action.replaceAll("_", " ")}: ${message.external.status}. ${message.external.reason}`;
            } else {
              const outcome = perform(message.action, message.label);
              result = outcome.result;
              socketRef.current?.send(
                JSON.stringify({ type: "action_result", decision_id: message.id, status: outcome.changed ? "applied" : "no_change" }),
              );
            }
            const cooldown = (policy?.cooldown_s ?? 1.2) * 1000;
            setCooldownUntil(performance.now() + cooldown);
            setLocked(true);
            window.clearTimeout(lockTimer.current);
            lockTimer.current = window.setTimeout(() => setLocked(false), cooldown);
            lastAcceptedAt.current = performance.now();
          }
          // Every pause after a Keyword starts a Silence run, and the word's tail
          // often scores Unknown. Neither may hide the decision the user just made.
          const fresh = performance.now() - lastAcceptedAt.current < LAST_DECISION_HOLD_MS;
          if (message.label !== "silence" && (message.accepted || message.label !== "unknown" || !fresh)) {
            setLast({ ...message, result });
          }
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

  const sendExternal = useCallback((control: Record<string, unknown>) => {
    const socket = socketRef.current;
    if (socket?.readyState === WebSocket.OPEN && readyRef.current) {
      setExternalBusy(true);
      socket.send(JSON.stringify(control));
      if (control.type === "external_revoke") {
        pendingGrantRef.current = null;
      }
    } else {
      // Idle mode: allow editing and saving settings freely before mic turns on
      if (control.type === "external_grant") {
        const targetId = control.target_id as string;
        const target = external.targets.find((t) => t.id === targetId);
        pendingGrantRef.current = {
          target_id: targetId,
          kind: target?.kind,
          actions: control.actions as Action[],
          acknowledge_experimental: true,
        };
        setExternal((prev) => ({
          ...prev,
          enabled: true,
          target_id: targetId,
          allowed_actions: control.actions as Action[],
          error: null,
        }));
      } else if (control.type === "external_revoke") {
        pendingGrantRef.current = null;
        setExternal((prev) => ({
          ...prev,
          enabled: false,
          target_id: null,
          allowed_actions: [],
          error: null,
        }));
      } else if (control.type === "external_targets") {
        setExternal((prev) => ({
          ...prev,
          targets: DEFAULT_EXTERNAL_TARGETS,
          error: null,
        }));
      }
    }
  }, [external.targets]);

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
      }, new URLSearchParams(window.location.search).get("dsp") !== "off");
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
    const socket = openLiveSocket();
    socket.binaryType = "arraybuffer";
    socketRef.current = socket;
    replayRef.current = "idle";
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

  // Lock background scrolling and support Escape key when the settings modal is open
  useEffect(() => {
    if (!showExternal) return;

    const originalBodyOverflow = document.body.style.overflow;
    const originalHtmlOverflow = document.documentElement.style.overflow;

    document.body.style.overflow = "hidden";
    document.documentElement.style.overflow = "hidden";

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setShowExternal(false);
      }
    };
    window.addEventListener("keydown", onKeyDown);

    return () => {
      document.body.style.overflow = originalBodyOverflow;
      document.documentElement.style.overflow = originalHtmlOverflow;
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [showExternal]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (showExternal) return;
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
  }, [manual, showExternal]);

  const status = session.status;
  const listening = status === "listening";
  const pending = status === "requesting" || status === "connecting";
  const canStart = backend.status === "ready" && !listening && !pending;
  const actionList = actionMap?.actions ?? actionMap?.commands ?? DEFAULT_ACTIONS;
  // Rejected decisions (accepted:false) always use rejected styling, even for
  // cooldown/low_confidence. The text badge still names the reason.
  const lastColor = last ? (last.accepted ? keywordColor(last.label) : REJECTED_COLOR) : REJECTED_COLOR;

  return (
    <div className="page page-live">
      <div className="page-head page-head-live">
        <div className="live-header-copy">
          <p className="eyebrow">Voice Control</p>
          <h1>Drive the demo with six Keywords</h1>
          <p>
            Say a trained keyword or press a shortcut key to navigate slides and control audio playback.
          </p>
        </div>
        <div className="live-header-actions">
          {DEPLOYED ? null : <button
            type="button"
            className={`live-settings-btn ${showExternal ? "is-active" : ""}`}
            onClick={() => setShowExternal((v) => !v)}
            title="Windows Actions · OS Control Settings"
            aria-label="Windows OS control settings"
          >
            <Settings size={22} className="live-settings-icon" />
            {external?.enabled ? <span className="dot ok" style={{ width: 6, height: 6 }} /> : null}
          </button>}
        </div>
      </div>

      {showExternal && !DEPLOYED ? (
        <div className="external-modal-overlay" onClick={() => setShowExternal(false)}>
          <div
            className="external-modal-content"
            onClick={(e) => e.stopPropagation()}
            role="dialog"
            aria-modal="true"
            aria-labelledby="external-modal-heading"
          >
            <div className="external-modal-head">
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <Settings size={20} />
                <h3 id="external-modal-heading" style={{ margin: 0, fontSize: "1.08rem" }}>Windows Actions · Experimental</h3>
              </div>
              <button
                type="button"
                className="btn small icon ghost"
                onClick={() => setShowExternal(false)}
                aria-label="Close"
              >
                <XIcon size={16} />
              </button>
            </div>
            <ExternalControls
              state={external}
              report={negativeReport}
              pending={externalBusy}
              send={sendExternal}
            />
          </div>
        </div>
      ) : null}

      {backend.status === "offline" || backend.status === "unavailable" || session.error || (slowNotice && listening) ? (
        <div style={{ display: "grid", gap: 12, marginTop: 4, marginBottom: 16 }}>
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
      ) : null}

      <section className="card" aria-label="Action map">
        <div className="commands">
            {actionList.map((mapping) => {
              const action = mapping.action as Action;
              return (
                <button
                  key={mapping.keyword}
                  type="button"
                  className={`command${flash?.keyword === mapping.keyword ? " flash" : ""}${locked ? " locked" : ""}`}
                  style={{ "--kw": keywordColor(mapping.keyword) } as CSSProperties}
                  onClick={() => manual(action)}
                  aria-label={`${mapping.label}: say ${mapping.keyword} or press ${KEY_LABELS[action]}`}
                >
                  <span className="kw-name">“{mapping.keyword}”</span>
                  <span className="act">{mapping.label}</span>
                  <span className="kbd">{KEY_LABELS[action]}</span>
                </button>
              );
            })}
          </div>
      </section>

      <div className="live-grid">
        <div className="stage">
          <SlideDeck stage={stage} flash={flash?.action ?? null} direction={direction} onAction={manual} />
          <Player stage={stage} flash={flash?.action ?? null} onAction={manual} />
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
            <p className="reason">{last ? (REASON_COPY[last.reason] ?? `Rejected: ${last.reason}`) : listening ? "Say one of the six Keywords." : "Start the mic, or use the keys below."}</p>
            {last?.result ? (
              <p className="result">{last.external && !["applied", "no_change"].includes(last.external.status) ? <XIcon size={16} /> : <CheckIcon size={16} />} {last.result} <span className="tag">{last.processing_ms.toFixed(0)} ms window → decision</span></p>
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
                    style={{ "--kw": entry.source === "voice" && entry.accepted ? keywordColor(entry.label) : REJECTED_COLOR } as CSSProperties}
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

      <FloatingDock
        listening={listening}
        pending={pending}
        status={status}
        startedAt={session.startedAt}
        canStart={canStart}
        hearing={hearing}
        analyser={analyser}
        stopReasonText={session.stopReason ? STOP_REASONS[session.stopReason] ?? "Stopped." : null}
        lastDecision={last}
        onStart={start}
        onStop={() => stop("user")}
      />
    </div>
  );
}
