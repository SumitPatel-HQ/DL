"use client";

import { useState, type CSSProperties } from "react";
import { formatScore } from "../../lib/format.ts";
import { keywordColor } from "../../lib/keywords.ts";
import { LockIcon, MicIcon, StopIcon, XIcon } from "../icons";
import { Elapsed } from "./meters";
import { LiveWaveform } from "./live-waveform";

interface FloatingDockProps {
  listening: boolean;
  pending: boolean;
  status: string;
  startedAt: number | null;
  canStart: boolean;
  hearing: { label: string; score: number } | null;
  analyser: AnalyserNode | null;
  stopReasonText?: string | null;
  lastDecision?: { label: string; score: number; accepted: boolean; result: string | null } | null;
  onStart: () => void;
  onStop: () => void;
}

export function FloatingDock({
  listening,
  pending,
  status,
  startedAt,
  canStart,
  hearing,
  analyser,
  stopReasonText,
  lastDecision,
  onStart,
  onStop,
}: FloatingDockProps) {
  const [showPrivacy, setShowPrivacy] = useState(false);

  const hasLiveSpeech = hearing && hearing.label && hearing.label !== "_silence_" && hearing.label !== "_unknown_";
  const displayLabel = hasLiveSpeech ? hearing.label : (listening && lastDecision?.accepted ? lastDecision.label : null);
  const displayScore = hasLiveSpeech ? hearing.score : (listening && lastDecision?.accepted ? lastDecision.score : null);

  return (
    <aside
      className={`floating-dock-widget ${listening ? "is-live" : ""}`}
      aria-label="Voice Control dock"
      role="region"
    >
      {/* Attached Top Shelf (unified directly above the main bar) */}
      <div className="dock-attached-shelf" role="status" aria-live="polite">
        <div className="dock-shelf-left">
          {pending ? (
            <span className="dock-shelf-status">
              <span className="dot busy" />
              <span>Connecting microphone...</span>
            </span>
          ) : listening ? (
            displayLabel ? (
              <span className="dock-shelf-status hearing-active">
                <span className="dock-shelf-eyebrow">hear —</span>
                <span
                  className="dock-shelf-kw"
                  style={{ "--kw": keywordColor(displayLabel) } as CSSProperties}
                >
                  {displayLabel}
                </span>
                {typeof displayScore === "number" ? (
                  <span className="dock-shelf-score mono">{formatScore(displayScore)}</span>
                ) : null}
              </span>
            ) : (
              <span className="dock-shelf-status">
                <span className="dot live" />
                <span>Listening · say &ldquo;next&rdquo;, &ldquo;left&rdquo;, &ldquo;go&rdquo;...</span>
              </span>
            )
          ) : (
            <span className="dock-shelf-status is-idle">
              <span className={`dot ${status === "error" ? "critical" : ""}`} />
              <span>
                {status === "error"
                  ? "Microphone error"
                  : stopReasonText
                  ? `${stopReasonText} Mic is released.`
                  : "Ready when you are."}
              </span>
            </span>
          )}
        </div>

        {listening && startedAt !== null ? (
          <div className="dock-shelf-right">
            <Elapsed since={startedAt} />
          </div>
        ) : null}
      </div>

      {/* Main Bar (Attached beneath the shelf, modern 16px radius) */}
      <div className="dock-main-bar">
        {/* Left: Privacy Lock Button */}
        <button
          type="button"
          className={`dock-bar-btn dock-btn-privacy ${showPrivacy ? "is-active" : ""}`}
          onClick={() => setShowPrivacy((v) => !v)}
          title="Memory-only audio processing details"
          aria-label="Privacy information"
          aria-expanded={showPrivacy}
        >
          <LockIcon size={20} />
        </button>

        {/* Center: Real-Time Discrete Sound Waveform */}
        <div className="dock-bar-waveform">
          <LiveWaveform analyser={analyser} listening={listening} />
        </div>

        {/* Right Action Controls: Single Mic/Stop toggle button */}
        <div className="dock-bar-actions">
          {listening || pending ? (
            <button
              type="button"
              className="dock-bar-btn dock-btn-mic is-active"
              onClick={onStop}
              title="Stop listening"
              aria-label="Stop listening"
            >
              <StopIcon size={18} />
            </button>
          ) : (
            <button
              type="button"
              className="dock-bar-btn dock-btn-mic"
              onClick={onStart}
              disabled={!canStart}
              title="Start listening"
              aria-label="Start listening"
            >
              <MicIcon size={22} />
            </button>
          )}
        </div>
      </div>

      {/* Privacy Popover */}
      {showPrivacy && (
        <div className="dock-privacy-popover" role="dialog" aria-label="Privacy details">
          <div className="popover-head">
            <strong>Memory-Only Processing</strong>
            <button
              type="button"
              className="btn small icon ghost"
              onClick={() => setShowPrivacy(false)}
              aria-label="Close"
            >
              <XIcon size={13} />
            </button>
          </div>
          <p>
            Mic audio is scored in memory in 1-second windows and discarded immediately.
            No raw mic audio is recorded or saved. Only decision labels and confidence are logged.
          </p>
          <p className="popover-foot">
            Stopping the mic, closing or hiding this tab immediately releases all audio hardware.
          </p>
        </div>
      )}
    </aside>
  );
}
