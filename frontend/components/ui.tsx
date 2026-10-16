"use client";

import Link from "next/link";
import type { CSSProperties, ReactNode } from "react";
import { useEffect, useRef, useState } from "react";
import { describeError, type Recovery } from "../lib/errors.ts";
import { formatScore } from "../lib/format.ts";
import { keywordColor } from "../lib/keywords.ts";
import { AlertIcon, InfoIcon, OfflineIcon, RefreshIcon } from "./icons";

export function kwStyle(keyword: string): CSSProperties {
  return { "--kw": keywordColor(keyword), "--chip-color": keywordColor(keyword) } as CSSProperties;
}

export function KeywordName({ keyword }: { keyword: string }) {
  return (
    <span className="kw-name" style={kwStyle(keyword)}>
      {keyword}
    </span>
  );
}

export function Banner({
  tone,
  icon,
  title,
  children,
  actions,
  live,
}: {
  tone: "experimental" | "error" | "info";
  icon?: ReactNode;
  title: ReactNode;
  children?: ReactNode;
  actions?: ReactNode;
  live?: boolean;
}) {
  return (
    <div className={`banner ${tone}`} role={live ? (tone === "error" ? "alert" : "status") : undefined}>
      <span className="banner-icon">{icon ?? (tone === "info" ? <InfoIcon /> : <AlertIcon />)}</span>
      <div className="banner-body">
        <strong>{title}</strong>
        {children ? <div>{children}</div> : null}
      </div>
      {actions ? <div className="banner-actions">{actions}</div> : null}
    </div>
  );
}

export type RecoveryHandlers = Partial<Record<Recovery, () => void>>;

const RECOVERY_LABEL: Partial<Record<Recovery, string>> = {
  retry: "Retry",
  choose_file: "Choose another file",
  wait_retry: "Retry",
  reanalyze: "Run analysis again",
  refresh: "Refresh",
  reload: "Reload page",
  enable_mic: "Try again",
  mic_settings: "I changed the setting — Start",
  start_again: "Start again",
  import_runs: "Import committed runs",
};

// One line of cause plus one recovery action for a stable error code.
export function ErrorBanner({ code, handlers = {} }: { code: string; handlers?: RecoveryHandlers }) {
  const copy = describeError(code);
  const handler = copy.recovery === "reload" ? () => window.location.reload() : handlers[copy.recovery];
  return (
    <Banner
      tone="error"
      live
      icon={code === "offline" ? <OfflineIcon /> : <AlertIcon />}
      title={copy.title}
      actions={
        handler ? (
          <button type="button" className="btn small" onClick={handler}>
            {copy.recovery === "retry" || copy.recovery === "wait_retry" ? <RefreshIcon size={14} /> : null}
            {RECOVERY_LABEL[copy.recovery] ?? "Retry"}
          </button>
        ) : null
      }
    >
      {copy.detail} <code className="table-note">{code}</code>
    </Banner>
  );
}

export function ExperimentalBadge({ evidence = "/dashboard?tab=continuous" }: { evidence?: string }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const handleDown = (e: MouseEvent | TouchEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("mousedown", handleDown);
    window.addEventListener("keydown", handleKey);
    return () => {
      window.removeEventListener("mousedown", handleDown);
      window.removeEventListener("keydown", handleKey);
    };
  }, [open]);

  return (
    <div className="experimental-badge-wrap" ref={ref}>
      <button
        type="button"
        className="experimental-badge-btn"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        title="Experimental sliding-window detector. Click for details."
      >
        <span className="experimental-badge-dot" aria-hidden="true" />
        <span>Experimental</span>
        <InfoIcon size={12} />
      </button>

      {open && (
        <div className="experimental-popover" role="dialog" aria-label="Detector limitations">
          <div className="experimental-popover-head">
            <strong>Sliding-Window Detector</strong>
            <button
              type="button"
              className="experimental-popover-close"
              onClick={() => setOpen(false)}
              aria-label="Close"
            >
              ×
            </button>
          </div>
          <p className="experimental-popover-body">
            Approximate 1s spans (event-F1 0.010). Verify detections by listening.
          </p>
          <div className="experimental-popover-actions">
            <Link className="btn small" href={evidence} onClick={() => setOpen(false)}>
              Feasibility run →
            </Link>
          </div>
        </div>
      )}
    </div>
  );
}

export function ExperimentalBanner({ evidence = "/dashboard?tab=continuous" }: { evidence?: string }) {
  return (
    <Banner
      tone="experimental"
      title="Experimental sliding-window detector"
      actions={
        <Link className="btn small" href={evidence}>
          Feasibility run →
        </Link>
      }
    >
      Approximate 1s spans (event-F1 0.010). Verify detections by listening.
    </Banner>
  );
}

export function ScoreBar({ value, threshold, color }: { value: number; threshold?: number; color: string }) {
  return (
    <span className="scorebar" title={threshold !== undefined ? `Threshold ${formatScore(threshold)}` : undefined}>
      <span className="track" aria-hidden="true">
        <span className="fill" style={{ width: `${Math.min(1, value) * 100}%`, background: color }} />
        {threshold !== undefined ? <span className="threshold" style={{ left: `${threshold * 100}%` }} /> : null}
      </span>
      <span className="value">{formatScore(value)}</span>
    </span>
  );
}

export function EmptyState({
  icon,
  title,
  children,
  actions,
}: {
  icon: ReactNode;
  title: string;
  children?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <div className="empty">
      <div className="empty-icon">{icon}</div>
      <h3>{title}</h3>
      {children ? <p>{children}</p> : null}
      {actions ? <div style={{ display: "flex", gap: 8, marginTop: 6 }}>{actions}</div> : null}
    </div>
  );
}
