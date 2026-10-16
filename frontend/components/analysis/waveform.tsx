"use client";

import { useEffect, useLayoutEffect, useMemo, useRef, useState, type RefObject } from "react";
import { api, ApiError, type Detection } from "../../lib/api.ts";
import { assignLanes } from "../../lib/detections.ts";
import { formatClock } from "../../lib/format.ts";
import { keywordColor } from "../../lib/keywords.ts";
import { computePeaks, decodeWav, envelope, type Peaks } from "../../lib/wav.ts";
import { ZoomIn, ZoomOut } from "../icons";
import { kwStyle } from "../ui";

const ZOOMS = [1, 2, 4, 8, 16, 32];
const LANE_H = 26;
const WAVE_H = 132;
const RULER_H = 24;
const TICK_STEPS = [0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120];

type Audio = { peaks: Peaks; sampleRate: number; length: number };

function tickLabel(seconds: number, step: number): string {
  const minutes = Math.floor(seconds / 60);
  const rest = seconds - minutes * 60;
  return `${minutes}:${rest.toFixed(step < 1 ? 2 : 0).padStart(step < 1 ? 5 : 2, "0")}`;
}

export function Waveform({
  recordingId,
  duration,
  detections,
  visible,
  activeId,
  hoverId,
  audioRef,
  onRegion,
  onHover,
  onSeek,
}: {
  recordingId: string;
  duration: number;
  detections: readonly Detection[];
  visible: ReadonlySet<string>;
  activeId: number | null;
  hoverId: number | null;
  audioRef: RefObject<HTMLAudioElement | null>;
  onRegion: (detection: Detection) => void;
  onHover: (id: number | null) => void;
  onSeek: (seconds: number) => void;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const playheadRef = useRef<HTMLDivElement>(null);
  const [audio, setAudio] = useState<Audio | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [width, setWidth] = useState(0);
  const [zoom, setZoom] = useState(1);
  const [scrollLeft, setScrollLeft] = useState(0);

  const { lanes, count } = useMemo(() => assignLanes(detections), [detections]);
  const lanesH = detections.length ? count * LANE_H + 12 : 12;
  const height = lanesH + WAVE_H + RULER_H;
  const innerWidth = Math.max(1, width * zoom);

  useEffect(() => {
    const controller = new AbortController();
    fetch(api.audioUrl(recordingId), { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new ApiError(response.status === 410 ? "audio_unavailable" : "recording_not_found", response.status, "");
        const { samples, sampleRate } = decodeWav(await response.arrayBuffer());
        setAudio({ peaks: computePeaks(samples, 32), sampleRate, length: samples.length });
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        setLoadError(error instanceof ApiError ? error.code : error instanceof TypeError ? "offline" : "invalid_wav");
      });
    return () => controller.abort();
  }, [recordingId]);

  useLayoutEffect(() => {
    const element = scrollRef.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.floor(entry.contentRect.width)));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  // Viewport-only drawing keeps long Recordings cheap at any zoom.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || width === 0) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.clearRect(0, 0, width, height);
    const top = lanesH;
    const mid = top + WAVE_H / 2;
    context.fillStyle = "#252e38";
    context.fillRect(0, mid, width, 1);
    if (audio) {
      const shown = detections
        .filter((d) => visible.has(d.keyword))
        .sort((a, b) => a.start_s - b.start_s);
      const secondsPerPx = duration / innerWidth;
      const amp = WAVE_H / 2 - 6;
      for (let x = 0; x < width; x++) {
        const t0 = (scrollLeft + x) * secondsPerPx;
        if (t0 > duration) break;
        const [low, high] = envelope(
          audio.peaks,
          Math.floor(t0 * audio.sampleRate),
          Math.ceil((t0 + secondsPerPx) * audio.sampleRate),
        );
        const covering = shown.find((d) => d.start_s <= t0 && t0 < d.end_s);
        context.fillStyle = covering ? keywordColor(covering.keyword) : "#4f677b";
        const y1 = mid - Math.max(high, 0.004) * amp;
        const y2 = mid - Math.min(low, -0.004) * amp;
        context.fillRect(x, y1, 1, Math.max(1, y2 - y1));
      }
    }
    // Ruler
    const pxPerSecond = innerWidth / Math.max(duration, 1e-6);
    const step = TICK_STEPS.find((s) => s * pxPerSecond >= 72) ?? 300;
    const rulerTop = top + WAVE_H;
    context.fillStyle = "#405567";
    context.fillRect(0, rulerTop, width, 1);
    context.font = "11px ui-monospace, Consolas, monospace";
    context.textBaseline = "top";
    const first = Math.floor(scrollLeft / pxPerSecond / step) * step;
    for (let t = first; t <= duration + 1e-9; t += step) {
      const x = t * pxPerSecond - scrollLeft;
      if (x > width) break;
      if (x < -60) continue;
      context.fillStyle = "#405567";
      context.fillRect(Math.round(x), rulerTop, 1, 6);
      context.fillStyle = "#969ca6";
      context.fillText(tickLabel(t, step), Math.round(x) + 4, rulerTop + 7);
    }
  }, [audio, width, height, lanesH, zoom, scrollLeft, innerWidth, duration, detections, visible]);

  // Playhead follows the audio clock without re-rendering React.
  useEffect(() => {
    let frame = 0;
    const loop = () => {
      const element = audioRef.current;
      const head = playheadRef.current;
      const scroller = scrollRef.current;
      if (element && head && scroller) {
        const x = (element.currentTime / Math.max(duration, 1e-6)) * innerWidth;
        head.style.transform = `translateX(${x}px)`;
        if (!element.paused && zoom > 1) {
          const view = scroller.clientWidth;
          if (x < scroller.scrollLeft || x > scroller.scrollLeft + view - 40) {
            scroller.scrollLeft = Math.max(0, x - view * 0.2);
          }
        }
      }
      frame = requestAnimationFrame(loop);
    };
    frame = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(frame);
  }, [audioRef, duration, innerWidth, zoom]);

  const changeZoom = (next: number) => {
    const scroller = scrollRef.current;
    const element = audioRef.current;
    setZoom(next);
    if (scroller && element) {
      // Keep the playhead in view after zooming.
      const x = (element.currentTime / Math.max(duration, 1e-6)) * width * next;
      requestAnimationFrame(() => {
        scroller.scrollLeft = Math.max(0, x - scroller.clientWidth / 2);
      });
    }
  };

  const seekFromPointer = (clientX: number, target: HTMLElement) => {
    const rect = target.getBoundingClientRect();
    onSeek(((clientX - rect.left) / rect.width) * duration);
  };

  return (
    <>
      <div className="wave-scroll" ref={scrollRef} onScroll={(e) => setScrollLeft(e.currentTarget.scrollLeft)}>
        <div className="wave-inner" style={{ width: innerWidth, height }}>
          <canvas ref={canvasRef} className="wave-canvas" aria-hidden="true" />
          <div
            className="wave-hit"
            style={{ top: lanesH }}
            onPointerDown={(e) => seekFromPointer(e.clientX, e.currentTarget)}
            aria-hidden="true"
          />
          {detections.map((d) => {
            const left = (d.start_s / duration) * innerWidth;
            const regionWidth = Math.max(4, ((d.end_s - d.start_s) / duration) * innerWidth);
            const dim = !visible.has(d.keyword);
            const active = d.id === activeId;
            return (
              <div key={`wash-${d.id}`}>
                {!dim ? (
                  <div
                    className={`wash${active ? " active" : ""}`}
                    style={{ ...kwStyle(d.keyword), left, width: regionWidth, top: lanesH, bottom: RULER_H }}
                  />
                ) : null}
                <button
                  type="button"
                  className={`region${active ? " active" : ""}${d.id === hoverId ? " hover" : ""}${dim ? " dim" : ""}`}
                  style={{ ...kwStyle(d.keyword), left, width: regionWidth, top: 8 + (lanes.get(d.id) ?? 0) * LANE_H }}
                  onClick={() => onRegion(d)}
                  onMouseEnter={() => onHover(d.id)}
                  onMouseLeave={() => onHover(null)}
                  onFocus={() => onHover(d.id)}
                  onBlur={() => onHover(null)}
                  tabIndex={dim ? -1 : 0}
                  aria-label={`${d.keyword} Detection ${formatClock(d.start_s)} to ${formatClock(d.end_s)}, Confidence ${d.confidence.toFixed(3)}. Play segment`}
                  aria-current={active ? "true" : undefined}
                >
                  {regionWidth > 34 ? d.keyword : ""}
                  {regionWidth > 78 ? <span className="score">{d.confidence.toFixed(2)}</span> : null}
                </button>
              </div>
            );
          })}
          <div ref={playheadRef} className="playhead" style={{ top: lanesH - 4, bottom: RULER_H }} />
        </div>
      </div>
      {loadError ? (
        <div className="wave-legend" role="status">
          Waveform unavailable (<code>{loadError}</code>). Table seeking still works if audio plays.
        </div>
      ) : !audio ? (
        <div className="wave-legend" role="status">Decoding audio…</div>
      ) : null}
      <div className="wave-legend">
        <span>Zoom</span>
        <button
          type="button"
          className="btn small icon ghost"
          onClick={() => changeZoom(ZOOMS[Math.max(0, ZOOMS.indexOf(zoom) - 1)])}
          disabled={zoom === ZOOMS[0]}
          aria-label="Zoom out"
        >
          <ZoomOut />
        </button>
        <span className="mono">{zoom}×</span>
        <button
          type="button"
          className="btn small icon ghost"
          onClick={() => changeZoom(ZOOMS[Math.min(ZOOMS.length - 1, ZOOMS.indexOf(zoom) + 1)])}
          disabled={zoom === ZOOMS[ZOOMS.length - 1]}
          aria-label="Zoom in"
        >
          <ZoomIn />
        </button>
        <span className="toolbar-sep" />
        <span>Regions show Keyword and Confidence; the waveform is tinted under each visible Detection.</span>
      </div>
    </>
  );
}
