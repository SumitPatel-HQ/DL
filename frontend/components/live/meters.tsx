"use client";

import { useEffect, useRef, useState } from "react";

const BARS = 28;

// RMS level at animation-frame cadence, written to the DOM directly.
export function LevelMeter({ analyser }: { analyser: AnalyserNode | null }) {
  const barsRef = useRef<HTMLDivElement>(null);
  const labelRef = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    const bars = barsRef.current;
    if (!bars) return;
    const cells = Array.from(bars.children) as HTMLElement[];
    if (!analyser) {
      for (const cell of cells) cell.className = "";
      if (labelRef.current) labelRef.current.textContent = "mic off";
      return;
    }
    const data = new Float32Array(analyser.fftSize);
    let frame = 0;
    const loop = () => {
      analyser.getFloatTimeDomainData(data);
      let sum = 0;
      let peak = 0;
      for (const value of data) {
        sum += value * value;
        peak = Math.max(peak, Math.abs(value));
      }
      const db = 20 * Math.log10(Math.sqrt(sum / data.length) || 1e-6);
      const level = Math.min(1, Math.max(0, (db + 60) / 60));
      const lit = Math.round(level * BARS);
      cells.forEach((cell, i) => {
        cell.className = i < lit ? `on${peak > 0.99 ? " clip" : i > BARS * 0.8 ? " hot" : ""}` : "";
      });
      if (labelRef.current) labelRef.current.textContent = `${Math.max(-60, db).toFixed(0)} dBFS${peak > 0.99 ? " · clipping" : ""}`;
      frame = requestAnimationFrame(loop);
    };
    frame = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(frame);
  }, [analyser]);
  return (
    <div className="meter">
      <div className="bars" ref={barsRef} aria-hidden="true">
        {Array.from({ length: BARS }, (_, i) => <i key={i} />)}
      </div>
      <span className="meter-label"><span>Input level (proves the mic is alive, not that a Keyword was heard)</span><span ref={labelRef} className="mono">mic off</span></span>
    </div>
  );
}

export function Elapsed({ since }: { since: number }) {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    const timer = window.setInterval(() => setSeconds(Math.floor((performance.now() - since) / 1000)), 250);
    return () => window.clearInterval(timer);
  }, [since]);
  return <span className="mono">{`${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`}</span>;
}

export function CooldownBar({ until, duration }: { until: number; duration: number }) {
  const fillRef = useRef<HTMLDivElement>(null);
  const labelRef = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    let frame = 0;
    const loop = () => {
      const left = Math.max(0, until - performance.now());
      if (fillRef.current) fillRef.current.style.width = `${(left / duration) * 100}%`;
      if (labelRef.current) labelRef.current.textContent = left > 0 ? `cooldown ${(left / 1000).toFixed(1)} s` : "ready for next utterance";
      if (left > 0) frame = requestAnimationFrame(loop);
    };
    frame = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(frame);
  }, [until, duration]);
  return (
    <div className="cooldown">
      <span ref={labelRef} style={{ minWidth: 170 }}>ready for next utterance</span>
      <div className="ctrack" aria-hidden="true"><div ref={fillRef} style={{ width: 0 }} /></div>
    </div>
  );
}
