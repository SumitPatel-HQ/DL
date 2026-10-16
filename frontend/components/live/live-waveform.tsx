"use client";

import { useEffect, useRef } from "react";

interface LiveWaveformProps {
  analyser: AnalyserNode | null;
  listening: boolean;
}

export function LiveWaveform({ analyser, listening }: LiveWaveformProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    let animId = 0;
    let width = container.clientWidth || 400;
    let height = container.clientHeight || 36;

    const dpr = window.devicePixelRatio || 1;
    const resize = () => {
      if (!container || !canvas) return;
      width = container.clientWidth;
      height = container.clientHeight || 36;
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
    };

    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(container);

    const ctx = canvas.getContext("2d");
    if (!ctx) return () => observer.disconnect();

    const bufferLength = analyser ? analyser.fftSize : 512;
    const dataArray = new Float32Array(bufferLength);

    const PITCH = 5.8; // center-to-center bar pitch
    const BAR_WIDTH = 2.4; // width of each vertical tick
    const MAX_BARS = 140;
    const smoothedHeights = new Float32Array(MAX_BARS);

    let idlePhase = 0;

    const render = () => {
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);

      const midY = height / 2;
      const availableWidth = Math.max(80, width - 20);
      const numBars = Math.min(MAX_BARS, Math.max(16, Math.floor(availableWidth / PITCH)));
      const startX = (width - numBars * PITCH) / 2 + PITCH / 2;
      const fadeRange = 36; // px over which edges fade out

      if (listening && analyser) {
        analyser.getFloatTimeDomainData(dataArray);

        const samplesPerBar = Math.floor(bufferLength / numBars);

        for (let b = 0; b < numBars; b++) {
          let peak = 0;
          const startIdx = b * samplesPerBar;
          for (let s = 0; s < samplesPerBar; s++) {
            const val = Math.abs(dataArray[startIdx + s] || 0);
            if (val > peak) peak = val;
          }

          // Gentle taper near ends
          const t = b / (numBars - 1);
          const taper = Math.sin(t * Math.PI);

          // Non-linear amplification to bring out speech while keeping ceiling graceful
          const amplified = Math.min(1, Math.pow(peak, 0.62) * 3.4 * (0.35 + 0.65 * taper));
          const targetH = Math.max(0, amplified * (height * 0.76));

          // Smooth interpolation
          smoothedHeights[b] += (targetH - smoothedHeights[b]) * 0.36;
        }

        // Render each bar with edge-aware opacity and sound level
        for (let b = 0; b < numBars; b++) {
          const x = startX + b * PITCH;
          const h = smoothedHeights[b];
          const edgeDist = Math.min(x, width - x);
          const fade = Math.max(0.12, Math.min(1, edgeDist / fadeRange));

          ctx.beginPath();
          ctx.lineWidth = BAR_WIDTH;
          ctx.lineCap = "round";

          if (h < 1.4) {
            // Calm dot on silence
            ctx.moveTo(x, midY);
            ctx.lineTo(x, midY);
            ctx.strokeStyle = `rgba(165, 180, 195, ${(0.38 * fade).toFixed(3)})`;
          } else {
            // Active speech tick expanding symmetrically
            const isHot = h > height * 0.44;
            ctx.moveTo(x, midY - h / 2);
            ctx.lineTo(x, midY + h / 2);
            ctx.strokeStyle = isHot
              ? `rgba(91, 208, 214, ${(0.96 * fade).toFixed(3)})`
              : `rgba(240, 246, 252, ${(0.92 * fade).toFixed(3)})`;
          }

          ctx.stroke();
        }
      } else {
        // Idle standby mode: discrete baseline dots with subtle breathing ripple
        idlePhase += 0.035;

        for (let b = 0; b < numBars; b++) {
          const x = startX + b * PITCH;
          const edgeDist = Math.min(x, width - x);
          const fade = Math.max(0.12, Math.min(1, edgeDist / fadeRange));

          // Reset smoothed height gracefully
          smoothedHeights[b] = 0;

          // Gentle sine breathing wave across dots (+/- 0.6px)
          const wave = Math.sin((b / numBars) * Math.PI * 3 + idlePhase) * 0.7;

          ctx.beginPath();
          ctx.lineWidth = BAR_WIDTH;
          ctx.lineCap = "round";
          ctx.moveTo(x, midY + wave);
          ctx.lineTo(x, midY + wave);
          ctx.strokeStyle = `rgba(150, 165, 180, ${(0.32 * fade).toFixed(3)})`;
          ctx.stroke();
        }
      }

      animId = requestAnimationFrame(render);
    };

    animId = requestAnimationFrame(render);

    return () => {
      cancelAnimationFrame(animId);
      observer.disconnect();
    };
  }, [analyser, listening]);

  return (
    <div ref={containerRef} className="dock-strip-canvas-wrap" aria-label="Microphone live waveform">
      <canvas ref={canvasRef} className="dock-strip-canvas" />
    </div>
  );
}
