import Link from "next/link";
import { Suspense } from "react";
import { ArrowRight, ChartIcon, MicIcon, WaveIcon } from "../components/icons";
import { ModelStatus } from "../components/model-status";
import { KEYWORD_ORDER } from "../lib/keywords";

export const dynamic = "force-dynamic";

// 340-slice acoustic waveform accurately matching DAW / audio editor wave tracks
function pseudoRandom(s: number) {
  const x = Math.sin(s * 12.9898 + 78.233) * 43758.5453;
  return x - Math.floor(x);
}

const TOTAL_WAVE_BARS = 340;
const DENSE_WAVE_BARS = Array.from({ length: TOTAL_WAVE_BARS }, (_, i) => {
  const t = i / (TOTAL_WAVE_BARS - 1);

  // Multi-frequency speech energy envelope modeled after DAW reference recording
  const p1 = Math.sin(t * Math.PI * 15.5 + 0.3) * 0.22;
  const p2 = Math.cos(t * Math.PI * 5.8 - 0.2) * 0.18;
  const p3 = Math.sin(t * Math.PI * 27.2 + 1.1) * 0.12;
  const p4 = Math.cos(t * Math.PI * 43.1 + 2.5) * 0.08;

  let macro = 0.44 + p1 + p2 + p3 + p4;

  // Natural speech phrases and pauses
  if (t > 0.65 && t < 0.72) macro *= 0.62;
  if (t < 0.02) macro *= (t / 0.02) * 0.6 + 0.4;
  if (t > 0.98) macro *= ((1 - t) / 0.02) * 0.5 + 0.5;

  macro = Math.max(0.18, Math.min(0.86, macro));

  // High-frequency jitter simulating pitch harmonics and consonant plosives
  const r1 = pseudoRandom(i * 1.731);
  const r2 = pseudoRandom(i * 3.419 + 17.1);
  const r3 = pseudoRandom(i * 7.193 + 53.2);

  const isSpike = r1 > 0.82;
  const spikeBoost = isSpike ? 0.24 + 0.36 * r2 : 0;
  const jitter = 0.60 + 0.40 * r3;

  const amp = Math.min(0.96, macro * jitter + spikeBoost);

  // Acoustic pressure asymmetry (realistic physical speech wave)
  const top = Math.round(Math.max(0.1, Math.min(0.98, amp * (0.94 + 0.12 * r1))) * 30 * 10) / 10;
  const bot = Math.round(Math.max(0.1, Math.min(0.98, amp * (0.90 + 0.14 * r2))) * 29 * 10) / 10;

  return { top, bot, isSpike };
});

const WAVE_STEP = (540 - 24) / (TOTAL_WAVE_BARS - 1);
const WAVE_POLYGON_POINTS = (() => {
  const topPts: string[] = [];
  const botPts: string[] = [];
  DENSE_WAVE_BARS.forEach((b, i) => {
    const x = Math.round((12 + i * WAVE_STEP) * 10) / 10;
    const yTop = Math.round((42 - b.top) * 10) / 10;
    const yBot = Math.round((42 + b.bot) * 10) / 10;
    topPts.push(`${x},${yTop}`);
    botPts.unshift(`${x},${yBot}`);
  });
  return `${topPts.join(" ")} ${botPts.join(" ")}`;
})();

function WaveformPreview() {
  return (
    <svg
      viewBox="0 0 540 84"
      width="100%"
      height="84"
      preserveAspectRatio="none"
      aria-hidden="true"
      style={{ display: "block" }}
    >
      {/* Center zero-axis baseline */}
      <line x1="0" y1="42" x2="540" y2="42" stroke="#192837" strokeWidth="1" />

      {/* Waveform body fill */}
      <polygon points={WAVE_POLYGON_POINTS} fill="rgba(88, 134, 163, 0.16)" />

      {/* Waveform dense acoustic slices */}
      {DENSE_WAVE_BARS.map((b, i) => {
        const x = 12 + i * WAVE_STEP;
        return (
          <rect
            key={i}
            x={x}
            y={42 - b.top}
            width="1.2"
            height={b.top + b.bot}
            rx="0.4"
            fill={b.isSpike ? "#7eb3d4" : "#5a88a6"}
          />
        );
      })}
    </svg>
  );
}

const TOTAL_MIC_BARS = 68;
const MIC_SPECTRUM_BARS = Array.from({ length: TOTAL_MIC_BARS }, (_, i) => {
  const t = i / (TOTAL_MIC_BARS - 1);
  // Human vocal frequency profile (fundamental speech frequencies and formants)
  const f1 = Math.exp(-Math.pow((t - 0.28) / 0.14, 2)) * 0.95;
  const f2 = Math.exp(-Math.pow((t - 0.52) / 0.16, 2)) * 0.88;
  const f3 = Math.exp(-Math.pow((t - 0.74) / 0.15, 2)) * 0.65;
  const base = Math.max(0.14, f1 + f2 + f3);

  // Frequency bin variation
  const jitter = 0.75 + 0.25 * Math.sin(i * 1.9 + 0.4);
  const h = Math.round(Math.min(28, Math.max(3, base * jitter * 27)) * 10) / 10;
  const delay = Math.round((Math.sin(i * 0.28) * 0.4 + 0.5) * 100) / 100;
  const isVocalCore = base > 0.45;

  return { h, delay, isVocalCore };
});

const MIC_BAR_STEP = (540 - 24) / (TOTAL_MIC_BARS - 1);
const MIC_POLYGON_POINTS = (() => {
  const topPts: string[] = [];
  const botPts: string[] = [];
  MIC_SPECTRUM_BARS.forEach((b, i) => {
    const x = Math.round((12 + i * MIC_BAR_STEP) * 10) / 10;
    const yTop = Math.round((42 - b.h) * 10) / 10;
    const yBot = Math.round((42 + b.h) * 10) / 10;
    topPts.push(`${x},${yTop}`);
    botPts.unshift(`${x},${yBot}`);
  });
  return `${topPts.join(" ")} ${botPts.join(" ")}`;
})();

function MicDemoPreview() {
  return (
    <svg
      viewBox="0 0 540 84"
      width="100%"
      height="84"
      preserveAspectRatio="none"
      aria-hidden="true"
      style={{ display: "block" }}
    >
      {/* Center zero-axis baseline */}
      <line x1="0" y1="42" x2="540" y2="42" stroke="#192837" strokeWidth="1" />

      {/* Voice energy body fill */}
      <polygon points={MIC_POLYGON_POINTS} fill="rgba(91, 208, 214, 0.08)" />

      {/* Multi-band real-time speech equalizer bars */}
      {MIC_SPECTRUM_BARS.map((b, i) => {
        const x = 12 + i * MIC_BAR_STEP;
        return (
          <rect
            key={i}
            x={x}
            y={42 - b.h}
            width="2.6"
            height={b.h * 2}
            rx="1.3"
            fill={b.isVocalCore ? (i % 2 === 0 ? "#5bd0d6" : "#70b9ee") : "#476e8a"}
            className="mic-voice-bar"
            style={{ animationDelay: `${b.delay}s` }}
          />
        );
      })}
    </svg>
  );
}

export default function Home() {
  return (
    <main className="page launcher-page">
      <div className="launcher">
        <header className="launcher-header">
          <div className="launcher-title-group">
            <h1 className="launcher-title">Keyword Spotting Using LSTM</h1>
            <p className="launcher-subtitle">
              Analyze recordings, test voice commands, and evaluate your LSTM model.
            </p>
          </div>
          <div className="launcher-vocab" aria-label="Active vocabulary keywords">
            <span className="launcher-vocab-label">Active Vocabulary</span>
            <div className="launcher-vocab-tags">
              {KEYWORD_ORDER.map((kw) => (
                <span key={kw} className="launcher-vocab-tag">
                  {kw}
                </span>
              ))}
            </div>
          </div>
        </header>

        <section className="launcher-primary-grid" aria-label="Primary tools">
          <Link href="/analyze" className="card launcher-card launcher-primary-card">
            <div className="launcher-card-content">
              <div className="launcher-card-title-row">
                <span className="launcher-card-icon" aria-hidden="true">
                  <WaveIcon size={24} />
                </span>
                <h2 className="launcher-card-title">Audio Analysis</h2>
              </div>
              <p className="launcher-card-desc">
                Inspect detected commands in a WAV recording.
              </p>
            </div>
            <div className="launcher-card-preview launcher-waveform-preview" aria-hidden="true">
              <WaveformPreview />
            </div>
            <div className="launcher-card-action">
              <span>Open Audio Analysis</span>
              <ArrowRight size={16} className="action-arrow" />
            </div>
          </Link>

          <Link href="/live" className="card launcher-card launcher-primary-card">
            <div className="launcher-card-content">
              <div className="launcher-card-title-row">
                <span className="launcher-card-icon" aria-hidden="true">
                  <MicIcon size={24} />
                </span>
                <h2 className="launcher-card-title">Voice Control</h2>
              </div>
              <p className="launcher-card-desc">
                Try real-time commands with the microphone.
              </p>
            </div>
            <div className="launcher-card-preview launcher-waveform-preview" aria-hidden="true">
              <MicDemoPreview />
            </div>
            <div className="launcher-card-action">
              <span>Open Voice Control</span>
              <ArrowRight size={16} className="action-arrow" />
            </div>
          </Link>
        </section>

        <section aria-label="Secondary tools">
          <Link href="/dashboard" className="card launcher-card launcher-eval-card">
            <div className="launcher-eval-inner">
              <div className="launcher-eval-left">
                <span className="launcher-card-icon" aria-hidden="true">
                  <ChartIcon size={24} />
                </span>
                <div className="launcher-eval-text">
                  <h2 className="launcher-card-title">Model Evaluation</h2>
                  <p className="launcher-card-desc">
                    Training history, confusion matrix, and detection metrics.
                  </p>
                </div>
              </div>
              <div className="launcher-card-action">
                <span>Open Evaluation</span>
                <ArrowRight size={16} className="action-arrow" />
              </div>
            </div>
          </Link>
        </section>

        <Suspense fallback={null}>
          <ModelStatus />
        </Suspense>

        <footer className="launcher-status" aria-label="System status">
          <div className="launcher-status-group">
            <span className="launcher-status-item">Local CPU inference (127.0.0.1)</span>
            <span className="launcher-status-sep" aria-hidden="true">·</span>
            <span className="launcher-status-item">6-keyword LSTM</span>
            <span className="launcher-status-sep" aria-hidden="true">·</span>
            <span className="launcher-status-item">Microphone audio is ephemeral and not stored</span>
          </div>
        </footer>
      </div>
    </main>
  );
}
