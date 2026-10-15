import Link from "next/link";
import { Suspense } from "react";
import { ArrowRight } from "../components/icons";
import { ModelStatus } from "../components/model-status";

export const dynamic = "force-dynamic";

const KW = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"];

function WaveArt() {
  const bars = Array.from({ length: 64 }, (_, i) => 6 + Math.abs(Math.sin(i * 0.55) * Math.cos(i * 0.21)) * 34);
  return (
    <svg viewBox="0 0 320 92" width="100%" height="100%" preserveAspectRatio="none" aria-hidden="true">
      <rect x="78" y="0" width="54" height="92" fill="#d55181" opacity="0.14" />
      <rect x="196" y="0" width="48" height="92" fill="#199e70" opacity="0.14" />
      <rect x="80" y="8" width="50" height="14" rx="4" fill="#d55181" opacity="0.85" />
      <rect x="198" y="8" width="44" height="14" rx="4" fill="#199e70" opacity="0.85" />
      {bars.map((h, i) => (
        <rect key={i} x={6 + i * 4.85} y={56 - h / 2} width="2.4" height={h} rx="1.2"
          fill={i * 4.85 > 72 && i * 4.85 < 126 ? "#d55181" : i * 4.85 > 190 && i * 4.85 < 238 ? "#199e70" : "#5d6876"} />
      ))}
      <rect x="150" y="26" width="2" height="62" fill="#fff" />
    </svg>
  );
}

function LiveArt() {
  return (
    <svg viewBox="0 0 320 92" width="100%" height="100%" preserveAspectRatio="none" aria-hidden="true">
      <rect x="16" y="16" width="150" height="60" rx="8" fill="#1e242c" stroke="rgba(255,255,255,0.12)" />
      <rect x="30" y="34" width="90" height="8" rx="4" fill="#eaeef2" opacity="0.8" />
      <rect x="30" y="50" width="60" height="6" rx="3" fill="#7c8693" />
      <circle cx="210" cy="46" r="18" fill="#eaeef2" />
      <path d="M204 37v18l14-9z" fill="#0b0d10" />
      {KW.map((c, i) => <rect key={c} x={246 + i * 11} y={30 + (i % 3) * 6} width="7" height={32 - (i % 3) * 12} rx="2" fill={c} />)}
    </svg>
  );
}

function ChartArt() {
  return (
    <svg viewBox="0 0 320 92" width="100%" height="100%" preserveAspectRatio="none" aria-hidden="true">
      <path d="M12 80 C60 40, 90 28, 140 22 S220 14, 300 12" stroke="#3987e5" strokeWidth="2.5" fill="none" />
      <path d="M12 82 C60 54, 90 44, 140 40 S220 46, 300 52" stroke="#d95926" strokeWidth="2.5" fill="none" />
      <line x1="128" x2="128" y1="6" y2="88" stroke="#eaeef2" opacity="0.5" />
      <line x1="12" x2="300" y1="26" y2="26" stroke="#aeb7c2" strokeDasharray="4 4" opacity="0.6" />
    </svg>
  );
}

export default function Home() {
  return (
    <main className="page">
      <section className="hero">
        <p className="eyebrow">Local audio lab · LSTM keyword spotting</p>
        <h1>
          Find <span>left, right, up, down, go, stop</span> in audio, honestly.
        </h1>
        <p>
          One CPU-trained LSTM, used two ways: reviewing Detections in recorded WAV files and driving a hands-free demo
          from the microphone. Every quality number comes from a saved Evaluation Run.
        </p>
      </section>
      <div className="slices">
        <Link href="/analyze" className="card slice">
          <div className="slice-art"><WaveArt /></div>
          <h2>Audio Analysis</h2>
          <p>Upload a WAV, pick Keywords, and verify each Detection on a linked waveform and table by listening.</p>
          <span className="go">Open workbench <ArrowRight size={15} /></span>
        </Link>
        <Link href="/live" className="card slice">
          <div className="slice-art"><LiveArt /></div>
          <h2>Voice Control</h2>
          <p>Start the mic and steer slides and a demo player. Rejections are as visible as actions; nothing is stored.</p>
          <span className="go">Open live demo <ArrowRight size={15} /></span>
        </Link>
        <Link href="/dashboard" className="card slice">
          <div className="slice-art"><ChartArt /></div>
          <h2>Evaluation</h2>
          <p>Training curves, confusion matrix, and event metrics, each traced to checkpoint, split, and config.</p>
          <span className="go">Open dashboard <ArrowRight size={15} /></span>
        </Link>
      </div>
      <div className="facts">
        <div className="card fact"><span className="k">Vocabulary</span><span className="v">6 Keywords + Unknown + Silence</span></div>
        <div className="card fact"><span className="k">Runs on</span><span className="v">CPU · loopback 127.0.0.1 only</span></div>
        <div className="card fact"><span className="k">Continuous detector</span><span className="v">Experimental (ADR-0006)</span></div>
        <div className="card fact"><span className="k">Privacy</span><span className="v">No mic audio stored</span></div>
      </div>
      <div style={{ marginTop: 16 }}>
        <Suspense fallback={<div className="card" style={{ padding: 18 }} role="status">Checking backend readiness…</div>}>
          <ModelStatus />
        </Suspense>
      </div>
    </main>
  );
}
