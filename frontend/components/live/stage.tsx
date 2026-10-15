"use client";

import type { Action, Stage } from "../../lib/live/stage.ts";
import { ChevronLeft, ChevronRight, PauseIcon, PlayIcon, VolumeIcon } from "../icons";

export const SLIDES = [
  { kicker: "01 / Problem", title: "Find a few words in audio, locally", body: "Six trained Keywords: left, right, up, down, go, stop. Unknown speech and Silence are classes too, so the model can say no." },
  { kicker: "02 / Features", title: "40 MFCC every 10 ms", body: "16 kHz mono, 25 ms Hamming frames, train-only normalization. The same pipeline feeds training, batch analysis, and this live demo." },
  { kicker: "03 / Model", title: "A 2-layer LSTM on CPU", body: "E02: hidden size 128, dropout 0.2, Adam with early stopping. It classifies one 1 s window at a time." },
  { kicker: "04 / Live path", title: "One utterance, at most one action", body: "Windows every 0.25 s. A Keyword must win two windows in a row above threshold, and a 1.2 s cooldown swallows repeats." },
  { kicker: "05 / Honesty", title: "Confidence is a score, not accuracy", body: "Isolated macro-F1 and continuous event-F1 are reported separately, from saved Evaluation Runs, with unmet targets left unmet." },
  { kicker: "06 / Done", title: "Say “stop” to pause, “go” to play", body: "Every action stays inside this page. Nothing reaches the OS, and no mic audio is stored." },
];

export function SlideDeck({
  stage,
  flash,
  direction,
  onAction,
}: {
  stage: Stage;
  flash: Action | null;
  direction: 1 | -1;
  onAction: (action: Action) => void;
}) {
  const slide = SLIDES[stage.slide - 1];
  return (
    <div className="slide-frame" aria-roledescription="slide deck" aria-label={`Slide ${stage.slide} of ${stage.slides}`}>
      <div key={stage.slide} className="slide" style={{ ["--from" as string]: `${direction * 28}px` }} aria-live="polite">
        <span className="slide-kicker">{slide.kicker}</span>
        <h3>{slide.title}</h3>
        <p>{slide.body}</p>
      </div>
      <button type="button" className={`slide-nav prev${flash === "previous_slide" ? " flash" : ""}`} onClick={() => onAction("previous_slide")} aria-label="Previous slide">
        <ChevronLeft />
      </button>
      <button type="button" className={`slide-nav next${flash === "next_slide" ? " flash" : ""}`} onClick={() => onAction("next_slide")} aria-label="Next slide">
        <ChevronRight />
      </button>
      <div className="slide-dots" aria-hidden="true">
        {SLIDES.map((s, i) => <i key={s.kicker} className={i + 1 === stage.slide ? "on" : undefined} />)}
      </div>
    </div>
  );
}

const BARS = 36;

export function Player({ stage, flash, onAction }: { stage: Stage; flash: Action | null; onAction: (action: Action) => void }) {
  return (
    <div className="card player">
      <button
        type="button"
        className={`play-btn${flash === "play" || flash === "pause" ? " flash" : ""}`}
        onClick={() => onAction(stage.playing ? "pause" : "play")}
        aria-label={stage.playing ? "Pause demo track" : "Play demo track"}
      >
        {stage.playing ? <PauseIcon size={20} /> : <PlayIcon size={20} />}
      </button>
      <div className="track">
        <span className="title">Demo track · {stage.playing ? "playing" : "paused"}</span>
        <div className={`viz${stage.playing ? " playing" : ""}`} aria-hidden="true">
          {Array.from({ length: BARS }, (_, i) => (
            <i key={i} style={{ animationDelay: `${(i * 137) % 900}ms`, opacity: 0.35 + (stage.volume / 100) * 0.65 }} />
          ))}
        </div>
        <span className="sub">Silent visualizer: no sound is played, so the mic never hears the player.</span>
      </div>
      <div className={`volume${flash === "volume_up" || flash === "volume_down" ? " flash" : ""}`}>
        <span className="vlabel"><span style={{ display: "inline-flex", gap: 6, alignItems: "center" }}><VolumeIcon size={14} /> Volume</span><b className="mono">{stage.volume}</b></span>
        <div className="vtrack" role="meter" aria-valuemin={0} aria-valuemax={100} aria-valuenow={stage.volume} aria-label="Demo volume">
          <div style={{ width: `${stage.volume}%` }} />
        </div>
      </div>
    </div>
  );
}
