// In-app demo stage. Voice and keyboard both call applyAction, so they share
// one action path; nothing here reaches the OS (ADR-0004).
export type Action =
  | "next_slide"
  | "previous_slide"
  | "play"
  | "pause"
  | "volume_up"
  | "volume_down";

export const ACTIONS: readonly Action[] = [
  "next_slide",
  "previous_slide",
  "play",
  "pause",
  "volume_up",
  "volume_down",
];

export function isAction(value: unknown): value is Action {
  return typeof value === "string" && (ACTIONS as readonly string[]).includes(value);
}

export type Stage = { slide: number; slides: number; playing: boolean; volume: number };

export const VOLUME_STEP = 10;

export function initialStage(slides: number): Stage {
  return { slide: 1, slides, playing: false, volume: 50 };
}

export type ActionOutcome = { stage: Stage; result: string; changed: boolean };

export function applyAction(stage: Stage, action: Action): ActionOutcome {
  const outcome = (next: Stage, result: string): ActionOutcome => ({
    stage: next,
    result,
    changed: next !== stage,
  });
  switch (action) {
    case "next_slide":
      return stage.slide < stage.slides
        ? outcome({ ...stage, slide: stage.slide + 1 }, `Slide ${stage.slide} → ${stage.slide + 1}`)
        : outcome(stage, "Already on the last slide");
    case "previous_slide":
      return stage.slide > 1
        ? outcome({ ...stage, slide: stage.slide - 1 }, `Slide ${stage.slide} → ${stage.slide - 1}`)
        : outcome(stage, "Already on the first slide");
    case "play":
      return stage.playing ? outcome(stage, "Already playing") : outcome({ ...stage, playing: true }, "Playing");
    case "pause":
      return stage.playing ? outcome({ ...stage, playing: false }, "Paused") : outcome(stage, "Already paused");
    case "volume_up": {
      const volume = Math.min(100, stage.volume + VOLUME_STEP);
      return volume === stage.volume
        ? outcome(stage, "Volume already 100")
        : outcome({ ...stage, volume }, `Volume ${stage.volume} → ${volume}`);
    }
    case "volume_down": {
      const volume = Math.max(0, stage.volume - VOLUME_STEP);
      return volume === stage.volume
        ? outcome(stage, "Volume already 0")
        : outcome({ ...stage, volume }, `Volume ${stage.volume} → ${volume}`);
    }
  }
}

// Keyboard parity: one key per action, same as each Keyword.
export const KEY_BINDINGS: Record<string, Action> = {
  ArrowRight: "next_slide",
  ArrowLeft: "previous_slide",
  g: "play",
  s: "pause",
  ArrowUp: "volume_up",
  ArrowDown: "volume_down",
};

export const KEY_LABELS: Record<Action, string> = {
  next_slide: "→",
  previous_slide: "←",
  play: "G",
  pause: "S",
  volume_up: "↑",
  volume_down: "↓",
};
