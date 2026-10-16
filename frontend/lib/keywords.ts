// Keyword identity colors: one fixed hue per Keyword, never cycled or reassigned.
// Validated as a categorical set against the #14181d surface (dataviz validator,
// dark mode: CVD adjacent dE >= 8.4, normal-vision >= 19.3, contrast >= 3:1).
// Every colored mark also carries its Keyword name, so color is never alone.
export const KEYWORD_COLORS: Record<string, string> = {
  left: "#3987e5",
  right: "#d95926",
  up: "#199e70",
  down: "#c98500",
  go: "#d55181",
  stop: "#008300",
};

export const REJECTED_COLOR = "#969ca6";

export function keywordColor(keyword: string): string {
  return KEYWORD_COLORS[keyword] ?? REJECTED_COLOR;
}

export const KEYWORD_ORDER = ["left", "right", "up", "down", "go", "stop"];
