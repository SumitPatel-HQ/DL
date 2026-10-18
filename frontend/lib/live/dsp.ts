// Browser port of the frozen MFCC-v1 pipeline (backend/ml/features/pipeline.py)
// and of scipy's resample_poly, so in-browser live sessions see the same
// features the API computes. tests/dsp.test.mjs checks both against Python.
export const SAMPLE_RATE = 16000;
export const WINDOW_SAMPLES = 16000;
const FRAME_SAMPLES = 400;
const HOP_SAMPLES = 160;
const N_FFT = 512;
const N_MELS = 40;
const N_MFCC = 40;
const MEL_LOW_HZ = 20;
const MEL_HIGH_HZ = 7600;
const LOG_FLOOR = 1e-10;
const KAISER_BETA = 5;
export const WINDOW_FRAMES = Math.floor((WINDOW_SAMPLES - FRAME_SAMPLES) / HOP_SAMPLES) + 1;

function gcd(a: number, b: number): number {
  while (b) [a, b] = [b, a % b];
  return a;
}

// Modified Bessel function of the first kind, order 0 (power series).
function besselI0(x: number): number {
  let sum = 1;
  let term = 1;
  for (let k = 1; k < 200; k++) {
    term *= (x / (2 * k)) ** 2;
    sum += term;
    if (term < sum * 1e-17) break;
  }
  return sum;
}

function sinc(x: number): number {
  if (x === 0) return 1;
  const y = Math.PI * x;
  return Math.sin(y) / y;
}

// scipy.signal.firwin(numtaps, cutoff, window=("kaiser", 5.0)): low-pass, scaled to unit DC gain.
function firwin(numtaps: number, cutoff: number): Float64Array {
  const alpha = (numtaps - 1) / 2;
  const h = new Float64Array(numtaps);
  const norm = besselI0(KAISER_BETA);
  let sum = 0;
  for (let n = 0; n < numtaps; n++) {
    const m = n - alpha;
    const ratio = (n - alpha) / alpha;
    const window = besselI0(KAISER_BETA * Math.sqrt(Math.max(0, 1 - ratio * ratio))) / norm;
    h[n] = cutoff * sinc(cutoff * m) * window;
    sum += h[n];
  }
  for (let n = 0; n < numtaps; n++) h[n] /= sum;
  return h;
}

function upfirdnLength(filterLength: number, inputLength: number, up: number, down: number): number {
  const padded = inputLength + Math.floor((filterLength + (((-filterLength % up) + up) % up)) / up) - 1;
  const total = padded * up;
  return Math.floor(total / down) + (total % down > 0 ? 1 : 0);
}

type Polyphase = { up: number; down: number; h: Float64Array; preRemove: number };
const designs = new Map<string, Polyphase>();

function design(up: number, down: number, inputLength: number): Polyphase {
  const key = `${up}/${down}/${inputLength}`;
  const cached = designs.get(key);
  if (cached) return cached;
  const maxRate = Math.max(up, down);
  const halfLength = 10 * maxRate;
  const base = firwin(2 * halfLength + 1, 1 / maxRate);
  const prePad = down - (halfLength % down);
  const preRemove = Math.floor((halfLength + prePad) / down);
  const outputLength = Math.ceil((inputLength * up) / down);
  let postPad = 0;
  while (upfirdnLength(base.length + prePad + postPad, inputLength, up, down) < outputLength + preRemove) {
    postPad++;
  }
  const h = new Float64Array(prePad + base.length + postPad);
  for (let i = 0; i < base.length; i++) h[prePad + i] = base[i] * up;
  const result = { up, down, h, preRemove };
  if (designs.size > 16) designs.clear();
  designs.set(key, result);
  return result;
}

// scipy.signal.resample_poly(x, 16000 / g, rate / g) with the default Kaiser window
// and zero padding, as `canonical_audio` calls it.
export function resampleTo16k(input: Float64Array, rate: number): Float64Array {
  if (rate === SAMPLE_RATE) return input.slice();
  const divisor = gcd(rate, SAMPLE_RATE);
  const up = SAMPLE_RATE / divisor;
  const down = rate / divisor;
  const { h, preRemove } = design(up, down, input.length);
  const outputLength = Math.ceil((input.length * up) / down);
  const output = new Float64Array(outputLength);
  const taps = h.length;
  for (let i = 0; i < outputLength; i++) {
    const position = (i + preRemove) * down;
    const first = Math.max(0, Math.ceil((position - taps + 1) / up));
    const last = Math.min(input.length - 1, Math.floor(position / up));
    let sum = 0;
    for (let q = first; q <= last; q++) sum += h[position - q * up] * input[q];
    output[i] = sum;
  }
  return output;
}

// One second of 16 kHz audio from a window at the source rate: resample, then
// truncate or right-pad with zeros, like LiveStream.take_windows.
export function canonicalWindow(samples: Float32Array, rate: number): Float32Array {
  const resampled = resampleTo16k(Float64Array.from(samples), rate);
  const window = new Float32Array(WINDOW_SAMPLES);
  window.set(resampled.subarray(0, WINDOW_SAMPLES));
  return window;
}

const HAMMING = Float64Array.from(
  { length: FRAME_SAMPLES },
  (_, n) => 0.54 - 0.46 * Math.cos((2 * Math.PI * n) / (FRAME_SAMPLES - 1)),
);

const MEL_FILTERS: Float64Array[] = (() => {
  const toMel = (hz: number) => 2595 * Math.log10(1 + hz / 700);
  const low = toMel(MEL_LOW_HZ);
  const high = toMel(MEL_HIGH_HZ);
  const edges = Array.from({ length: N_MELS + 2 }, (_, i) => {
    const mel = low + ((high - low) * i) / (N_MELS + 1);
    return 700 * (10 ** (mel / 2595) - 1);
  });
  const bins = N_FFT / 2 + 1;
  return Array.from({ length: N_MELS }, (_, m) => {
    const filter = new Float64Array(bins);
    for (let k = 0; k < bins; k++) {
      const hz = (k * SAMPLE_RATE) / N_FFT;
      const rising = (hz - edges[m]) / (edges[m + 1] - edges[m]);
      const falling = (edges[m + 2] - hz) / (edges[m + 2] - edges[m + 1]);
      filter[k] = Math.max(0, Math.min(rising, falling));
    }
    return filter;
  });
})();

// Orthonormal DCT-II basis, first N_MFCC rows.
const DCT: Float64Array[] = Array.from({ length: N_MFCC }, (_, k) => {
  const scale = Math.sqrt((k === 0 ? 1 : 2) / N_MELS);
  return Float64Array.from(
    { length: N_MELS },
    (_, n) => scale * Math.cos((Math.PI * k * (2 * n + 1)) / (2 * N_MELS)),
  );
});

const TWIDDLE_RE = Float64Array.from({ length: N_FFT / 2 }, (_, k) => Math.cos((-2 * Math.PI * k) / N_FFT));
const TWIDDLE_IM = Float64Array.from({ length: N_FFT / 2 }, (_, k) => Math.sin((-2 * Math.PI * k) / N_FFT));
const BIT_REVERSED = Uint16Array.from({ length: N_FFT }, (_, i) => {
  let reversed = 0;
  for (let bit = 1, j = i; bit < N_FFT; bit <<= 1, j >>= 1) reversed = (reversed << 1) | (j & 1);
  return reversed;
});

// In-place radix-2 FFT of length N_FFT.
function fft(re: Float64Array, im: Float64Array): void {
  for (let i = 0; i < N_FFT; i++) {
    const j = BIT_REVERSED[i];
    if (j > i) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let size = 2; size <= N_FFT; size <<= 1) {
    const half = size >> 1;
    const step = N_FFT / size;
    for (let start = 0; start < N_FFT; start += size) {
      for (let k = 0; k < half; k++) {
        const wr = TWIDDLE_RE[k * step];
        const wi = TWIDDLE_IM[k * step];
        const a = start + k;
        const b = a + half;
        const tr = re[b] * wr - im[b] * wi;
        const ti = re[b] * wi + im[b] * wr;
        re[b] = re[a] - tr;
        im[b] = im[a] - ti;
        re[a] += tr;
        im[a] += ti;
      }
    }
  }
}

// Normalized [WINDOW_FRAMES, 40] MFCCs of one canonical 16 kHz window, row-major.
export function mfcc(window: Float32Array, mean: readonly number[], std: readonly number[]): Float32Array {
  const features = new Float32Array(WINDOW_FRAMES * N_MFCC);
  const re = new Float64Array(N_FFT);
  const im = new Float64Array(N_FFT);
  const power = new Float64Array(N_FFT / 2 + 1);
  const logMel = new Float64Array(N_MELS);
  for (let frame = 0; frame < WINDOW_FRAMES; frame++) {
    re.fill(0);
    im.fill(0);
    const offset = frame * HOP_SAMPLES;
    for (let n = 0; n < FRAME_SAMPLES; n++) re[n] = window[offset + n] * HAMMING[n];
    fft(re, im);
    for (let k = 0; k < power.length; k++) power[k] = (re[k] * re[k] + im[k] * im[k]) / N_FFT;
    for (let m = 0; m < N_MELS; m++) {
      const filter = MEL_FILTERS[m];
      let energy = 0;
      for (let k = 0; k < power.length; k++) energy += power[k] * filter[k];
      logMel[m] = Math.log(Math.max(energy, LOG_FLOOR));
    }
    for (let k = 0; k < N_MFCC; k++) {
      const basis = DCT[k];
      let value = 0;
      for (let n = 0; n < N_MELS; n++) value += basis[n] * logMel[n];
      // Python rounds the coefficient to float32 before normalizing.
      features[frame * N_MFCC + k] = (Math.fround(value) - mean[k]) / std[k];
    }
  }
  return features;
}

export function windowRms(window: Float32Array): number {
  if (window.length === 0) return 0;
  let sum = 0;
  for (const value of window) sum += value * value;
  return Math.sqrt(sum / window.length);
}
