// Client-side WAV header checks and PCM decoding for the waveform.
// The API validates again; these checks only stop uploads it would reject.
export type WavInfo = {
  sampleRate: number;
  channels: number;
  bitsPerSample: number;
  dataOffset: number;
  dataBytes: number;
  durationS: number;
};

export type WavProblem = { code: "invalid_wav" | "unsupported_wav"; message: string };

const PCM = 1;
const EXTENSIBLE = 0xfffe;

function tag(view: DataView, offset: number): string {
  return String.fromCharCode(
    view.getUint8(offset),
    view.getUint8(offset + 1),
    view.getUint8(offset + 2),
    view.getUint8(offset + 3),
  );
}

// Reads chunks from the header only, so the first 64 KiB of a file is enough.
export function inspectWav(buffer: ArrayBuffer): WavInfo | WavProblem {
  const view = new DataView(buffer);
  if (view.byteLength < 12 || tag(view, 0) !== "RIFF" || tag(view, 8) !== "WAVE") {
    return { code: "invalid_wav", message: "Not a RIFF/WAVE file." };
  }
  let format: { audio: number; channels: number; rate: number; bits: number } | null = null;
  let offset = 12;
  while (offset + 8 <= view.byteLength) {
    const id = tag(view, offset);
    const size = view.getUint32(offset + 4, true);
    const body = offset + 8;
    if (id === "fmt ") {
      if (size < 16 || body + 16 > view.byteLength) {
        return { code: "invalid_wav", message: "WAV format chunk is truncated." };
      }
      let audio = view.getUint16(body, true);
      if (audio === EXTENSIBLE && size >= 40 && body + 26 <= view.byteLength) {
        audio = view.getUint16(body + 24, true);
      }
      format = {
        audio,
        channels: view.getUint16(body + 2, true),
        rate: view.getUint32(body + 4, true),
        bits: view.getUint16(body + 14, true),
      };
    } else if (id === "data") {
      if (format === null) return { code: "invalid_wav", message: "WAV data precedes its format." };
      if (format.audio !== PCM) {
        return { code: "invalid_wav", message: "Use uncompressed integer PCM, not float or compressed audio." };
      }
      if (![8, 16, 24, 32].includes(format.bits)) {
        return { code: "invalid_wav", message: "Use 8, 16, 24 or 32-bit integer PCM." };
      }
      if (format.channels < 1 || format.channels > 2 || format.rate < 8000 || format.rate > 96000) {
        return { code: "unsupported_wav", message: "Use mono or stereo PCM at 8-96 kHz." };
      }
      const frameBytes = (format.channels * format.bits) / 8;
      if (size < frameBytes) return { code: "invalid_wav", message: "WAV contains no audio samples." };
      return {
        sampleRate: format.rate,
        channels: format.channels,
        bitsPerSample: format.bits,
        dataOffset: body,
        dataBytes: size,
        durationS: Math.floor(size / frameBytes) / format.rate,
      };
    }
    offset = body + size + (size % 2);
  }
  return { code: "invalid_wav", message: "WAV has no data chunk." };
}

export function isWavProblem(value: WavInfo | WavProblem): value is WavProblem {
  return "code" in value;
}

// Decode integer PCM into mono floats in [-1, 1), averaging channels.
export function decodeWav(buffer: ArrayBuffer): { sampleRate: number; samples: Float32Array } {
  const info = inspectWav(buffer);
  if (isWavProblem(info)) throw new Error(info.message);
  const view = new DataView(buffer);
  const width = info.bitsPerSample / 8;
  const available = Math.min(info.dataBytes, buffer.byteLength - info.dataOffset);
  const frames = Math.floor(available / (width * info.channels));
  const samples = new Float32Array(frames);
  for (let frame = 0; frame < frames; frame++) {
    let sum = 0;
    for (let channel = 0; channel < info.channels; channel++) {
      const at = info.dataOffset + (frame * info.channels + channel) * width;
      if (width === 1) sum += (view.getUint8(at) - 128) / 128;
      else if (width === 2) sum += view.getInt16(at, true) / 32768;
      else if (width === 3) {
        const value = view.getUint8(at) | (view.getUint8(at + 1) << 8) | (view.getInt8(at + 2) << 16);
        sum += value / 8388608;
      } else sum += view.getInt32(at, true) / 2147483648;
    }
    samples[frame] = sum / info.channels;
  }
  return { sampleRate: info.sampleRate, samples };
}

export type Peaks = { min: Float32Array; max: Float32Array; samplesPerBucket: number };

// Min/max envelope at a fixed resolution; drawing picks from it per pixel.
export function computePeaks(samples: Float32Array, samplesPerBucket: number): Peaks {
  const buckets = Math.max(1, Math.ceil(samples.length / samplesPerBucket));
  const min = new Float32Array(buckets);
  const max = new Float32Array(buckets);
  for (let bucket = 0; bucket < buckets; bucket++) {
    let low = 0;
    let high = 0;
    const end = Math.min(samples.length, (bucket + 1) * samplesPerBucket);
    for (let i = bucket * samplesPerBucket; i < end; i++) {
      const value = samples[i];
      if (value < low) low = value;
      if (value > high) high = value;
    }
    min[bucket] = low;
    max[bucket] = high;
  }
  return { min, max, samplesPerBucket };
}

// Envelope of the bucket range covering [fromSample, toSample).
export function envelope(peaks: Peaks, fromSample: number, toSample: number): [number, number] {
  const first = Math.max(0, Math.floor(fromSample / peaks.samplesPerBucket));
  const last = Math.min(peaks.min.length, Math.max(first + 1, Math.ceil(toSample / peaks.samplesPerBucket)));
  let low = 0;
  let high = 0;
  for (let i = first; i < last; i++) {
    if (peaks.min[i] < low) low = peaks.min[i];
    if (peaks.max[i] > high) high = peaks.max[i];
  }
  return [low, high];
}
