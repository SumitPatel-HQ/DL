import assert from "node:assert/strict";
import { test } from "node:test";
import { computePeaks, decodeWav, envelope, inspectWav } from "../lib/wav.ts";
import { precheck } from "../lib/analysis.ts";

function wav({ rate = 16000, channels = 1, bits = 16, format = 1, samples = [0, 16384, -32768, 32767] } = {}) {
  const width = bits / 8;
  const data = samples.length * width;
  const buffer = new ArrayBuffer(44 + data);
  const view = new DataView(buffer);
  const write = (offset, text) => [...text].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0)));
  write(0, "RIFF");
  view.setUint32(4, 36 + data, true);
  write(8, "WAVE");
  write(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, format, true);
  view.setUint16(22, channels, true);
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * channels * width, true);
  view.setUint16(32, channels * width, true);
  view.setUint16(34, bits, true);
  write(36, "data");
  view.setUint32(40, data, true);
  samples.forEach((s, i) => (bits === 16 ? view.setInt16(44 + i * 2, s, true) : view.setUint8(44 + i, s)));
  return buffer;
}

test("header inspection reports format and duration", () => {
  const info = inspectWav(wav({ samples: new Array(8000).fill(0) }));
  assert.deepEqual(
    { rate: info.sampleRate, channels: info.channels, bits: info.bitsPerSample, duration: info.durationS },
    { rate: 16000, channels: 1, bits: 16, duration: 0.5 },
  );
});

test("header inspection rejects what the API rejects", () => {
  assert.equal(inspectWav(new ArrayBuffer(10)).code, "invalid_wav");
  assert.equal(inspectWav(wav({ format: 3 })).code, "invalid_wav");
  assert.equal(inspectWav(wav({ rate: 4000 })).code, "unsupported_wav");
  assert.equal(inspectWav(wav({ channels: 3, samples: [0, 0, 0] })).code, "unsupported_wav");
  assert.equal(inspectWav(wav({ samples: [] })).code, "invalid_wav");
});

test("decoding mixes PCM to mono floats and peaks keep the envelope", () => {
  const { sampleRate, samples } = decodeWav(wav());
  assert.equal(sampleRate, 16000);
  assert.deepEqual([...samples], [0, 0.5, -1, 32767 / 32768]);
  const stereo = decodeWav(wav({ channels: 2, samples: [16384, 0, -16384, -16384] }));
  assert.deepEqual([...stereo.samples], [0.25, -0.5]);
  const peaks = computePeaks(samples, 2);
  assert.deepEqual([...peaks.min], [0, -1]);
  assert.deepEqual([...peaks.max], [0.5, 32767 / 32768]);
  assert.deepEqual(envelope(peaks, 0, 4), [-1, 32767 / 32768]);
});

test("precheck stops uploads the API would reject", () => {
  const ok = inspectWav(wav());
  assert.deepEqual(precheck({ name: "meeting (1).wav", size: 52 }, ok), []);
  assert.deepEqual(precheck({ name: "réunion.wav", size: 52 }, ok), []);
  const codes = (file, header = ok) => precheck(file, header).map((p) => p.code);
  assert.deepEqual(codes({ name: "../x.wav", size: 52 }), ["invalid_filename"]);
  assert.deepEqual(codes({ name: "a.mp3", size: 52 }), ["invalid_filename"]);
  assert.deepEqual(codes({ name: "a.wav", size: 41 * 1024 * 1024 }), ["upload_too_large"]);
  assert.deepEqual(codes({ name: "a.wav", size: 0 }), ["invalid_wav"]);
  assert.deepEqual(codes({ name: "a.wav", size: 52 }, inspectWav(wav({ format: 3 }))), ["invalid_wav"]);
  const long = { ...ok, durationS: 601 };
  assert.deepEqual(codes({ name: "a.wav", size: 52 }, long), ["recording_too_long"]);
});
