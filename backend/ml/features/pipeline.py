"""Explicit framing, HTK Mel triangles, log power and orthonormal DCT-II.

No peak normalization, pre-emphasis, centering, liftering or delta coefficients.
One-second classifier windows are right-padded or truncated before framing.
"""

import json
import math
import wave
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.fft import dct
from scipy.signal import resample_poly

Audio = NDArray[np.float32]
LABELS = ("left", "right", "up", "down", "go", "stop", "unknown", "silence")
SAMPLE_RATE = 16000
FRAME_MS, HOP_MS = 25, 10
FRAME_SAMPLES, HOP_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000, SAMPLE_RATE * HOP_MS // 1000
N_MFCC = N_MELS = 40
N_FFT = 512
WINDOW_SAMPLES = SAMPLE_RATE
MEL_LOW_HZ, MEL_HIGH_HZ = 20, 7600
LOG_FLOOR, STD_FLOOR = 1e-10, 1e-6
FEATURE_CONFIG = {
    "version": "mfcc-v1",
    "sample_rate": SAMPLE_RATE,
    "channels": 1,
    "dtype": "float32",
    "frame_ms": FRAME_MS,
    "hop_ms": HOP_MS,
    "n_mfcc": N_MFCC,
    "n_mels": N_MELS,
    "n_fft": N_FFT,
    "window_samples": WINDOW_SAMPLES,
    "window": "symmetric-hamming",
    "mel_scale": f"htk-unscaled-triangles-{MEL_LOW_HZ}-{MEL_HIGH_HZ}Hz",
    "spectrum": f"abs-rfft-squared/{N_FFT}",
    "log": "natural-floor-1e-10",
    "dct": "type-2-ortho",
    "padding": "right-zero-to-1s-truncate-longer-no-center",
    "resampling": "scipy-resample-poly-kaiser-5-constant",
    "normalization": "train-only-per-coefficient-population-std-floor-1e-6",
}


def canonical_audio(samples: NDArray[Any], sample_rate: int) -> Audio:
    """Accept normalized floating PCM [samples] or [samples, channels]."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if samples.size == 0:
        raise ValueError("Audio must not be empty")
    if samples.ndim not in (1, 2) or not np.issubdtype(samples.dtype, np.floating):
        raise ValueError("Audio must be floating PCM with one or two dimensions")
    if not np.isfinite(samples).all():
        raise ValueError("Audio must contain only finite samples")
    audio = samples.astype(np.float64)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        divisor = math.gcd(sample_rate, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // divisor, sample_rate // divisor)
    result = np.asarray(audio, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Canonical audio must be finite")
    return result


def read_wav(path: Path) -> Audio:
    """Decode uncompressed integer PCM WAV, including 8/16/24/32-bit samples."""
    try:
        with wave.open(str(path), "rb") as wav:
            width, channels, rate = wav.getsampwidth(), wav.getnchannels(), wav.getframerate()
            frames = wav.getnframes()
            raw = wav.readframes(frames)
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"Invalid PCM WAV: {path.name}") from exc
    if len(raw) != frames * channels * width:
        raise ValueError("Truncated WAV samples")
    if width == 1:
        pcm = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128
    elif width in (2, 4):
        pcm = np.frombuffer(raw, dtype=f"<i{width}").astype(np.float32) / (2 ** (8 * width - 1))
    elif width == 3:
        octets = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        signed = octets[:, 0] | (octets[:, 1] << 8) | (octets[:, 2] << 16)
        pcm = ((signed ^ 0x800000) - 0x800000).astype(np.float32) / 8388608
    else:
        raise ValueError("WAV must contain 8/16/24/32-bit integer PCM")
    return canonical_audio(pcm.reshape(-1, channels), rate)


@dataclass(frozen=True)
class Normalization:
    mean: tuple[float, ...]
    std: tuple[float, ...]
    frame_count: int
    dataset_version: str
    split_version: str
    manifest_sha256: str

    def __post_init__(self) -> None:
        if len(self.mean) != N_MFCC or len(self.std) != N_MFCC:
            raise ValueError("Normalization requires 40 coefficients")
        if not np.isfinite([self.mean, self.std]).all() or min(self.std) < STD_FLOOR:
            raise ValueError("Normalization must be finite with positive std")
        if self.frame_count <= 0 or not all(
            (self.dataset_version, self.split_version, self.manifest_sha256)
        ):
            raise ValueError("Normalization requires training provenance")

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "features": FEATURE_CONFIG,
                    "labels": LABELS,
                    "fit_split": "train",
                    **asdict(self),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "Normalization":
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            value.pop("features") != FEATURE_CONFIG
            or value.pop("labels") != list(LABELS)
            or value.pop("fit_split") != "train"
        ):
            raise ValueError("Incompatible feature metadata or class order")
        value["mean"], value["std"] = tuple(value["mean"]), tuple(value["std"])
        return cls(**value)


class FeaturePipeline:
    def __init__(self, normalization: Normalization | None = None) -> None:
        self.normalization = normalization
        mel = np.linspace(
            2595 * np.log10(1 + MEL_LOW_HZ / 700),
            2595 * np.log10(1 + MEL_HIGH_HZ / 700),
            N_MELS + 2,
        )
        edges = 700 * (10 ** (mel / 2595) - 1)
        hz = np.fft.rfftfreq(N_FFT, 1 / SAMPLE_RATE)
        self._filters = np.maximum(
            0,
            np.minimum(
                (hz[None, :] - edges[:-2, None]) / (edges[1:-1] - edges[:-2])[:, None],
                (edges[2:, None] - hz[None, :]) / (edges[2:] - edges[1:-1])[:, None],
            ),
        )

    def from_wav(self, path: Path) -> Audio:
        return self.from_audio(read_wav(path), SAMPLE_RATE)

    def from_audio(self, samples: NDArray[Any], sample_rate: int) -> Audio:
        audio = canonical_audio(samples, sample_rate)[:WINDOW_SAMPLES]
        return self._mfcc(np.pad(audio, (0, WINDOW_SAMPLES - len(audio))))

    def from_stream(self, samples: NDArray[Any], sample_rate: int) -> Audio:
        """Same frames as `from_audio`, over audio of any length: no padding or truncation."""
        audio = canonical_audio(samples, sample_rate)
        if len(audio) < FRAME_SAMPLES:
            raise ValueError("Audio is shorter than one frame")
        return self._mfcc(audio)

    def _mfcc(self, audio: Audio) -> Audio:
        frames = np.lib.stride_tricks.sliding_window_view(audio, FRAME_SAMPLES)[::HOP_SAMPLES]
        spectrum = np.abs(np.fft.rfft(frames * np.hamming(FRAME_SAMPLES), n=N_FFT)) ** 2 / N_FFT
        log_mel = np.log(np.maximum(spectrum @ self._filters.T, LOG_FLOOR))
        features = np.asarray(
            dct(log_mel, type=2, norm="ortho", axis=1)[:, :N_MFCC], dtype=np.float32
        )
        if self.normalization is not None:
            features = np.asarray(
                (features - np.array(self.normalization.mean)) / np.array(self.normalization.std),
                dtype=np.float32,
            )
        return features

    @staticmethod
    def fit_normalization(
        training_features: Iterable[Audio],
        *,
        split: str,
        dataset_version: str,
        split_version: str,
        manifest_sha256: str,
    ) -> Normalization:
        if split != "train":
            raise ValueError("Normalization may only be fitted on train")
        count = 0
        mean, m2 = np.zeros(N_MFCC), np.zeros(N_MFCC)
        for features in training_features:
            if features.ndim != 2 or features.shape[1] != N_MFCC or not np.isfinite(features).all():
                raise ValueError("Expected finite [frames, 40] training features")
            if len(features) == 0:
                continue
            batch = features.astype(np.float64)
            delta = batch.mean(axis=0) - mean
            total = count + len(batch)
            m2 += ((batch - batch.mean(axis=0)) ** 2).sum(axis=0)
            m2 += delta**2 * count * len(batch) / total
            mean += delta * len(batch) / total
            count = total
        if count == 0:
            raise ValueError("No training frames")
        return Normalization(
            tuple(mean),
            tuple(np.maximum(np.sqrt(m2 / count), STD_FLOOR)),
            count,
            dataset_version,
            split_version,
            manifest_sha256,
        )
