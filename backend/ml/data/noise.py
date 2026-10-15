"""Generate deterministic background noise files and mix noise for Evaluation Runs."""

import math
from pathlib import Path
from typing import Literal

import numpy as np
import scipy.io.wavfile

DEFAULT_SNR_DB = 10.0
NoiseColor = Literal["brown", "blue", "violet"]


def generate_colored_noise(
    output_path: Path,
    color: NoiseColor,
    duration_s: float = 60.0,
    sample_rate: int = 16000,
    seed: int = 42,
) -> None:
    """Generate deterministic colored noise and write 16-bit PCM WAV."""
    samples = int(duration_s * sample_rate)
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(samples)
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(samples, d=1.0 / sample_rate)
    freqs[0] = freqs[1]  # avoid division by zero at DC

    if color == "brown":
        fft = fft / freqs
    elif color == "blue":
        fft = fft * np.sqrt(freqs)
    elif color == "violet":
        fft = fft * freqs
    else:
        raise ValueError(f"Unsupported noise color: {color}")

    noise = np.fft.irfft(fft, n=samples)
    noise = noise / np.max(np.abs(noise)) * 0.33
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.wavfile.write(output_path, sample_rate, (noise * 32767).astype(np.int16))


def ensure_additional_noise_sources(noise_dir: Path) -> list[Path]:
    """Create any missing blue/brown/violet noise files; return all three paths."""
    colors: list[tuple[str, NoiseColor, int]] = [
        ("blue_noise.wav", "blue", 101),
        ("brown_noise.wav", "brown", 102),
        ("violet_noise.wav", "violet", 103),
    ]
    paths = []
    for filename, color, seed in colors:
        path = noise_dir / filename
        if not path.exists():
            generate_colored_noise(path, color, duration_s=60.0, seed=seed)
        paths.append(path)
    return paths


def mix_noise(
    signal: np.ndarray,
    noise: np.ndarray,
    snr_db: float = DEFAULT_SNR_DB,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Mix additive noise into signal at target SNR (dB).

    signal: 1D float array in [-1.0, 1.0]
    noise: 1D float array in [-1.0, 1.0]
    """
    if len(signal) == 0:
        return signal.copy()
    if len(noise) == 0:
        raise ValueError("Cannot mix an empty noise recording")
    if len(noise) < len(signal):
        repeats = int(math.ceil(len(signal) / len(noise)))
        noise = np.tile(noise, repeats)

    if rng is not None and len(noise) > len(signal):
        max_start = len(noise) - len(signal)
        start = rng.integers(0, max_start + 1)
        noise_clip = noise[start : start + len(signal)]
    else:
        noise_clip = noise[: len(signal)]

    sig_rms = float(np.sqrt(np.mean(signal**2))) + 1e-10
    noise_rms = float(np.sqrt(np.mean(noise_clip**2))) + 1e-10

    target_noise_rms = sig_rms * (10.0 ** (-snr_db / 20.0))
    scaled_noise = noise_clip * (target_noise_rms / noise_rms)
    mixed = signal + scaled_noise
    clipped: np.ndarray = np.clip(mixed, -1.0, 1.0).astype(np.float32)
    return clipped
