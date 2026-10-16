"""CLI: train the live-mic LSTM with waveform augmentation (gain, noise, shift).

The P0 models were trained on clean, loudly recorded clips and fail on laptop mics:
quiet speech, room noise and word position inside a sliding window all fall outside
their training distribution. This script keeps the network, features and
normalization, and changes only the data: every epoch draws fresh random gain,
noise mixes and shifts, and Silence is synthesized from train-split noise
recordings plus very quiet synthetic noise. Validation uses the manifest validation
speakers with a fixed augmentation draw. Test speakers are never read.
"""

import argparse
import copy
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from multiprocessing import Pool
from multiprocessing.pool import AsyncResult
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from ml.checkpoint import save_checkpoint
from ml.data.manifest import load_manifest, noise_sources_by_split, speaker_split
from ml.features import Normalization
from ml.features.pipeline import (
    FRAME_SAMPLES,
    HOP_SAMPLES,
    LABELS,
    WINDOW_SAMPLES,
    Audio,
    FeaturePipeline,
    read_wav,
)
from ml.networks import LstmClassifier, parameter_count
from ml.runtime import code_revision, seed_everything
from ml.vocabulary import keywords_for

KEYWORDS = keywords_for(LABELS)


def speaker_of(path: Path) -> str:
    return path.name.split("_nohash_")[0]


def keyword_count(labels: list[int]) -> int:
    """Median clips per Keyword: the unit for Silence and synthetic Unknown counts."""
    return int(np.median([labels.count(i) for i in range(len(KEYWORDS))]))


def unknown_words(root: Path) -> tuple[str, ...]:
    """Every word folder in the corpus except the Keywords; the 128-clip subset has 5."""
    return tuple(
        sorted(
            d.name
            for d in root.iterdir()
            if d.is_dir() and d.name not in KEYWORDS and not d.name.startswith("_")
        )
    )


def read_pcm16(path: Path) -> NDArray[np.int16]:
    """16-bit storage halves the memory of the full corpus; 16-bit WAVs convert back exactly."""
    return np.clip(np.round(read_wav(path) * 32768), -32768, 32767).astype(np.int16)


def load_clips(
    root: Path, split: str, unknown_cap: int | None = None, pool: Any = None
) -> tuple[list[Any], list[int]]:
    """Keyword clips of one speaker split, plus up to `unknown_cap` clips per other word.

    With `pool`, files are decoded in worker processes.
    """
    chosen: list[Path] = []
    labels: list[int] = []
    for word in (*KEYWORDS, *unknown_words(root)):
        label = LABELS.index(word if word in KEYWORDS else "unknown")
        paths = [
            path
            for path in sorted((root / word).glob("*.wav"))
            if speaker_split(speaker_of(path)) == split
        ]
        if label == LABELS.index("unknown") and unknown_cap is not None:
            paths = paths[:unknown_cap]
        chosen += paths
        labels += [label] * len(paths)
    clips = pool.map(read_pcm16, chosen, chunksize=256) if pool else [read_pcm16(p) for p in chosen]
    return clips, labels


HOLDOUT_WORDS = frozenset(
    "hello hi okay testing computer yellow morning low below follow row open pause dawn town"
    .split()
)


def load_tts(tts_dir: Path, holdout: bool) -> tuple[list[Audio], list[int], list[Audio], list[int]]:
    """Synthetic speech (scripts/gen_tts.ps1): keywords and other words as Unknown.

    With `holdout`, HOLDOUT_WORDS go to the validation set only, to measure how well
    Unknown generalizes to words never seen in training.
    """
    train_clips, train_labels, val_clips, val_labels = [], [], [], []
    for path in sorted(tts_dir.glob("*.wav")) if tts_dir.is_dir() else []:
        word = path.name.split("__")[0].lower()
        label = LABELS.index(word if word in KEYWORDS else "unknown")
        if holdout and word in HOLDOUT_WORDS:
            val_clips.append(read_wav(path))
            val_labels.append(label)
        elif not (holdout and "_" in word and set(word.split("_")) & HOLDOUT_WORDS):
            train_clips.append(read_wav(path))
            train_labels.append(label)
    return train_clips, train_labels, val_clips, val_labels


def load_noise(root: Path, stems: list[str]) -> list[Audio]:
    return [read_wav(root / "_background_noise_" / f"{stem}.wav") for stem in stems]


AMI_CONTEXT_SAMPLES = WINDOW_SAMPLES + WINDOW_SAMPLES // 2
AMI_NEGATIVES_PER_MEETING = 400
AMI_SILENCE_DB = 10.0
AMI_KEYWORD_CAP = 150


def keyword_runs(labels: NDArray[np.int64]) -> list[tuple[int, int, int]]:
    """(class, first frame, end frame) for every contiguous Keyword run of frame labels."""
    edges = np.flatnonzero(np.diff(labels)) + 1
    bounds = np.r_[0, edges, len(labels)]
    return [
        (int(labels[a]), int(a), int(b))
        for a, b in zip(bounds[:-1], bounds[1:], strict=True)
        if labels[a] < len(KEYWORDS)
    ]


def load_ami(ami_dir: Path, role: str, seed: int) -> tuple[list[Audio], list[int]]:
    """Fluent meeting speech from the temporal AMI split; `role` is train or dev.

    Keyword tokens become 1.5 s clips centered on the word, so a random 1 s crop
    keeps it inside the window. Hard negatives are 1 s windows with no Keyword
    frame: Unknown when louder than the meeting's quiet floor, else Silence. The
    validation and test meetings (IN1002, EN2004a) and their speakers are excluded
    by the temporal manifest.
    """
    manifest = json.loads((ami_dir / "manifest.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(seed)
    clips, labels = [], []
    half = AMI_CONTEXT_SAMPLES // 2
    for row in manifest["recordings"]:
        if row["kind"] != "meeting" or row["role"] != role:
            continue
        audio = read_wav(ami_dir / "sources" / f"{row['recording_id']}.Mix-Headset.wav")
        frames = np.load(ami_dir / "features" / row["file"])["labels"]
        is_keyword = frames < len(KEYWORDS)
        for label, first, end in keyword_runs(frames):
            center = (first + end - 1) * HOP_SAMPLES // 2 + FRAME_SAMPLES // 2
            if half <= center <= len(audio) - half:
                clips.append(audio[center - half : center + half].copy())
                labels.append(label)
        # A window is clean when none of its frames is a Keyword frame.
        keyword_frames = np.r_[0, np.cumsum(is_keyword)]
        starts = np.arange(0, len(audio) - WINDOW_SAMPLES, WINDOW_SAMPLES // 2)
        span = WINDOW_SAMPLES // HOP_SAMPLES
        first_frames = np.minimum(starts // HOP_SAMPLES, len(frames))
        last_frames = np.minimum(first_frames + span, len(frames))
        clean = starts[keyword_frames[last_frames] == keyword_frames[first_frames]]
        if len(clean) == 0:  # short or Keyword-dense meeting: no negatives to draw
            continue
        chosen = rng.choice(clean, min(AMI_NEGATIVES_PER_MEETING, len(clean)), replace=False)
        windows = [audio[s : s + WINDOW_SAMPLES].copy() for s in chosen]
        rms = np.array([np.sqrt(np.mean(w**2)) + 1e-9 for w in windows])
        floor = np.percentile(rms, 10)
        for window, level in zip(windows, rms, strict=True):
            clips.append(window)
            loud = 20 * np.log10(level / floor) >= AMI_SILENCE_DB
            labels.append(LABELS.index("unknown" if loud else "silence"))
    return clips, labels


def ami_keep(labels: list[int]) -> list[float]:
    """Cap each AMI Keyword at about AMI_KEYWORD_CAP draws per epoch (`right` dominates)."""
    counts = {label: labels.count(label) for label in set(labels)}
    return [
        min(1.0, AMI_KEYWORD_CAP / counts[label]) if label < len(KEYWORDS) else 1.0
        for label in labels
    ]


def pad(audio: Audio) -> Audio:
    audio = audio[:WINDOW_SAMPLES]
    return np.pad(audio, (0, WINDOW_SAMPLES - len(audio)))


def synthetic_noise(rng: np.random.Generator) -> Audio:
    """White, pink, brown or hum-plus-hiss at a random microphone-floor level."""
    kind = rng.integers(4)
    white = rng.standard_normal(WINDOW_SAMPLES)
    if kind == 0:
        noise = white
    elif kind == 1:
        noise = np.cumsum(white)
        noise -= np.convolve(noise, np.ones(400) / 400, "same")
    elif kind == 2:
        noise = np.cumsum(np.cumsum(white))
        noise -= np.convolve(noise, np.ones(400) / 400, "same")
    else:
        t = np.arange(WINDOW_SAMPLES) / 16000
        noise = np.sin(2 * np.pi * rng.choice([50.0, 60.0]) * t) * 3 + white
    noise = noise / (noise.std() + 1e-9)
    return (noise * 10 ** rng.uniform(-4.2, -1.5)).astype(np.float32)


def noise_crop(noises: list[Audio], rng: np.random.Generator) -> Audio:
    source = noises[rng.integers(len(noises))]
    start = int(rng.integers(0, max(1, len(source) - WINDOW_SAMPLES)))
    return pad(source[start : start + WINDOW_SAMPLES])


def augment_speech(audio: Audio, noises: list[Audio], rng: np.random.Generator) -> Audio:
    if len(audio) > WINDOW_SAMPLES:  # phrases: a random one-second crop
        start = int(rng.integers(0, len(audio) - WINDOW_SAMPLES + 1))
        audio = audio[start : start + WINDOW_SAMPLES]
    audio = pad(audio)
    shift = int(rng.integers(-5000, 5000))
    audio = np.roll(audio, shift)
    if shift > 0:
        audio[:shift] = 0
    elif shift < 0:
        audio[shift:] = 0
    peak = float(np.abs(audio).max()) + 1e-9
    audio = audio / peak * 10 ** rng.uniform(-1.9, -0.05)  # peak -38 dBFS .. -1 dBFS
    if rng.random() < 0.85:
        noise = noise_crop(noises, rng) if rng.random() < 0.6 else synthetic_noise(rng)
        speech_rms = float(np.sqrt((audio**2).mean())) + 1e-9
        noise = noise / (float(np.sqrt((noise**2).mean())) + 1e-9)
        audio = audio + noise * speech_rms / 10 ** (rng.uniform(3, 30) / 20)
    return np.clip(audio, -1, 1).astype(np.float32)


def augment_real(audio: Audio, noises: list[Audio], rng: np.random.Generator) -> Audio:
    """Meeting audio keeps its own acoustics: a random 1 s crop, gain, and sometimes noise."""
    if len(audio) > WINDOW_SAMPLES:
        start = int(rng.integers(0, len(audio) - WINDOW_SAMPLES + 1))
        audio = audio[start : start + WINDOW_SAMPLES]
    audio = pad(audio) * 10 ** rng.uniform(-0.6, 0.6)
    if rng.random() < 0.3:
        speech_rms = float(np.sqrt((audio**2).mean())) + 1e-9
        noise = noise_crop(noises, rng)
        noise = noise / (float(np.sqrt((noise**2).mean())) + 1e-9)
        audio = audio + noise * speech_rms / 10 ** (rng.uniform(10, 30) / 20)
    return np.clip(audio, -1, 1).astype(np.float32)


def make_silence(noises: list[Audio], rng: np.random.Generator) -> Audio:
    if rng.random() < 0.5:
        audio = noise_crop(noises, rng) * 10 ** rng.uniform(-2.5, 0.3)
    else:
        audio = synthetic_noise(rng)
    silence: Audio = np.clip(audio, -1, 1).astype(np.float32)
    return silence


# Worker-process state, set once by `init_worker`: noise recordings and the pipeline.
WORKER: dict[str, Any] = {}
# One task: clip (None synthesizes Silence), meeting-audio flag, noise split, seed.
Task = tuple[NDArray[Any] | None, bool, str, int]
Pending = tuple["AsyncResult[list[NDArray[np.float16]]]", list[int]]


def init_worker(noise: dict[str, list[Audio]], normalization: Normalization) -> None:
    WORKER["noise"] = noise
    WORKER["pipeline"] = FeaturePipeline(normalization)
    torch.set_num_threads(1)


def featurize(task: Task) -> NDArray[np.float16]:
    """Augment one clip with its own seeded generator and return its features."""
    clip, is_real, split, seed = task
    if clip is not None and clip.dtype == np.int16:
        clip = clip.astype(np.float32) / 32768
    rng = np.random.default_rng(seed)
    noises = WORKER["noise"][split]
    if clip is None:
        audio = make_silence(noises, rng)
    else:
        audio = (augment_real if is_real else augment_speech)(clip, noises, rng)
    # Half precision while queued: two epochs of float32 features need over a gigabyte.
    features: NDArray[np.float16] = WORKER["pipeline"].from_audio(audio, 16000).astype(np.float16)
    return features


def epoch_tasks(
    clips: list[Any],
    labels: list[int],
    split: str,
    rng: np.random.Generator,
    silence_count: int,
    keep: list[float],
    real: list[bool],
) -> tuple[list[Task], list[int]]:
    """One epoch's draws. `keep[i]` is the chance clip i is drawn this epoch.

    `real[i]` marks meeting audio, which gets `augment_real` instead of
    `augment_speech`. Every task carries its own seed, so the epoch does not depend
    on how many workers featurize it.
    """
    tasks: list[Task] = []
    targets = []
    for clip, label, chance, is_real in zip(clips, labels, keep, real, strict=True):
        if rng.random() > chance:
            continue
        tasks.append((clip, is_real, split, int(rng.integers(2**63))))
        targets.append(label)
    for _ in range(silence_count):
        tasks.append((None, False, split, int(rng.integers(2**63))))
        targets.append(LABELS.index("silence"))
    return tasks, targets


def start_epoch(pool: Any, tasks: list[Task], targets: list[int]) -> Pending:
    return pool.map_async(featurize, tasks, chunksize=64), targets


def finish_epoch(pending: Pending) -> tuple[torch.Tensor, torch.Tensor]:
    result, targets = pending
    return torch.from_numpy(np.stack(result.get())), torch.tensor(targets)


def train(
    root: Path,
    features_dir: Path,
    manifest_path: Path,
    output: Path,
    epochs: int,
    tts: Path,
    holdout: bool,
    ami: Path | None,
    experiment_id: str,
    unknown_cap: int | None = None,
    patience: int = 20,
    workers: int = 0,
) -> None:
    seed_everything(42, 4)
    rng = np.random.default_rng(42)
    normalization = Normalization.load(features_dir / "normalization.json")
    manifest = load_manifest(manifest_path)
    stems = noise_sources_by_split(manifest)
    noise = {split: load_noise(root, stems[split]) for split in ("train", "validation")}
    # Feature extraction runs in worker processes, one epoch ahead of training;
    # 4 cores stay with the training thread pool.
    workers = workers or max(1, (os.cpu_count() or 2) - 4)
    with Pool(workers, init_worker, (noise, normalization)) as pool:
        train_model(
            root,
            rng,
            normalization,
            stems,
            pool,
            output,
            epochs,
            tts,
            holdout,
            ami,
            experiment_id,
            unknown_cap,
            patience,
        )


def train_model(
    root: Path,
    rng: np.random.Generator,
    normalization: Normalization,
    stems: dict[str, list[str]],
    pool: Any,
    output: Path,
    epochs: int,
    tts: Path,
    holdout: bool,
    ami: Path | None,
    experiment_id: str,
    unknown_cap: int | None,
    patience: int,
) -> None:
    train_clips, train_labels = load_clips(root, "train", unknown_cap, pool)
    val_clips, val_labels = load_clips(root, "validation", unknown_cap, pool)
    tts_train, tts_train_labels, tts_val, tts_val_labels = load_tts(tts, holdout)
    unknown = LABELS.index("unknown")
    per_keyword = keyword_count(train_labels + tts_train_labels)
    val_per_keyword = keyword_count(val_labels + tts_val_labels)
    # Real unknown words are always drawn; synthetic ones are subsampled to about one
    # keyword's worth per epoch so the model cannot tell real from synthetic by domain.
    tts_unknown = max(1, tts_train_labels.count(unknown))
    tts_keep = min(1.0, per_keyword / tts_unknown)
    train_keep = [1.0] * len(train_clips) + [
        tts_keep if label == unknown else 1.0 for label in tts_train_labels
    ]
    val_keep = [1.0] * len(val_clips) + [
        0.5 if label == unknown else 1.0 for label in tts_val_labels
    ]
    train_clips += tts_train
    train_labels += tts_train_labels
    val_clips += tts_val
    val_labels += tts_val_labels
    train_real = [False] * len(train_clips)
    val_real = [False] * len(val_clips)
    if ami is not None:
        # Fluent meeting speech: real Keyword tokens and hard negatives, the
        # distribution the isolated clips never cover.
        for split, seed, clips, labels, keep, real in (
            ("train", 11, train_clips, train_labels, train_keep, train_real),
            ("dev", 12, val_clips, val_labels, val_keep, val_real),
        ):
            ami_clips, ami_labels = load_ami(ami, split, seed)
            clips += ami_clips
            labels += ami_labels
            keep += ami_keep(ami_labels)
            real += [True] * len(ami_clips)
            print(f"ami {split}: {len(ami_clips)} clips", flush=True)
    print(f"train clips {len(train_clips)}, keyword {per_keyword}, noise {stems}", flush=True)

    val_rng = np.random.default_rng(7)
    x_val, y_val = finish_epoch(
        start_epoch(
            pool,
            *epoch_tasks(
                val_clips,
                val_labels,
                "validation",
                val_rng,
                2 * val_per_keyword,
                val_keep,
                val_real,
            ),
        )
    )

    def next_epoch() -> Pending:
        return start_epoch(
            pool,
            *epoch_tasks(
                train_clips, train_labels, "train", rng, 2 * per_keyword, train_keep, train_real
            ),
        )

    model = LstmClassifier(128, 2, 0.4, n_classes=len(LABELS))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-3)
    best_loss, best_state, best_epoch, stale = float("inf"), copy.deepcopy(model.state_dict()), 0, 0
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    pending = next_epoch()
    for epoch in range(1, epochs + 1):
        x, y = finish_epoch(pending)
        if epoch < epochs:
            pending = next_epoch()
        model.train()
        loss_sum = 0.0
        for batch in torch.split(torch.randperm(len(x)), 32):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(x[batch].float()), y[batch])
            loss.backward()  # type: ignore[no-untyped-call]
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += loss.item() * len(batch)
        model.eval()
        with torch.inference_mode():
            # Batched: one pass over the full-corpus validation set needs gigabytes.
            logits = torch.cat([model(chunk.float()) for chunk in torch.split(x_val, 512)])
            val_loss = float(nn.functional.cross_entropy(logits, y_val))
            val_acc = float((logits.argmax(1) == y_val).float().mean())
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / len(x),
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        print(history[-1], flush=True)
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
            # Written at every improvement so an interrupted run keeps its best epoch.
            save_model(
                model,
                normalization,
                output,
                root,
                ami,
                experiment_id,
                unknown_cap,
                best_epoch,
                best_loss,
                started,
                history,
            )
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    save_model(
        model,
        normalization,
        output,
        root,
        ami,
        experiment_id,
        unknown_cap,
        best_epoch,
        best_loss,
        started,
        history,
    )


def save_model(
    model: LstmClassifier,
    normalization: Normalization,
    output: Path,
    root: Path,
    ami: Path | None,
    experiment_id: str,
    unknown_cap: int | None,
    best_epoch: int,
    best_loss: float,
    started: float,
    history: list[dict[str, float | int]],
) -> None:
    augmentation = "gain/noise/shift+tts-oov" + ("+ami-hard-negatives" if ami else "")
    config: dict[str, Any] = {
        "experiment_id": experiment_id,
        "augmentation": augmentation,
        "speech_root": root.name,
        "unknown_words": list(unknown_words(root)),
        "unknown_cap": unknown_cap,
        "seed": 42,
    }
    metadata = {
        "labels": list(LABELS),
        "experiment_id": experiment_id,
        "created_at": datetime.now(UTC).isoformat(),
        "code_revision": code_revision(),
        "config": config,
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "network": {
            "type": "lstm",
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.4,
            "pooling": LstmClassifier.pooling,
            "parameters": parameter_count(model),
        },
        "training": {
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "total_seconds": time.perf_counter() - started,
            "history": history,
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    sha = save_checkpoint(output / "model.pt", model, normalization, metadata)
    (output / "training.json").write_text(json.dumps(metadata, indent=2) + "\n", "utf-8")
    print(f"saved {output / 'model.pt'} sha256 {sha} best epoch {best_epoch}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--tts", type=Path, default=Path("../data/tts"))
    parser.add_argument("--holdout", action="store_true", help="keep HOLDOUT_WORDS out of training")
    parser.add_argument(
        "--ami",
        type=Path,
        help="temporal AMI split (data/temporal-ami): meeting tokens and negatives",
    )
    parser.add_argument("--experiment-id", default="LIVE1")
    parser.add_argument(
        "--unknown-cap", type=int, help="max clips per non-Keyword word and split (memory bound)"
    )
    parser.add_argument("--patience", type=int, default=20, help="early-stopping epochs")
    parser.add_argument("--workers", type=int, default=0, help="feature processes (0: cores - 4)")
    args = parser.parse_args()
    train(
        args.root,
        args.features,
        args.manifest,
        args.output,
        args.epochs,
        args.tts,
        args.holdout,
        args.ami,
        args.experiment_id,
        args.unknown_cap,
        args.patience,
        args.workers,
    )


if __name__ == "__main__":
    main()
