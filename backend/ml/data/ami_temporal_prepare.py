"""Download selected AMI meetings and cache MFCC frames plus frame labels for temporal training."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ml.data.ami import ANNOTATIONS_SHA256, AUDIO_BASE, load_annotations
from ml.data.ami_prepare import SOURCE, fetch
from ml.data.ami_temporal import (
    AUGMENT_CLIPS_PER_STREAM,
    AUGMENT_STREAM_S,
    AUGMENT_STREAMS,
    EXCLUDED_MEETINGS,
    FRAME_CLASSES,
    KEYWORDS,
    frame_labels,
    paste_keywords,
    select_meetings,
    spans_to_labels,
)
from ml.data.manifest import file_sha256, load_manifest, local_path
from ml.features import FeaturePipeline, read_wav
from ml.features.pipeline import SAMPLE_RATE
from ml.runtime import code_revision

DATASET_VERSION = "ami-temporal-train-v1"
AUGMENT_SEED = 2027


def save_recording(path: Path, features: Any, labels: Any, **info: Any) -> dict[str, Any]:
    np.savez(path, features=features, labels=labels)
    counts = np.bincount(labels, minlength=len(FRAME_CLASSES))
    return {
        "file": path.name,
        "frames": len(labels),
        "frame_counts": dict(zip(FRAME_CLASSES, map(int, counts), strict=True)),
        "sha256": file_sha256(path),
        **info,
    }


def augmented_streams(
    meeting_audio: dict[str, Any], isolated_manifest: Path, speech_root: Path
) -> list[tuple[str, str, float, Any, list[tuple[str, float, float]]]]:
    """Seeded 120 s meeting segments with train-split Speech Commands Keywords pasted in."""
    manifest = load_manifest(isolated_manifest)
    by_keyword = {
        k: [e for e in manifest.examples if e.split == "train" and e.original_word == k]
        for k in KEYWORDS
    }
    rng = np.random.default_rng(AUGMENT_SEED)
    meetings = sorted(meeting_audio)
    streams = []
    for n in range(AUGMENT_STREAMS):
        meeting_id = meetings[n % len(meetings)]
        audio = meeting_audio[meeting_id]
        length = AUGMENT_STREAM_S * SAMPLE_RATE
        start = int(rng.integers(0, len(audio) - length))
        clips = []
        for i in range(AUGMENT_CLIPS_PER_STREAM):
            keyword = KEYWORDS[i % len(KEYWORDS)]
            example = by_keyword[keyword][int(rng.integers(len(by_keyword[keyword])))]
            clips.append((keyword, read_wav(local_path(speech_root, example.path))))
        mixed, spans = paste_keywords(audio[start : start + length], clips, rng)
        streams.append(
            (f"aug-{n:02d}-{meeting_id}-{start}", meeting_id, start / SAMPLE_RATE, mixed, spans)
        )
    return streams


def prepare(root: Path, annotations: Path, isolated_manifest: Path, speech_root: Path) -> None:
    corpus = load_annotations(annotations, include_scenario=True)
    train, dev = select_meetings(corpus)
    print(f"train {train}\ndev {dev}", flush=True)
    meetings = [*train, *dev]

    def get(meeting_id: str) -> tuple[str, Path]:
        path = fetch(
            f"{AUDIO_BASE}/{meeting_id}/audio/{meeting_id}.Mix-Headset.wav",
            root / f"sources/{meeting_id}.Mix-Headset.wav",
            ranged=True,
        )
        return meeting_id, path

    with ThreadPoolExecutor(max_workers=3) as executor:
        paths = dict(executor.map(get, meetings))
    features_dir = root / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    pipeline = FeaturePipeline()
    recordings: list[dict[str, Any]] = []
    train_audio: dict[str, Any] = {}
    for role, ids in (("train", train), ("dev", dev)):
        for meeting_id in ids:
            audio = read_wav(paths[meeting_id])
            features = pipeline.from_stream(audio, SAMPLE_RATE)
            labels = frame_labels(corpus.words[meeting_id], len(features))
            meeting = corpus.meetings[meeting_id]
            recordings.append(
                save_recording(
                    features_dir / f"{meeting_id}.npz",
                    features,
                    labels,
                    recording_id=meeting_id,
                    role=role,
                    kind="meeting",
                    speaker_ids=list(meeting.speaker_ids),
                    duration_s=len(audio) / SAMPLE_RATE,
                    source_audio_sha256=file_sha256(paths[meeting_id]),
                )
            )
            if role == "train":
                train_audio[meeting_id] = audio
            print(f"{meeting_id}: {len(features)} frames", flush=True)
    for name, meeting_id, offset_s, audio, spans in augmented_streams(
        train_audio, isolated_manifest, speech_root
    ):
        features = pipeline.from_stream(audio, SAMPLE_RATE)
        # Keep the segment's own aligned Keywords; pasted clips add to them.
        base = frame_labels(corpus.words[meeting_id], len(features), offset_s)
        recordings.append(
            save_recording(
                features_dir / f"{name}.npz",
                features,
                spans_to_labels(spans, base),
                recording_id=name,
                role="train",
                kind="augmented",
                pasted_keywords=len(spans),
                duration_s=len(audio) / SAMPLE_RATE,
            )
        )
    manifest = {
        "dataset_version": DATASET_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "code_revision": code_revision(),
        "frame_classes": list(FRAME_CLASSES),
        "excluded_meetings": list(EXCLUDED_MEETINGS),
        "annotation_archive_sha256": ANNOTATIONS_SHA256,
        "source": asdict(SOURCE),
        "isolated_manifest_sha256": file_sha256(isolated_manifest),
        "augment": {
            "seed": AUGMENT_SEED,
            "streams": AUGMENT_STREAMS,
            "stream_s": AUGMENT_STREAM_S,
            "clips_per_stream": AUGMENT_CLIPS_PER_STREAM,
            "source": "Speech Commands train split only",
        },
        "recordings": recordings,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    all_speakers = {s for r in recordings if "speaker_ids" in r for s in r["speaker_ids"]}
    forbidden = {s for m in EXCLUDED_MEETINGS for s in corpus.meetings[m].speaker_ids}
    if all_speakers & forbidden:
        raise SystemExit("Training speakers overlap validation or test speakers")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--isolated-manifest", type=Path, required=True)
    parser.add_argument("--speech-root", type=Path, required=True)
    args = parser.parse_args()
    if file_sha256(args.annotations) != ANNOTATIONS_SHA256:
        raise ValueError("Annotations are not the pinned AMI 1.6.2 release")
    prepare(args.root, args.annotations, args.isolated_manifest, args.speech_root)


if __name__ == "__main__":
    main()
