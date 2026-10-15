"""Validate human annotations. This module does not create speech or labels."""

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from ml.data.manifest import Source, file_sha256, load_manifest, local_path
from ml.features import LABELS, read_wav


@dataclass(frozen=True)
class ContinuousRecording:
    recording_id: str
    path: str
    speaker_ids: tuple[str, ...]
    source_group: str
    source_id: str
    split: str
    duration_s: float
    sha256: str
    natural: bool
    consent_reference: str


@dataclass(frozen=True)
class Event:
    recording_id: str
    keyword: str
    start_s: float
    end_s: float


def read_events(path: Path) -> tuple[Event, ...]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["recording_id", "keyword", "start_s", "end_s"]:
            raise ValueError("CSV header must be recording_id,keyword,start_s,end_s")
        result = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Malformed annotation CSV row")
            result.append(
                Event(
                    row["recording_id"], row["keyword"], float(row["start_s"]), float(row["end_s"])
                )
            )
        return tuple(result)


def validate_annotations(
    recordings: tuple[ContinuousRecording, ...],
    events: tuple[Event, ...],
    *,
    isolated_speakers: set[str] | None = None,
    isolated_sources: set[str] | None = None,
    isolated_hashes: set[str] | None = None,
) -> float:
    by_id = {r.recording_id: r for r in recordings}
    if len(by_id) != len(recordings):
        raise ValueError("Duplicate recording ID")
    total = sum(r.duration_s for r in recordings)
    if not 600 <= total <= 1200:
        raise ValueError("Natural continuous set requires 600–1200 seconds")
    if {r.split for r in recordings} != {"validation", "test"}:
        raise ValueError("Continuous set requires separate validation and test recordings")
    membership: dict[tuple[str, str], str] = {}
    paths: set[str] = set()
    hashes: set[str] = set()
    for recording in recordings:
        if recording.natural is not True:
            raise ValueError("Final evaluation requires natural speech; synthetic is debug-only")
        if not math.isfinite(recording.duration_s) or not 0 < recording.duration_s <= 600:
            raise ValueError("Recording duration must be positive and at most 600 seconds")
        if not recording.speaker_ids or not all(recording.speaker_ids):
            raise ValueError("Every recording needs speaker IDs")
        if (
            not all(
                (
                    recording.recording_id,
                    recording.source_group,
                    recording.source_id,
                    recording.consent_reference,
                )
            )
            or len(recording.sha256) != 64
        ):
            raise ValueError("Recording provenance and consent required")
        if recording.path in paths:
            raise ValueError("Duplicate recording path")
        paths.add(recording.path)
        if recording.sha256 in hashes or recording.sha256 in (isolated_hashes or set()):
            raise ValueError("Duplicate recording audio or overlap with isolated dataset")
        hashes.add(recording.sha256)
        if set(recording.speaker_ids) & (isolated_speakers or set()):
            raise ValueError("Continuous speaker overlaps isolated dataset")
        if recording.source_group in (isolated_sources or set()):
            raise ValueError("Continuous source overlaps isolated dataset")
        keys = [("speaker", speaker) for speaker in recording.speaker_ids]
        keys += [("source", recording.source_group), ("audio", recording.sha256)]
        for key in keys:
            if key in membership and membership[key] != recording.split:
                raise ValueError("Continuous speaker/source leakage across splits")
            membership[key] = recording.split
    grouped: dict[str, list[Event]] = defaultdict(list)
    for event in events:
        if event.recording_id not in by_id or event.keyword not in LABELS:
            raise ValueError("Unknown recording ID or unsupported event keyword")
        if not 0 <= event.start_s < event.end_s <= by_id[event.recording_id].duration_s:
            raise ValueError("Event interval must be finite, positive and inside recording")
        grouped[event.recording_id].append(event)
    for recording in recordings:
        rows = sorted(grouped[recording.recording_id], key=lambda e: e.start_s)
        if not rows:
            raise ValueError("Every recording needs reviewed annotations")
        if any(a.end_s > b.start_s for a, b in zip(rows, rows[1:], strict=False)):
            raise ValueError("Annotations must not overlap")
    for split in ("validation", "test"):
        labels = {e.keyword for e in events if by_id[e.recording_id].split == split}
        if labels != set(LABELS):
            raise ValueError(f"{split} requires Keywords, hard negatives and Silence/noise")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", required=True, type=Path)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--isolated-manifest", required=True, type=Path)
    args = parser.parse_args()
    metadata = json.loads(args.recordings.read_text(encoding="utf-8"))
    if not metadata.get("dataset_version") or not metadata.get("split_version"):
        raise ValueError("Continuous dataset and split versions required")
    sources = {key: Source(**value) for key, value in metadata["sources"].items()}
    recordings = tuple(
        ContinuousRecording(**{**row, "speaker_ids": tuple(row["speaker_ids"])})
        for row in metadata["recordings"]
    )
    isolated = load_manifest(args.isolated_manifest)
    total = validate_annotations(
        recordings,
        read_events(args.events),
        isolated_speakers={e.speaker_id for e in isolated.examples},
        isolated_sources={e.source_group for e in isolated.examples},
        isolated_hashes={e.sha256 for e in isolated.examples},
    )
    for recording in recordings:
        if recording.source_id not in sources:
            raise ValueError("Missing recording source provenance")
        path = local_path(args.root, recording.path)
        if file_sha256(path) != recording.sha256:
            raise ValueError("Recording checksum mismatch")
        if abs(len(read_wav(path)) / 16000 - recording.duration_s) > 1 / 16000:
            raise ValueError("Recording duration mismatch")
    print(f"Validated {len(recordings)} natural recordings, {total:.1f} seconds")


if __name__ == "__main__":
    main()
