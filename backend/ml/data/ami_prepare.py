"""Download selected AMI mixes, crop natural speech and export automatic labels."""

import argparse
import csv
import json
import re
import shutil
import urllib.request
import wave
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ml.data.ami import (
    ANNOTATIONS_SHA256,
    ANNOTATIONS_URL,
    AUDIO_BASE,
    KEYWORDS,
    annotate_clip,
    keyword_words,
    load_annotations,
    select_clips,
)
from ml.data.annotations import ContinuousRecording, Event, validate_dataset
from ml.data.manifest import Source, file_sha256
from ml.features import read_wav

SOURCE = Source(
    url="https://groups.inf.ed.ac.uk/ami/corpus/",
    version="AMI manual annotations 1.6.2; official headset mixes",
    license="CC-BY-4.0; https://creativecommons.org/licenses/by/4.0/",
    attribution="AMI Meeting Corpus, AMI Project; https://groups.inf.ed.ac.uk/ami/corpus/",
    consent="Publisher documents signed participant consent: "
    "https://groups.inf.ed.ac.uk/ami/corpus/ethicsandconsent.shtml . "
    "Individual signed forms are not held by this project.",
    permitted_use="Local academic keyword evaluation with CC-BY-4.0 attribution; "
    "cropped and converted copies are identified as modifications.",
    retention="Ignored local data until project completion or manual deletion; no app upload.",
)


def download_ranges(url: str, path: Path) -> None:
    """Read a large WAV through checked 4MiB HTTP ranges instead of a stalled full GET."""
    request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        match = re.fullmatch(r"bytes 0-0/(\d+)", response.headers.get("Content-Range", ""))
        if response.status != 206 or match is None or len(response.read()) != 1:
            raise ValueError("AMI audio server did not honor the byte-range request")
        length = int(match[1])
        etag = response.headers.get("ETag")
    chunk_size = 4 * 1024 * 1024
    with path.open("wb") as output:
        for start in range(0, length, chunk_size):
            end = min(start + chunk_size, length) - 1
            headers = {"Range": f"bytes={start}-{end}"}
            if etag is not None:
                headers["If-Match"] = etag
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=30) as response:
                expected_range = f"bytes {start}-{end}/{length}"
                if (
                    response.status != 206
                    or response.headers.get("Content-Range") != expected_range
                ):
                    raise ValueError("Unexpected HTTP range or changing source audio")
                payload = response.read(end - start + 2)
                if len(payload) != end - start + 1:
                    raise ValueError("Incomplete audio range")
                output.write(payload)
            if start % (8 * chunk_size) == 0:
                print(f"{path.name}: {end + 1}/{length} source bytes", flush=True)


def fetch(
    url: str, path: Path, expected_sha256: str | None = None, *, ranged: bool = False
) -> Path:
    """Cache atomically; validate pinned annotation hash or a cached audio checksum."""
    digest_path = path.with_suffix(path.suffix + ".sha256")
    if path.exists():
        expected = expected_sha256 or (
            digest_path.read_text(encoding="utf-8").strip() if digest_path.exists() else None
        )
        if expected is None or file_sha256(path) != expected:
            raise ValueError(f"Unverified or changed cached source: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    print(f"Downloading {url}", flush=True)
    if ranged:
        download_ranges(url, temporary)
    else:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
            length = response.headers.get("Content-Length")
        if length is not None and temporary.stat().st_size != int(length):
            raise ValueError("Incomplete source download")
    digest = file_sha256(temporary)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Source checksum differs from pinned annotation release")
    temporary.replace(path)
    digest_path.write_text(digest + "\n", encoding="utf-8")
    return path


def write_events(path: Path, events: list[Event]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["recording_id", "keyword", "start_s", "end_s"])
        writer.writeheader()
        writer.writerows(asdict(event) for event in events)


def prepare_dataset(
    annotations_path: Path,
    audio_paths: dict[str, Path],
    output: Path,
    isolated_manifest: Path,
) -> dict[str, Any]:
    """Offline preparation seam. Caller supplies permitted, downloaded source files."""
    corpus = load_annotations(annotations_path)
    clips = select_clips(corpus)
    output.mkdir(parents=True, exist_ok=True)
    (output / "audio").mkdir(exist_ok=True)
    recordings = []
    all_events: list[Event] = []
    provenance: dict[str, Any] = {}
    target_provenance = []
    source_hashes = {
        meeting: file_sha256(audio_paths[meeting])
        for meeting in sorted({clip.meeting_id for clip in clips})
    }
    if len(set(source_hashes.values())) != len(source_hashes):
        raise ValueError("Distinct meeting IDs have duplicate source audio")
    for meeting_id in sorted(source_hashes):
        full_audio = read_wav(audio_paths[meeting_id])
        for clip in (c for c in clips if c.meeting_id == meeting_id):
            start, end = round(clip.start_s * 16000), round(clip.end_s * 16000)
            if end > len(full_audio):
                raise ValueError("Selected interval exceeds source audio")
            # round-to-nearest PCM_s16le, preserving source gain; no normalization or splicing.
            pcm = np.rint(np.clip(full_audio[start:end], -1, 32767 / 32768) * 32768).astype("<i2")
            relative = f"audio/{clip.recording_id}.wav"
            with wave.open(str(output / relative), "wb") as wav:
                wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                wav.writeframes(pcm.tobytes())
            audio = read_wav(output / relative)
            clip_events, quality = annotate_clip(corpus, clip, audio)
            all_events.extend(clip_events)
            recording = ContinuousRecording(
                clip.recording_id,
                relative,
                corpus.meetings[meeting_id].speaker_ids,
                f"ami:{meeting_id}",
                "ami",
                clip.split,
                len(audio) / 16000,
                file_sha256(output / relative),
                True,
                "https://groups.inf.ed.ac.uk/ami/corpus/ethicsandconsent.shtml",
            )
            recordings.append(recording)
            provenance[clip.recording_id] = {
                **asdict(clip),
                "source_audio_url": f"{AUDIO_BASE}/{meeting_id}/audio/{meeting_id}.Mix-Headset.wav",
                "source_audio_sha256": source_hashes[meeting_id],
                "annotation_archive_sha256": file_sha256(annotations_path),
                "audio_modifications": "intact crop, canonical mono 16kHz, PCM_s16le",
                "quality": quality,
            }
            for word in keyword_words(corpus, clip):
                target_provenance.append(
                    {
                        "recording_id": clip.recording_id,
                        "keyword": word.token,
                        "speaker_id": word.speaker_id,
                        "source_start_s": word.start_s,
                        "source_end_s": word.end_s,
                        "truncated": word.truncated,
                        "crop_start_s": clip.start_s,
                        "annotation_method": "publisher forced alignment",
                        "manually_verified": False,
                    }
                )
    recordings.sort(key=lambda r: (r.split, r.recording_id))
    all_events.sort(key=lambda e: (e.recording_id, e.start_s, e.end_s, e.keyword))
    metadata = {
        "dataset_version": "ami-natural-p0-v1",
        "split_version": "in1002-test-global-speaker-disjoint-lexical-selection-v1",
        "sources": {"ami": asdict(SOURCE)},
        "recordings": [asdict(r) for r in recordings],
        "annotation_quality": "automatic, not manually verified",
        "provenance": provenance,
        "selection_method": (
            "pinned IN1002 test; capped lexical coverage for validation, no model scores"
        ),
    }
    (output / "ATTRIBUTION.txt").write_text(
        "AMI Meeting Corpus, AMI Project\nhttps://groups.inf.ed.ac.uk/ami/corpus/\n"
        "License: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/\n"
        "Modifications: intact crops, mono 16 kHz PCM_s16le conversion.\n"
        "Word timings: publisher forced alignment; not manually verified by this project.\n",
        encoding="utf-8",
    )
    (output / "recordings.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output / "keyword-provenance.json").write_text(
        json.dumps(target_provenance, indent=2) + "\n", encoding="utf-8"
    )
    write_events(output / "events.csv", all_events)
    split_reports: dict[str, Any] = {}
    for split in ("validation", "test"):
        rows = [r for r in recordings if r.split == split]
        ids = {r.recording_id for r in rows}
        events = [e for e in all_events if e.recording_id in ids]
        write_events(output / f"{split}.csv", events)
        counts = Counter(e.keyword for e in events)
        labeled_duration = sum(e.end_s - e.start_s for e in events)
        split_reports[split] = {
            "duration_s": sum(r.duration_s for r in rows),
            "speaker_ids": sorted({s for r in rows for s in r.speaker_ids}),
            "source_groups": sorted({r.source_group for r in rows}),
            "keyword_counts": {key: counts[key] for key in sorted(KEYWORDS)},
            "unflagged_keyword_counts": dict(
                Counter(
                    row["keyword"]
                    for row in target_provenance
                    if row["recording_id"] in ids and not row["truncated"]
                )
            ),
            "truncated_keyword_counts": dict(
                Counter(
                    row["keyword"]
                    for row in target_provenance
                    if row["recording_id"] in ids and row["truncated"]
                )
            ),
            "unknown_spans": counts["unknown"],
            "silence_spans": counts["silence"],
            "unknown_duration_s": sum(
                e.end_s - e.start_s for e in events if e.keyword == "unknown"
            ),
            "silence_duration_s": sum(
                e.end_s - e.start_s for e in events if e.keyword == "silence"
            ),
            "unlabeled_duration_s": sum(r.duration_s for r in rows) - labeled_duration,
            "hard_negative_candidate_counts": dict(
                sum(
                    (
                        Counter(
                            provenance[r.recording_id]["quality"]["hard_negative_candidate_counts"]
                        )
                        for r in rows
                    ),
                    Counter(),
                )
            ),
        }
    report = {
        "dataset_version": metadata["dataset_version"],
        "split_version": metadata["split_version"],
        "source": asdict(SOURCE),
        "structural_validation": "not run",
        "total_duration_s": sum(r.duration_s for r in recordings),
        "manual_review_performed": False,
        "scoring_ready": False,
        "speaker_intersection": sorted(
            set(split_reports["test"]["speaker_ids"])
            & set(split_reports["validation"]["speaker_ids"])
        ),
        "source_intersection": sorted(
            set(split_reports["test"]["source_groups"])
            & set(split_reports["validation"]["source_groups"])
        ),
        "splits": split_reports,
        "clips": provenance,
        "source_checksums": {"annotations": file_sha256(annotations_path), "audio": source_hashes},
        "isolated_manifest_sha256": file_sha256(isolated_manifest),
        "output_checksums": {
            name: file_sha256(output / name)
            for name in (
                "recordings.json",
                "events.csv",
                "test.csv",
                "validation.csv",
                "keyword-provenance.json",
                "ATTRIBUTION.txt",
            )
        },
        "remaining_requirements": [
            "Word boundaries are automatically aligned, not manually verified.",
            "Unknown and quiet Silence spans are inferred; acoustic correctness is unverified.",
            "Hard negatives are lexical candidates, not listening-verified acoustic confusions.",
            "AMI IDs cannot establish real-person separation from anonymous Speech Commands.",
            "Sparse keyword counts and overlapping ordinary speech limit evaluation strength.",
            "Publisher-marked truncated Keywords are flagged separately; "
            "they are uncertain labels.",
            "Unlabeled gaps must not be treated as verified negative audio.",
        ],
    }
    try:
        validate_dataset(
            output / "recordings.json", output / "events.csv", output, isolated_manifest
        )
        report["structural_validation"] = "passed"
    except ValueError as exc:
        report["structural_validation"] = "failed"
        report["validation_error"] = str(exc)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--isolated-manifest", type=Path, required=True)
    parser.add_argument(
        "--annotations", type=Path, help="Reuse an existing official annotation ZIP"
    )
    args = parser.parse_args()
    if args.annotations is not None:
        if file_sha256(args.annotations) != ANNOTATIONS_SHA256:
            raise ValueError("Annotations are not the pinned AMI 1.6.2 release")
        annotations = args.annotations
    else:
        annotations = fetch(
            ANNOTATIONS_URL, args.root / "sources/ami_public_manual_1.6.2.zip", ANNOTATIONS_SHA256
        )
    corpus = load_annotations(annotations)
    clips = select_clips(corpus)
    print("Selected intact source intervals:", [asdict(c) for c in clips], flush=True)

    def fetch_meeting(meeting_id: str) -> tuple[str, Path]:
        path = fetch(
            f"{AUDIO_BASE}/{meeting_id}/audio/{meeting_id}.Mix-Headset.wav",
            args.root / f"sources/{meeting_id}.Mix-Headset.wav",
            ranged=True,
        )
        return meeting_id, path

    with ThreadPoolExecutor(max_workers=2) as executor:
        audio_paths = dict(executor.map(fetch_meeting, sorted({clip.meeting_id for clip in clips})))
    report = prepare_dataset(annotations, audio_paths, args.root, args.isolated_manifest)
    print(
        f"Validation: {report['structural_validation']}; {report['total_duration_s']} seconds; "
        "automatic labels, no manual verification",
        flush=True,
    )
    if report["structural_validation"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
