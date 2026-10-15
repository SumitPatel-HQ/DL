"""Local Speech Commands subset inventory. No download or network at runtime."""

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ml.features import LABELS, read_wav

SPLITS = ("train", "validation", "test")
SPLIT_VERSION = "speaker-sha256-80-10-10-noise-source-round-robin-v1"


@dataclass(frozen=True)
class Source:
    url: str
    version: str
    license: str
    attribution: str
    consent: str
    permitted_use: str
    retention: str

    def __post_init__(self) -> None:
        if not all(isinstance(v, str) and v.strip() for v in asdict(self).values()):
            raise ValueError(
                "Source requires URL, version, license, attribution, consent and retention"
            )


@dataclass(frozen=True)
class Example:
    path: str
    label: str
    original_word: str
    speaker_id: str
    source_id: str
    source_group: str
    split: str
    sha256: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class Manifest:
    dataset_version: str
    split_version: str
    labels: tuple[str, ...]
    sources: dict[str, Source]
    examples: tuple[Example, ...]


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def speaker_split(speaker_id: str) -> str:
    bucket = int(hashlib.sha256(speaker_id.encode()).hexdigest()[:8], 16) % 100
    return "test" if bucket < 10 else "validation" if bucket < 20 else "train"


def local_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Audio path must stay within data root")
    return path


def build_manifest(
    root: Path,
    sources: dict[str, Source],
    *,
    max_per_class: int = 1000,
) -> Manifest:
    """Select a deterministic, capped subset after assigning speakers to splits.

    Unknown uses at least three unrelated word types per split. Noise segments
    never share a parent recording across splits. Noise IDs are source IDs, not
    claims that a human speaker has been identified.
    """
    if max_per_class < 1 or set(sources) != {"speech", "noise"}:
        raise ValueError("Positive class cap and speech/noise provenance required")
    grouped: dict[tuple[str, str], list[Example]] = defaultdict(list)
    for path in sorted(root.glob("*/*.wav")):
        word = path.parent.name
        if word.startswith("_"):
            continue
        match = re.fullmatch(r"(.+)_nohash_\d+", path.stem)
        if not match:
            raise ValueError(f"Missing Speech Commands speaker ID: {path.name}")
        speaker = match[1]
        split = speaker_split(speaker)
        label = word if word in LABELS[:6] else "unknown"
        relative = path.relative_to(root).as_posix()
        # Decode only selected clips below; inventory creation stays bounded.
        grouped[split, label].append(
            Example(
                relative,
                label,
                word,
                speaker,
                "speech",
                relative,
                split,
                "",
                0,
                0,
            )
        )
    selected: list[Example] = []
    for split in SPLITS:
        unknown = grouped[split, "unknown"]
        if len({e.original_word for e in unknown}) < 3:
            raise ValueError(
                f"Unknown needs at least three unrelated words in {split}; mini-8 is insufficient"
            )
        for label in LABELS[:-1]:
            candidates = grouped[split, label]
            # Round-robin Unknown word types so alphabetical order cannot remove diversity.
            words: dict[str, list[Example]] = defaultdict(list)
            for candidate in candidates:
                words[candidate.original_word].append(candidate)
            for values in words.values():
                values.sort(key=lambda e: hashlib.sha256(e.path.encode()).hexdigest())
            ordered = [
                values[i]
                for i in range(max(map(len, words.values()), default=0))
                for _, values in sorted(words.items())
                if i < len(values)
            ]
            cap = max(max_per_class, 3) if label == "unknown" else max_per_class
            for example in ordered[:cap]:
                path = local_path(root, example.path)
                audio = read_wav(path)
                selected.append(
                    Example(
                        **{
                            **asdict(example),
                            "sha256": file_sha256(path),
                            "end_s": len(audio) / 16000,
                        }
                    )
                )
    noises = sorted((root / "_background_noise_").glob("*.wav"))
    if len(noises) < 3:
        raise ValueError("Silence requires at least three independent noise source recordings")
    for i, path in enumerate(noises):
        split = SPLITS[i % 3]
        relative = path.relative_to(root).as_posix()
        audio, digest = read_wav(path), file_sha256(path)
        for start in range(min(len(audio) // 16000, max_per_class)):
            selected.append(
                Example(
                    relative,
                    "silence",
                    "_background_noise_",
                    f"noise-source:{path.stem}",
                    "noise",
                    relative,
                    split,
                    digest,
                    float(start),
                    float(start + 1),
                )
            )
    result = Manifest(
        "speech-commands-v0.02-p0-subset-v1",
        SPLIT_VERSION,
        LABELS,
        sources,
        tuple(sorted(selected, key=lambda e: (e.split, e.label, e.path, e.start_s))),
    )
    validate_manifest(result, root)
    return result


def validate_manifest(manifest: Manifest, root: Path, *, verify_audio: bool = False) -> None:
    if manifest.labels != LABELS or not manifest.dataset_version or not manifest.split_version:
        raise ValueError("Incompatible labels or missing dataset/split version")
    memberships: dict[tuple[str, str], str] = {}
    identities: set[tuple[str, float, float]] = set()
    verified: dict[str, float] = {}
    for example in manifest.examples:
        if example.split not in SPLITS or example.label not in LABELS:
            raise ValueError("Invalid split or class")
        if example.source_id not in manifest.sources or not all(
            (example.speaker_id, example.source_group, example.sha256)
        ):
            raise ValueError("Missing example provenance")
        for key in (
            ("speaker", example.speaker_id),
            ("source", example.source_group),
            ("audio", example.sha256),
        ):
            if key in memberships and memberships[key] != example.split:
                raise ValueError(f"Split leakage: {key}")
            memberships[key] = example.split
        identity = (example.path, example.start_s, example.end_s)
        if identity in identities:
            raise ValueError("Duplicate example")
        identities.add(identity)
        if not 0 <= example.start_s < example.end_s < float("inf"):
            raise ValueError("Invalid audio interval")
        path = local_path(root, example.path)
        if verify_audio:
            if example.path not in verified:
                if file_sha256(path) != example.sha256:
                    raise ValueError("Audio checksum changed")
                verified[example.path] = len(read_wav(path)) / 16000
            if example.end_s > verified[example.path] + 1e-6:
                raise ValueError("Interval exceeds audio duration")
    for split in SPLITS:
        rows = [e for e in manifest.examples if e.split == split]
        if {e.label for e in rows} != set(LABELS):
            raise ValueError(f"Missing classes in {split}")
        if len({e.original_word for e in rows if e.label == "unknown"}) < 3:
            raise ValueError(f"Unknown diversity missing in {split}")


def save_manifest(manifest: Manifest, path: Path) -> None:
    path.write_text(json.dumps(asdict(manifest), indent=2) + "\n", encoding="utf-8")


def load_manifest(path: Path) -> Manifest:
    value: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return Manifest(
        value["dataset_version"],
        value["split_version"],
        tuple(value["labels"]),
        {key: Source(**source) for key, source in value["sources"].items()},
        tuple(Example(**example) for example in value["examples"]),
    )
