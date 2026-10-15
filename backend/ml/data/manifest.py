"""Local Speech Commands subset inventory. No download or network at runtime."""

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ml.features import LABELS, read_wav
from ml.vocabulary import EXPANDED_DATASET_VERSION, NEW_KEYWORDS, keywords_for, validate_labels

SPLITS = ("train", "validation", "test")
SPLIT_VERSION = "speaker-sha256-80-10-10-noise-source-round-robin-v1"
SPLIT_VERSION_V2 = "speaker-sha256-80-10-10-noise-sources-v2"
DATASET_VERSION_V1 = "speech-commands-v0.02-p0-subset-v1"
DATASET_VERSION_V2 = "speech-commands-v0.02-p1-noise-v1"
# Noise recordings synthesized by ml.data.noise; the P0 round-robin never uses them.
GENERATED_NOISE_STEMS = frozenset({"blue_noise", "brown_noise", "violet_noise"})
# V2 keeps every P0 source in its P0 split (so the E02 checkpoint never trains on a
# test noise source) and adds one generated source per split: three per split.
NOISE_ASSIGNMENT_V2 = {
    "doing_the_dishes": "train",
    "pink_noise": "train",
    "blue_noise": "train",
    "dude_miaowing": "validation",
    "running_tap": "validation",
    "brown_noise": "validation",
    "exercise_bike": "test",
    "white_noise": "test",
    "violet_noise": "test",
}


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


def noise_assignment(root: Path, split_version: str) -> dict[Path, str]:
    """Map each noise recording to its split under the given split version."""
    noises = sorted((root / "_background_noise_").glob("*.wav"))
    if split_version == SPLIT_VERSION_V2:
        unassigned = {p.stem for p in noises} - set(NOISE_ASSIGNMENT_V2)
        if unassigned:
            raise ValueError(f"Noise sources without a V2 split: {sorted(unassigned)}")
        return {p: NOISE_ASSIGNMENT_V2[p.stem] for p in noises if p.stem in NOISE_ASSIGNMENT_V2}
    original = [p for p in noises if p.stem not in GENERATED_NOISE_STEMS]
    return {p: SPLITS[i % 3] for i, p in enumerate(original)}


NOISE_SOURCE_PREFIX = "noise-source:"


def noise_source_name(speaker_id: str) -> str:
    """Noise recording stem from a noise example's speaker_id; '' for speech."""
    return (
        speaker_id.removeprefix(NOISE_SOURCE_PREFIX)
        if speaker_id.startswith(NOISE_SOURCE_PREFIX)
        else ""
    )


def noise_sources_by_split(manifest: "Manifest") -> dict[str, list[str]]:
    """Sorted noise source stems of every split, read from the Silence examples."""
    found: dict[str, set[str]] = {split: set() for split in SPLITS}
    for example in manifest.examples:
        if example.source_id == "noise":
            found[example.split].add(noise_source_name(example.speaker_id))
    return {split: sorted(names) for split, names in found.items()}


def build_manifest(
    root: Path,
    sources: dict[str, Source],
    *,
    max_per_class: int = 1000,
    dataset_version: str = DATASET_VERSION_V1,
    split_version: str = SPLIT_VERSION,
    labels: tuple[str, ...] = LABELS,
) -> Manifest:
    """Select a deterministic, capped subset after assigning speakers to splits.

    Speech Commands groups are speaker identities, not independent recording
    sessions. The speaker ID is the speech leakage gate. Noise segments
    never share a parent recording across splits. Noise IDs are source IDs, not
    claims that a human speaker has been identified.
    """
    keywords = keywords_for(labels)
    new_words = set(keywords) - set(LABELS[:-2])
    if max_per_class < 1 or set(sources) != {"speech", "noise", *new_words}:
        raise ValueError("Positive class cap and speech/noise provenance required")
    if new_words:
        if dataset_version == DATASET_VERSION_V1:
            dataset_version = EXPANDED_DATASET_VERSION
        if dataset_version == DATASET_VERSION_V2:
            raise ValueError("Expanded vocabulary requires a bumped dataset version")
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
        label = word if word in keywords else "unknown"
        relative = path.relative_to(root).as_posix()
        # Decode only selected clips below; inventory creation stays bounded.
        grouped[split, label].append(
            Example(
                relative,
                label,
                word,
                speaker,
                word if word in new_words else "speech",
                f"speaker:{speaker}",
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
        for label in labels[:-1]:
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
    assignment = noise_assignment(root, split_version)
    if len(assignment) < 3:
        raise ValueError("Silence requires at least three independent noise source recordings")
    for path, split in assignment.items():
        relative = path.relative_to(root).as_posix()
        audio, digest = read_wav(path), file_sha256(path)
        for start in range(min(len(audio) // 16000, max_per_class)):
            selected.append(
                Example(
                    relative,
                    "silence",
                    "_background_noise_",
                    f"{NOISE_SOURCE_PREFIX}{path.stem}",
                    "noise",
                    relative,
                    split,
                    digest,
                    float(start),
                    float(start + 1),
                )
            )
    result = Manifest(
        dataset_version,
        split_version,
        labels,
        sources,
        tuple(sorted(selected, key=lambda e: (e.split, e.label, e.path, e.start_s))),
    )
    validate_manifest(result, root)
    return result


def validate_manifest(manifest: Manifest, root: Path, *, verify_audio: bool = False) -> None:
    validate_labels(manifest.labels)
    if not manifest.dataset_version or not manifest.split_version:
        raise ValueError("Incompatible labels or missing dataset/split version")
    if manifest.labels != LABELS and manifest.dataset_version in (
        DATASET_VERSION_V1,
        DATASET_VERSION_V2,
    ):
        raise ValueError("Expanded vocabulary requires a bumped dataset version")
    memberships: dict[tuple[str, str], str] = {}
    identities: set[tuple[str, float, float]] = set()
    verified: dict[str, float] = {}
    for example in manifest.examples:
        if example.split not in SPLITS or example.label not in manifest.labels:
            raise ValueError("Invalid split or class")
        if example.label == "unknown" and example.original_word in keywords_for(manifest.labels):
            raise ValueError("Unknown must exclude trained Keywords")
        if manifest.labels != LABELS and example.label in NEW_KEYWORDS:
            if example.source_id != example.label or example.original_word != example.label:
                raise ValueError("New Keywords require word-specific source provenance")
        if example.source_id not in manifest.sources or not all(
            (example.speaker_id, example.source_group, example.sha256)
        ):
            raise ValueError("Missing example provenance")
        # Speech Commands has no separate session/source boundary to infer from
        # its per-word files. Legacy per-file speech groups are provenance only.
        keys = [
            ("speaker", example.speaker_id),
            ("audio", example.sha256),
        ]
        if example.source_id != "speech":
            if example.source_id == "noise" and example.source_group != example.path:
                raise ValueError("Noise source_group must identify the parent recording path")
            keys.append(("source", example.source_group))
        for key in keys:
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
        if {e.label for e in rows} != set(manifest.labels):
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
