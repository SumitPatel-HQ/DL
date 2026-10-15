"""Stream the official archive while retaining a bounded isolated-word subset."""

import argparse
import json
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

from ml.data.manifest import file_sha256, speaker_split

URL = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
WORDS = ("left", "right", "up", "down", "go", "stop", "yes", "no", "cat", "dog", "tree")


def download_subset(root: Path, per_word_split: int = 128) -> None:
    """The complete archive crosses the network, but only selected files hit disk.

    Selection uses archive order within each word/split. The downstream manifest
    records the exact file list and hashes. An interrupted run leaves partial
    data; use a new empty directory for retrying, not a mixed-version corpus.
    """
    if per_word_split < 1:
        raise ValueError("per_word_split must be positive")
    if root.exists() and any(root.iterdir()):
        raise ValueError("Download requires an empty destination")
    root.mkdir(parents=True, exist_ok=True)
    counts: Counter[tuple[str, str]] = Counter()
    with urllib.request.urlopen(URL, timeout=60) as response:
        with tarfile.open(fileobj=response, mode="r|gz") as archive:
            for member in archive:
                relative = Path(member.name)
                if not member.isfile() or relative.is_absolute() or ".." in relative.parts:
                    continue
                parts = relative.parts
                keep = relative.as_posix() in {
                    "README.md",
                    "LICENSE",
                    "LICENSE.txt",
                    "_background_noise_/README.md",
                }
                if len(parts) == 2 and parts[0] == "_background_noise_":
                    keep = relative.suffix == ".wav" or keep
                elif len(parts) == 2 and parts[0] in WORDS and relative.suffix == ".wav":
                    key = (parts[0], speaker_split(relative.stem.split("_nohash_")[0]))
                    if counts[key] < per_word_split:
                        keep = True
                        counts[key] += 1
                if keep:
                    stream = archive.extractfile(member)
                    if stream is not None:
                        destination = root / relative
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        destination.write_bytes(stream.read())
    if len(counts) != len(WORDS) * 3 or min(counts.values()) != per_word_split:
        raise ValueError("Archive did not supply the requested per-word split counts")
    evidence = ("README.md", "LICENSE", "_background_noise_/README.md")
    summary = {
        "url": URL,
        "selection_version": "archive-order-11-words-per-split-v1",
        "per_word_split": per_word_split,
        "counts": {f"{word}/{split}": n for (word, split), n in sorted(counts.items())},
        "license_evidence_sha256": {path: file_sha256(root / path) for path in evidence},
    }
    (root / "acquisition.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Downloaded {sum(counts.values())} word clips plus noise and license evidence")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--per-word-split", type=int, default=128)
    args = parser.parse_args()
    download_subset(args.output, args.per_word_split)


if __name__ == "__main__":
    main()
