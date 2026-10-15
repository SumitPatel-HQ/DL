"""Prepare real AMI meeting crops with explicitly automatic event annotations."""

import math
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ml.data.annotations import Event
from ml.features import LABELS
from ml.features.pipeline import Audio

KEYWORDS = frozenset(LABELS[:6])
ANNOTATIONS_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
ANNOTATIONS_SHA256 = "b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d"
AUDIO_BASE = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"


@dataclass(frozen=True)
class Word:
    token: str
    start_s: float
    end_s: float
    speaker_id: str
    kind: str = "w"
    truncated: bool = False


@dataclass(frozen=True)
class Meeting:
    meeting_id: str
    speaker_ids: tuple[str, ...]
    duration_s: float
    natural: bool


@dataclass(frozen=True)
class Corpus:
    meetings: dict[str, Meeting]
    words: dict[str, tuple[Word, ...]]
    invalid_timing_count: int
    invalid_keywords: tuple[tuple[str, str, float | None, float | None], ...] = ()


@dataclass(frozen=True)
class Clip:
    meeting_id: str
    start_s: float
    end_s: float
    split: str

    @property
    def recording_id(self) -> str:
        return f"ami-{self.meeting_id}-{round(self.start_s * 16000)}-{round(self.end_s * 16000)}"


def load_annotations(path: Path) -> Corpus:
    """Read global identities and all timed lexical/nonverbal events, not XML gaps."""
    meetings: dict[str, Meeting] = {}
    speakers: dict[tuple[str, str], str] = {}
    words: dict[str, list[Word]] = defaultdict(list)
    invalid = 0
    invalid_keywords: list[tuple[str, str, float | None, float | None]] = []
    with zipfile.ZipFile(path) as archive:
        for row in ET.fromstring(archive.read("corpusResources/meetings.xml")):
            meeting_id = row.attrib["observation"]
            identities = []
            for speaker in row:
                identity = speaker.attrib["global_name"]
                speakers[meeting_id, speaker.attrib["nxt_agent"]] = identity
                identities.append(identity)
            # IB non-scenario meetings include elicited fictitious tasks. Exclude them.
            natural = row.get("type") == "nonscenario" and meeting_id.startswith(("IN", "EN"))
            meetings[meeting_id] = Meeting(
                meeting_id, tuple(sorted(set(identities))), float(row.attrib["duration"]), natural
            )
        for name in sorted(archive.namelist()):
            if not name.startswith("words/") or not name.endswith(".words.xml"):
                continue
            meeting_id, agent, _, _ = Path(name).name.split(".")
            if not meetings[meeting_id].natural:
                continue
            elements = ET.fromstring(archive.read(name))
            if len(elements) == 0:
                continue
            identity = speakers[meeting_id, agent]
            for element in elements:
                if element.tag == "gap":
                    continue
                token = (element.text or "").strip().casefold()
                if element.get("starttime") is None or element.get("endtime") is None:
                    if element.tag == "w" and token in KEYWORDS:
                        invalid_keywords.append((meeting_id, token, None, None))
                    continue
                start, end = float(element.attrib["starttime"]), float(element.attrib["endtime"])
                if not 0 <= start < end < math.inf:
                    if element.get("punc") != "true":
                        invalid += 1
                    if element.tag == "w" and token in KEYWORDS:
                        invalid_keywords.append((meeting_id, token, start, end))
                    # A zero-length token is not a timed Keyword, but its neighborhood
                    # must not be inferred to be quiet non-speech.
                    if math.isfinite(start) and start >= 0:
                        words[meeting_id].append(
                            Word(token, max(0, start - 0.1), start + 0.1, identity, "uncertain")
                        )
                    continue
                words[meeting_id].append(
                    Word(token, start, end, identity, element.tag, element.get("trunc") == "true")
                )
    return Corpus(
        meetings,
        {
            key: tuple(sorted(value, key=lambda w: (w.start_s, w.end_s, w.speaker_id, w.token)))
            for key, value in words.items()
        },
        invalid,
        tuple(invalid_keywords),
    )


def keyword_words(corpus: Corpus, clip: Clip) -> tuple[Word, ...]:
    return tuple(
        w
        for w in corpus.words[clip.meeting_id]
        if w.kind == "w"
        and w.token in KEYWORDS
        and clip.start_s <= w.start_s < w.end_s <= clip.end_s
    )


def usable_clip(corpus: Corpus, clip: Clip) -> bool:
    targets = [w for w in corpus.words[clip.meeting_id] if w.kind == "w" and w.token in KEYWORDS]
    if any(w.start_s < edge < w.end_s for w in targets for edge in (clip.start_s, clip.end_s)):
        return False
    selected = keyword_words(corpus, clip)
    # Preserve all events. Do not silently drop overlapping Keywords to pass the CSV validator.
    return not any(a.end_s > b.start_s for a, b in zip(selected, selected[1:], strict=False))


def select_clips(corpus: Corpus) -> tuple[Clip, ...]:
    """Freeze the researched test crop and select 2 disjoint 5-minute validation crops.

    Optimize capped lexical coverage, never model predictions. Prefer a single
    additional natural meeting to limit downloads. Tie-break by meeting and times.
    """
    anchor = Clip("IN1002", 482.14, 1082.14, "test")
    if "IN1002" not in corpus.meetings or not corpus.meetings["IN1002"].natural:
        raise ValueError("The researched natural IN1002 meeting is missing")
    if {w.token for w in keyword_words(corpus, anchor)} != KEYWORDS or not usable_clip(
        corpus, anchor
    ):
        raise ValueError("IN1002 candidate has missing or overlapping Keywords")
    forbidden = set(corpus.meetings["IN1002"].speaker_ids)
    choices: list[tuple[tuple[int, int, int], str, float, float]] = []
    for meeting_id, meeting in sorted(corpus.meetings.items()):
        if (
            not meeting.natural
            or forbidden.intersection(meeting.speaker_ids)
            or meeting.duration_s < 600
        ):
            continue
        targets = [
            w for w in corpus.words.get(meeting_id, ()) if w.kind == "w" and w.token in KEYWORDS
        ]
        starts = sorted(
            {round(max(0, min(w.start_s - 150, meeting.duration_s - 300)), 5) for w in targets}
        )
        candidates = []
        for start in starts:
            clip = Clip(meeting_id, start, start + 300, "validation")
            if usable_clip(corpus, clip):
                candidates.append((clip, Counter(w.token for w in keyword_words(corpus, clip))))
        for first, left in candidates:
            for second, right in candidates:
                if first.end_s > second.start_s:
                    continue
                counts = left + right
                if set(counts) == KEYWORDS:
                    score = (
                        min(counts.values()),
                        sum(min(v, 5) for v in counts.values()),
                        sum(min(v, 10) for v in counts.values()),
                    )
                    choices.append((score, meeting_id, first.start_s, second.start_s))
    if not choices:
        raise ValueError("No speaker-disjoint natural validation selection covers all six Keywords")
    # Highest coverage first; stable ascending meeting/time tie-break.
    _, meeting_id, first_start, second_start = min(
        choices, key=lambda x: (tuple(-v for v in x[0]), *x[1:])
    )
    return (
        anchor,
        Clip(meeting_id, first_start, first_start + 300, "validation"),
        Clip(meeting_id, second_start, second_start + 300, "validation"),
    )


HARD_NEGATIVE_TOKENS = frozenset(
    {
        "going",
        "gone",
        "good",
        "got",
        "goal",
        "glow",
        "slow",
        "stock",
        "stopped",
        "stopping",
        "write",
        "white",
        "bright",
        "upright",
        "update",
        "download",
        "downloading",
        "leftover",
    }
)


def merged_intervals(
    intervals: list[tuple[float, float]], gap: float = 0
) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1] + gap:
            result[-1] = result[-1][0], max(end, result[-1][1])
        else:
            result.append((start, end))
    return result


def subtract_intervals(
    intervals: list[tuple[float, float]],
    blocked: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    result = []
    for start, end in intervals:
        cursor = start
        for left, right in blocked:
            if right <= cursor:
                continue
            if left >= end:
                break
            if left > cursor:
                result.append((cursor, min(left, end)))
            cursor = max(cursor, right)
        if cursor < end:
            result.append((cursor, end))
    return result


def annotate_clip(
    corpus: Corpus, clip: Clip, audio: Audio
) -> tuple[tuple[Event, ...], dict[str, Any]]:
    """Keep aligned Keywords, infer Unknown spans and conservative quiet candidates.

    A transcript gap alone is insufficient evidence for Silence. Require no
    annotated lexical/nonverbal activity, 100ms guards and whole-gap RMS <= 0.01
    for at least 0.5s. Silence includes quiet background noise, not just zeroes.
    These remain unreviewed labels, not a speech-absence guarantee.
    """
    duration = len(audio) / 16000
    if abs(duration - (clip.end_s - clip.start_s)) > 1 / 16000:
        raise ValueError("Clip audio duration differs from selected source interval")
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError("Expected finite canonical mono audio")
    words = [
        w
        for w in corpus.words[clip.meeting_id]
        if w.end_s > clip.start_s and w.start_s < clip.end_s
    ]
    keywords = keyword_words(corpus, clip)
    if not usable_clip(corpus, clip):
        raise ValueError("Cannot export a partial or overlapping Keyword")
    events = [
        Event(
            clip.recording_id,
            w.token,
            round(w.start_s - clip.start_s, 6),
            round(w.end_s - clip.start_s, 6),
        )
        for w in keywords
    ]

    # Quantize all inferred bounds to samples so CSV rows and actual WAV agree.
    def bounds(start: float, end: float, guard: float = 0) -> tuple[float, float]:
        return (
            max(0, round((start - clip.start_s - guard) * 16000)) / 16000,
            min(len(audio), round((end - clip.start_s + guard) * 16000)) / 16000,
        )

    protected_words = [
        *keywords,
        *(w for w in words if w.kind == "uncertain" and w.token in KEYWORDS),
    ]
    protected = merged_intervals([bounds(w.start_s, w.end_s, 0.05) for w in protected_words])
    unknown = merged_intervals(
        [bounds(w.start_s, w.end_s) for w in words if w.kind == "w" and w.token not in KEYWORDS],
        gap=0.1,
    )
    unknown = subtract_intervals(unknown, protected)
    events += [
        Event(clip.recording_id, "unknown", start, end)
        for start, end in unknown
        if end - start >= 0.05
    ]
    occupied = merged_intervals([bounds(w.start_s, w.end_s, 0.1) for w in words])
    for start, end in subtract_intervals([(0, duration)], occupied):
        if end - start < 0.5:
            continue
        segment = audio[round(start * 16000) : round(end * 16000)].astype(np.float64)
        if float(np.sqrt(np.mean(segment**2))) <= 0.01:
            events.append(Event(clip.recording_id, "silence", round(start, 6), round(end, 6)))
    quality = {
        "manual_review_performed": False,
        "word_timestamps": "AMI forced alignment, not manually verified",
        "unknown_method": "unrelated aligned lexical intervals minus guarded Keywords",
        "silence_method": "all-speaker annotation gaps plus quiet RMS; inferred, not verified",
        "quiet_rms_threshold": 0.01,
        "hard_negative_candidate_counts": dict(
            Counter(w.token for w in words if w.kind == "w" and w.token in HARD_NEGATIVE_TOKENS)
        ),
        "truncated_keyword_count": sum(w.truncated for w in keywords),
        "invalid_keyword_timings": [
            {"keyword": token, "source_start_s": start, "source_end_s": end}
            for meeting, token, start, end in corpus.invalid_keywords
            if meeting == clip.meeting_id and (start is None or clip.start_s <= start < clip.end_s)
        ],
        "adjacent_annotation_overlap_count": sum(
            a.end_s > b.start_s for a, b in zip(words, words[1:], strict=False)
        ),
    }
    return tuple(sorted(events, key=lambda e: (e.start_s, e.end_s, e.keyword))), quality
