import io
import json
import wave
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pytest

from ml.data.ami import Clip, Corpus, Meeting, Word, annotate_clip, load_annotations, select_clips
from ml.data.ami_prepare import fetch, prepare_dataset


def annotation_fixture(path: Path) -> None:
    meetings = ET.Element("root")
    with zipfile.ZipFile(path, "w") as archive:
        for name, speaker, times in (
            ("IN1002", "person-a", (490, 500, 510, 520, 530, 540)),
            ("IN1005", "person-a", (100, 110, 120, 130, 140, 150)),
            ("IN1007", "person-b", (100, 200, 300, 700, 800, 900)),
        ):
            meeting = ET.SubElement(
                meetings, "meeting", observation=name, type="nonscenario", duration="1200"
            )
            ET.SubElement(meeting, "speaker", nxt_agent="A", global_name=speaker)
            words = ET.Element("root")
            for token, start in zip(
                ("left", "right", "up", "down", "go", "stop"), times, strict=True
            ):
                ET.SubElement(
                    words, "w", starttime=str(start), endtime=str(start + 0.25)
                ).text = token
            archive.writestr(f"words/{name}.A.words.xml", ET.tostring(words))
        archive.writestr("corpusResources/meetings.xml", ET.tostring(meetings))


def test_plan_preserves_in1002_and_covers_all_keywords_with_disjoint_global_speakers(
    tmp_path: Path,
) -> None:
    path = tmp_path / "annotations.zip"
    annotation_fixture(path)
    corpus = load_annotations(path)
    clips = select_clips(corpus)
    assert (clips[0].meeting_id, clips[0].start_s, clips[0].end_s, clips[0].split) == (
        "IN1002",
        482.14,
        1082.14,
        "test",
    )
    validation = [clip for clip in clips if clip.split == "validation"]
    assert {clip.meeting_id for clip in validation} == {"IN1007"}
    assert sum(clip.end_s - clip.start_s for clip in clips) == 1200
    assert corpus.meetings["IN1002"].speaker_ids == ("person-a",)
    assert corpus.meetings["IN1007"].speaker_ids == ("person-b",)


def test_annotation_keeps_each_keyword_and_separates_automatic_negatives_and_quiet_gaps() -> None:
    clip = Clip("test", 10, 13, "test")
    corpus = Corpus(
        {"test": Meeting("test", ("speaker",), 20, True)},
        {
            "test": (
                Word("going", 10.1, 10.5, "speaker"),
                Word("go", 11, 11.25, "speaker"),
                Word("right", 12, 12.3, "speaker"),
            )
        },
        0,
    )
    # A test signal checks label placement only. It is never a natural-speech substitute.
    audio = np.zeros(48000, dtype=np.float32)
    audio[8000:12000] = 0.2  # loud untranscribed sound must not be called quiet Silence
    events, quality = annotate_clip(corpus, clip, audio)
    keywords = [e for e in events if e.keyword in {"go", "right"}]
    assert [(e.keyword, e.start_s, e.end_s) for e in keywords] == [
        ("go", 1.0, 1.25),
        ("right", 2.0, 2.3),
    ]
    assert any(e.keyword == "unknown" for e in events)
    assert any(e.keyword == "silence" for e in events)
    assert not any(e.keyword == "silence" and e.start_s < 0.75 and e.end_s > 0.5 for e in events)
    assert quality["manual_review_performed"] is False
    assert quality["hard_negative_candidate_counts"] == {"going": 1}
    assert all(a.end_s <= b.start_s for a, b in zip(events, events[1:], strict=False))


def test_preparation_writes_validated_wavs_csv_and_honest_quality_metadata(tmp_path: Path) -> None:
    archive = tmp_path / "annotations.zip"
    annotation_fixture(archive)
    # Include ordinary unrelated speech for the validator's negative class.
    with zipfile.ZipFile(archive) as source:
        contents = {name: source.read(name) for name in source.namelist()}
    with zipfile.ZipFile(archive, "w") as target:
        for name, raw in contents.items():
            if name.endswith(".words.xml"):
                tree = ET.fromstring(raw)
                for start in (495, 105, 705):
                    ET.SubElement(
                        tree, "w", starttime=str(start), endtime=str(start + 1)
                    ).text = "going"
                raw = ET.tostring(tree)
            target.writestr(name, raw)
    audio_paths = {}
    for index, meeting_id in enumerate(("IN1002", "IN1007"), 1):
        audio = tmp_path / f"{meeting_id}.wav"
        samples = np.zeros(1200 * 16000, dtype="<i2")
        samples[::16000] = np.arange(1200) + index
        with wave.open(str(audio), "wb") as wav:
            wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            wav.writeframes(samples.tobytes())
        audio_paths[meeting_id] = audio
    isolated = tmp_path / "isolated.json"
    isolated.write_text(
        json.dumps(
            {
                "dataset_version": "fixture",
                "split_version": "fixture",
                "labels": ["left", "right", "up", "down", "go", "stop", "unknown", "silence"],
                "sources": {},
                "examples": [],
            }
        )
    )
    output = tmp_path / "prepared"
    report = prepare_dataset(archive, audio_paths, output, isolated)
    assert report["structural_validation"] == "passed"
    assert report["total_duration_s"] == 1200
    assert report["manual_review_performed"] is False
    assert report["speaker_intersection"] == []
    assert report["splits"]["test"]["keyword_counts"] == dict.fromkeys(
        ("left", "right", "up", "down", "go", "stop"), 1
    )
    assert (output / "events.csv").exists()
    assert (output / "test.csv").exists()
    assert (output / "validation.csv").exists()
    for path in (output / "audio").glob("*.wav"):
        with wave.open(str(path)) as wav:
            assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)


def test_bounded_download_verifies_ranges_and_rejects_changed_cached_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"permitted fixture source bytes"

    class Response(io.BytesIO):
        status = 206

        def __init__(self, start: int, end: int) -> None:
            super().__init__(payload[start : end + 1])
            self.headers = {
                "Content-Range": f"bytes {start}-{end}/{len(payload)}",
                "ETag": "fixture",
            }

    def urlopen(request: object, timeout: int) -> Response:
        interval = request.get_header("Range").removeprefix("bytes=")
        start, end = map(int, interval.split("-"))
        return Response(start, min(end, len(payload) - 1))

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    path = fetch("https://fixture.invalid/audio.wav", tmp_path / "source.wav", ranged=True)
    assert path.read_bytes() == payload
    assert fetch("https://fixture.invalid/audio.wav", path, ranged=True) == path
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        fetch("https://fixture.invalid/audio.wav", path, ranged=True)


@pytest.mark.parametrize("start,end", [(500.1, 500.2), (482.13, 482.15)])
def test_selection_refuses_to_drop_overlapping_or_boundary_cut_keywords(
    tmp_path: Path,
    start: float,
    end: float,
) -> None:
    path = tmp_path / "annotations.zip"
    annotation_fixture(path)
    with zipfile.ZipFile(path) as source:
        contents = {name: source.read(name) for name in source.namelist()}
    name = "words/IN1002.A.words.xml"
    root = ET.fromstring(contents[name])
    ET.SubElement(root, "w", starttime=str(start), endtime=str(end)).text = "go"
    contents[name] = ET.tostring(root)
    with zipfile.ZipFile(path, "w") as target:
        for name, raw in contents.items():
            target.writestr(name, raw)
    with pytest.raises(ValueError, match="overlapping"):
        select_clips(load_annotations(path))


def test_zero_length_source_keyword_is_preserved_as_an_unscorable_issue(tmp_path: Path) -> None:
    path = tmp_path / "annotations.zip"
    annotation_fixture(path)
    with zipfile.ZipFile(path) as source:
        contents = {name: source.read(name) for name in source.namelist()}
    name = "words/IN1002.A.words.xml"
    root = ET.fromstring(contents[name])
    ET.SubElement(root, "w", starttime="550", endtime="550").text = "go"
    contents[name] = ET.tostring(root)
    with zipfile.ZipFile(path, "w") as target:
        for name, raw in contents.items():
            target.writestr(name, raw)
    corpus = load_annotations(path)
    assert corpus.invalid_keywords == (("IN1002", "go", 550.0, 550.0),)
    assert any(w.token == "go" and w.kind == "uncertain" for w in corpus.words["IN1002"])
