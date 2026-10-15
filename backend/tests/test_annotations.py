from dataclasses import replace
from pathlib import Path

import pytest

from ml.data.annotations import ContinuousRecording, read_events, validate_annotations


def recordings() -> tuple[ContinuousRecording, ...]:
    return tuple(
        ContinuousRecording(
            recording_id=split,
            path=f"{split}.wav",
            speaker_ids=(f"local:{split}",),
            source_group=f"session:{split}",
            source_id="local-consent",
            split=split,
            duration_s=300,
            sha256=("a" if split == "validation" else "b") * 64,
            natural=True,
            consent_reference="consent/local.txt",
        )
        for split in ("validation", "test")
    )


def csv_file(tmp_path: Path) -> Path:
    path = tmp_path / "events.csv"
    rows = ["recording_id,keyword,start_s,end_s"]
    for split in ("validation", "test"):
        for i, keyword in enumerate(("left", "right", "up", "down", "go", "stop"), 1):
            rows.append(f"{split},{keyword},{i},{i + 0.5}")
        rows += [f"{split},unknown,10,20", f"{split},silence,20,30"]
    path.write_text("\n".join(rows) + "\n")
    return path


def test_natural_event_set_requires_duration_negatives_and_disjoint_speakers(
    tmp_path: Path,
) -> None:
    events = read_events(csv_file(tmp_path))
    assert validate_annotations(recordings(), events) == 600
    with pytest.raises(ValueError, match="speaker"):
        validate_annotations(recordings(), events, isolated_speakers={"local:test"})
    with pytest.raises(ValueError, match="natural"):
        validate_annotations(tuple(replace(r, natural=False) for r in recordings()), events)
    with pytest.raises(ValueError, match="600"):
        validate_annotations(tuple(replace(r, duration_s=100) for r in recordings()), events)


def test_annotation_rejects_bad_intervals_overlaps_and_unknown_recordings(tmp_path: Path) -> None:
    events = read_events(csv_file(tmp_path))
    for event in (
        replace(events[0], end_s=301),
        replace(events[0], start_s=-1),
        replace(events[0], recording_id="missing"),
    ):
        with pytest.raises(ValueError):
            validate_annotations(recordings(), (event, *events[1:]))
    with pytest.raises(ValueError, match="overlap"):
        validate_annotations(recordings(), (*events, replace(events[0], start_s=1.1)))
