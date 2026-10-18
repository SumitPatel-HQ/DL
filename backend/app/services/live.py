"""Live voice session policy: allowlisted in-app actions, smoothing and cooldown.

Mic audio lives only in a bounded in-memory tail of about one window. Decisions
are stored as metadata (label, Confidence, decision, action, action result);
raw mic audio is never written anywhere. Per ADR-0003 and ADR-0004.
"""

import sqlite3
from collections import deque
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.services.recordings import RecordingService
from ml.features import LABELS
from ml.features.pipeline import SAMPLE_RATE, WINDOW_SAMPLES, canonical_audio
from ml.spotting import WINDOW_S, peak_margin

PROTOCOL_VERSION = 1
ENCODING = "pcm_s16le"
MIN_SAMPLE_RATE, MAX_SAMPLE_RATE = 8000, 96000
ACTION_STATUSES = ("applied", "no_change")


@dataclass(frozen=True)
class KeywordAction:
    keyword: str
    action: str
    label: str


# The only actions a Keyword can trigger. Each maps to one in-app demo function.
# Voice "stop" triggers the pause action only; it never ends the session.
# Only the Stop button (control message), close, or disconnect ends the session.
KEYWORD_ACTIONS = (
    KeywordAction("right", "next_slide", "Next slide"),
    KeywordAction("left", "previous_slide", "Previous slide"),
    KeywordAction("go", "play", "Play"),
    KeywordAction("stop", "pause", "Pause"),
    KeywordAction("up", "volume_up", "Volume up"),
    KeywordAction("down", "volume_down", "Volume down"),
)
ACTION_BY_KEYWORD = {mapping.keyword: mapping.action for mapping in KEYWORD_ACTIONS}


@dataclass(frozen=True)
class LivePolicy:
    """Live operating point.

    `margin`: a Keyword must beat the runner-up class (any other label) by this much,
    the same rule `ml.spotting.detect` applies to Recordings.
    `smooth_windows`: window scores are averaged over this many hops before argmax.
    `rms_gate_dbfs`: windows quieter than this are Silence without running the model.
    Live audio has no whole-Recording floor to measure against, so this gate is an
    absolute level, unlike the Recording detector's floor-relative `min_level_db`.

    A 0.1 s hop with no smoothing keeps short Keywords ("up", "go") whole in more
    windows; averaging with a neighbouring window halved their peak. On replayed
    Speech Commands test clips this took hits from 214 to 241 of 300.
    """

    hop_s: float = 0.1
    threshold: float = 0.7
    margin: float = 0.3
    smooth_windows: int = 1
    rms_gate_dbfs: float = -50.0
    min_consecutive: int = 2
    cooldown_s: float = 1.2
    max_lag_s: float = 1.0
    max_frame_bytes: int = 32768
    window_s: float = field(default=WINDOW_S, init=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Decision:
    t_s: float
    label: str
    score: float
    accepted: bool
    reason: str
    action: str | None


class LiveDecider:
    """Turn per-window scores into at most one action per utterance.

    Consecutive windows with the same top label form one run. A Keyword run is
    accepted once `min_consecutive` windows in a row reach the threshold and the
    cooldown since the last accepted action for the same Keyword has expired. A
    different Keyword is a new utterance, so "right" then "go" 1 s apart both act.
    A window only counts toward the streak when its Keyword also beats the
    runner-up class by `margin`. Scores are averaged over `smooth_windows` hops
    first. A run that never gets there is rejected as low Confidence when it
    ends. Unknown and Silence runs are rejected once, as soon as they start.
    Later windows of a decided run are ignored, so one utterance can cause at
    most one action.
    """

    def __init__(self, policy: LivePolicy, labels: Sequence[str] = LABELS) -> None:
        self.policy = policy
        self.labels = tuple(labels)
        self.last_accept: tuple[float, str] | None = None
        self.history: deque[NDArray[np.float32]] = deque(maxlen=max(1, policy.smooth_windows))
        self.reset()

    def reset(self, keep_history: bool = False) -> None:
        """End the current run without a decision. Cooldown is kept."""
        if not keep_history:
            self.history.clear()
        self.run_label: str | None = None
        self.decided = False
        self.streak = 0
        self.best = 0.0
        self.last_s = 0.0

    def observe(self, t_s: float, scores: Sequence[float] | NDArray[np.float32]) -> list[Decision]:
        self.history.append(np.asarray(scores, dtype=np.float32))
        smoothed = np.mean(self.history, axis=0)
        index = int(np.argmax(smoothed))
        label, score = self.labels[index], float(smoothed[index])
        decisions = []
        if label != self.run_label:
            if self.run_label in ACTION_BY_KEYWORD and not self.decided:
                decisions.append(
                    Decision(self.last_s, self.run_label, self.best, False, "low_confidence", None)
                )
            self.reset(keep_history=True)
            self.run_label = label
        self.last_s = t_s
        self.best = max(self.best, score)
        if self.decided:
            return decisions
        if label not in ACTION_BY_KEYWORD:
            self.decided = True
            decisions.append(Decision(t_s, label, score, False, label, None))
            return decisions
        confident = (
            score >= self.policy.threshold and float(peak_margin(smoothed)) >= self.policy.margin
        )
        self.streak = self.streak + 1 if confident else 0
        if self.streak >= self.policy.min_consecutive:
            self.decided = True
            last = self.last_accept
            if last is not None and last[1] == label and t_s - last[0] < self.policy.cooldown_s:
                decisions.append(Decision(t_s, label, self.best, False, "cooldown", None))
            else:
                self.last_accept = (t_s, label)
                decisions.append(
                    Decision(t_s, label, self.best, True, "accepted", ACTION_BY_KEYWORD[label])
                )
        return decisions


def window_rms(audio: NDArray[np.float32]) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if audio.size else 0.0


@dataclass(frozen=True)
class WindowBatch:
    windows: list[NDArray[np.float32]]
    ends_s: list[float]
    dropped_s: float


class LiveStream:
    """Bounded mic tail at the browser's native rate, cut into 16 kHz windows."""

    def __init__(self, sample_rate: int, policy: LivePolicy) -> None:
        self.rate = sample_rate
        self.policy = policy
        self.window = round(sample_rate * policy.window_s)
        self.hop = round(sample_rate * policy.hop_s)
        self.samples = np.zeros(0, dtype=np.float32)
        self.offset = 0
        self.received = 0
        self.next_end = self.window

    def push(self, frame: bytes) -> None:
        if not frame or len(frame) % 2 or len(frame) > self.policy.max_frame_bytes:
            raise ValueError("Frames must be non-empty mono PCM_s16le within the byte limit")
        pcm = np.frombuffer(frame, dtype="<i2").astype(np.float32) / 32768
        self.samples = np.concatenate([self.samples, pcm])
        self.received += len(pcm)

    def take_windows(self) -> WindowBatch:
        """Every complete window since the last call, skipping audio that is too stale."""
        dropped = 0
        lag = self.received - self.next_end
        if lag > round(self.rate * self.policy.max_lag_s):
            dropped = lag // self.hop * self.hop
            self.next_end += dropped
        windows, ends = [], []
        while self.next_end <= self.received:
            first = self.next_end - self.window - self.offset
            audio = canonical_audio(self.samples[first : first + self.window], self.rate)
            windows.append(np.pad(audio[:WINDOW_SAMPLES], (0, max(0, WINDOW_SAMPLES - len(audio)))))
            ends.append(self.next_end / self.rate)
            self.next_end += self.hop
        keep = self.next_end - self.window - self.offset
        if keep > 0:
            self.samples = self.samples[keep:]
            self.offset += keep
        return WindowBatch(windows, ends, dropped / self.rate)

    def retained_seconds(self) -> float:
        return len(self.samples) / self.rate

    def discard(self) -> None:
        """Require a fresh complete window after a permission/test boundary."""
        self.samples = np.zeros(0, dtype=np.float32)
        self.offset = self.received
        self.next_end = self.received + self.window


class LiveService:
    """One live session at a time over the active model. Stores metadata, never audio."""

    def __init__(self, recordings: RecordingService, policy: LivePolicy | None = None) -> None:
        self.recordings = recordings
        self.policy = policy or LivePolicy()
        self.active = False

    def action_map(self) -> dict[str, Any]:
        return {
            "version": PROTOCOL_VERSION,
            "actions": [asdict(mapping) for mapping in KEYWORD_ACTIONS],
            "rejected_labels": ["unknown", "silence"],
            "policy": self.policy.to_dict(),
            "protocol": {
                "path": "/api/live",
                "encoding": ENCODING,
                "channels": 1,
                "sample_rates": [MIN_SAMPLE_RATE, MAX_SAMPLE_RATE],
                "max_frame_bytes": self.policy.max_frame_bytes,
            },
            "retention": "No raw mic audio is stored. Decisions are stored as metadata only.",
        }

    def command_map(self) -> dict[str, Any]:
        """Deprecated alias for GET /api/commands. Returns the legacy shape."""
        payload = self.action_map()
        return {**payload, "commands": payload["actions"]}

    def unavailable(self) -> str | None:
        health = self.recordings.model_health()
        return None if health["status"] == "ready" else str(health["code"])

    def acquire(self) -> bool:
        if self.active:
            return False
        self.active = True
        return True

    def release(self) -> None:
        self.active = False

    def score(self, windows: list[NDArray[np.float32]]) -> NDArray[np.float32]:
        loaded = self.recordings.loaded
        if loaded is None:
            raise RuntimeError("Model is unavailable")
        gate = 10 ** (self.policy.rms_gate_dbfs / 20)
        loud = [i for i, w in enumerate(windows) if window_rms(w) >= gate]
        result = np.zeros((len(windows), len(loaded.labels)), dtype=np.float32)
        result[:, loaded.labels.index("silence")] = 1.0
        if loud:
            pipeline = loaded.pipeline()
            result[loud] = loaded.probabilities(
                np.stack([pipeline.from_audio(windows[i], SAMPLE_RATE) for i in loud])
            )
        return result

    def labels(self) -> tuple[str, ...]:
        loaded = self.recordings.loaded
        return loaded.labels if loaded is not None else LABELS

    def record(self, session: str, decision: Decision) -> int:
        with self.recordings.store.transaction() as db:
            cursor = db.execute(
                "INSERT INTO live_events (session_id,keyword,score,accepted,action,action_status) "
                "VALUES (?,?,?,?,?,?)",
                (
                    session,
                    decision.label,
                    decision.score,
                    int(decision.accepted),
                    decision.action,
                    "dispatched" if decision.accepted else None,
                ),
            )
        assert cursor.lastrowid is not None
        return cursor.lastrowid

    def acknowledge(self, session: str, decision: object, status: object) -> bool:
        """Store the in-app result the browser reports for an accepted decision."""
        if status not in ACTION_STATUSES or not isinstance(decision, int):
            return False
        try:
            with self.recordings.store.transaction() as db:
                updated = db.execute(
                    "UPDATE live_events SET action_status=? "
                    "WHERE id=? AND session_id=? AND accepted=1 AND action_status='dispatched'",
                    (status, decision, session),
                ).rowcount
        except sqlite3.Error:
            return False
        return bool(updated)
