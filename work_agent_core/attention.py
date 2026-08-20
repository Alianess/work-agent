"""The pass where the assistant looks at your work and decides whether to speak.

The existing background loop is a delivery scheduler: it sends what was already
queued and audits missing daily reports. Nothing in it ever *notices* anything,
which is why the assistant only ever spoke to nag about paperwork.

An observer looks at one slice of workspace state and returns observations. The
loop collects them, drops what is not worth saying or was said recently, and
speaks at most once per pass. Observers are registered, not hardcoded: noticing
a new kind of thing means adding one, not editing the loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol
import hashlib
import json
import threading
import time


# Below this, an observation is not worth interrupting for.
DEFAULT_SALIENCE_FLOOR = 0.5
# An assistant that repeats itself is worse than one that stays quiet.
DEFAULT_QUIET_HOURS = 20.0
MAX_OBSERVATIONS_PER_PASS = 3


@dataclass(frozen=True)
class Observation:
    """Something worth telling the user, and how much it is worth."""

    key: str
    """Stable identity used to avoid repeating the same remark."""

    summary: str
    salience: float = 0.5
    quiet_hours: float = DEFAULT_QUIET_HOURS
    detail: str = ""
    source: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def dedup_key(self) -> str:
        return hashlib.sha1(f"{self.source}:{self.key}".encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ObserverContext:
    """What an observer is allowed to look at.

    Two sources, deliberately distinct. ``ledger`` is what the assistant itself
    handled, projected from its own record; ``data_root`` is the world, which
    changes without telling it. Deriving handled work from the filesystem is
    archaeology — it reads the residue and loses who asked for what — so an
    observer about past work must read the ledger, and only observers about
    unhandled input should scan.
    """

    workspace_root: Path
    data_root: Path
    now: datetime
    ledger: Any = None


class Observer(Protocol):
    id: str

    def observe(self, context: ObserverContext) -> Iterable[Observation]: ...


@dataclass
class FunctionObserver:
    """Wrap a plain function so simple checks do not need a class."""

    id: str
    fn: Callable[[ObserverContext], Iterable[Observation]]
    enabled: bool = True

    def observe(self, context: ObserverContext) -> Iterable[Observation]:
        return self.fn(context)


class ObserverRegistry:
    def __init__(self, observers: Iterable[Observer] = ()) -> None:
        self._lock = threading.RLock()
        self._observers: dict[str, Observer] = {}
        for observer in observers:
            self.register(observer)

    def register(self, observer: Observer) -> Observer:
        with self._lock:
            self._observers[observer.id] = observer
        return observer

    def list(self) -> list[Observer]:
        with self._lock:
            return [item for item in self._observers.values() if getattr(item, "enabled", True)]

    def run(self, context: ObserverContext) -> list[Observation]:
        """Run every observer. One failing observer must not silence the rest."""
        found: list[Observation] = []
        for observer in self.list():
            try:
                for observation in observer.observe(context) or ():
                    found.append(observation)
            except Exception:
                continue
        return found


class SpokenLedger:
    """Remembers what was already said, so the assistant does not repeat itself."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _load(self) -> dict[str, float]:
        if not self.path.is_file():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return {str(key): float(value) for key, value in payload.items()} if isinstance(payload, dict) else {}

    def filter_unsaid(self, observations: list[Observation], *, now: float | None = None) -> list[Observation]:
        moment = now if now is not None else time.time()
        with self._lock:
            said = self._load()
        keep: list[Observation] = []
        for observation in observations:
            last = said.get(observation.dedup_key())
            if last is not None and moment - last < observation.quiet_hours * 3600:
                continue
            keep.append(observation)
        return keep

    def mark_spoken(self, observations: Iterable[Observation], *, now: float | None = None) -> None:
        moment = now if now is not None else time.time()
        with self._lock:
            said = self._load()
            for observation in observations:
                said[observation.dedup_key()] = moment
            cutoff = moment - 30 * 24 * 3600
            said = {key: value for key, value in said.items() if value >= cutoff}
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(said, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.path)


def select_observations(
    observations: list[Observation],
    *,
    floor: float = DEFAULT_SALIENCE_FLOOR,
    limit: int = MAX_OBSERVATIONS_PER_PASS,
) -> list[Observation]:
    """Keep the few most salient remarks and drop the rest.

    Speaking about everything noticed is how a helpful assistant turns into
    noise the user learns to ignore.
    """

    worth_saying = [item for item in observations if item.salience >= floor]
    worth_saying.sort(key=lambda item: -item.salience)
    return worth_saying[:limit]


def compose_message(observations: list[Observation]) -> str:
    """Render selected observations as one short proactive message."""
    if not observations:
        return ""
    if len(observations) == 1:
        single = observations[0]
        return single.summary if not single.detail else f"{single.summary}\n\n{single.detail}"
    lines = ["我看了一下当前的工作状态，有几件事想提一下："]
    for index, observation in enumerate(observations, start=1):
        lines.append(f"{index}. {observation.summary}")
    return "\n".join(lines)
