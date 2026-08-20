from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any
import json
import os
import threading
import time
import uuid


FILE_REFERENCE_INDEX_SCHEMA_VERSION = 1
DEFAULT_REFRESH_INTERVAL_SECONDS = 5 * 60


class PersistentFileReferenceIndex:
    """Account-scoped filename index used by chat attachment resolution.

    The index is intentionally metadata-only. Image bytes are still read from
    their canonical workspace paths when a vision-capable model request is
    assembled, so later turns continue to receive the original pixels without
    storing Base64 payloads in conversation history.
    """

    def __init__(
        self,
        *,
        workspace_root: str | Path,
        index_path: str | Path,
        scan_roots: Iterable[str | Path],
        is_visible: Callable[[Path], bool],
        refresh_interval_seconds: int = DEFAULT_REFRESH_INTERVAL_SECONDS,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.index_path = Path(index_path).resolve()
        self.scan_roots = tuple(Path(path).resolve() for path in scan_roots)
        self.is_visible = is_visible
        self.refresh_interval_seconds = max(1, int(refresh_interval_seconds))
        self._lock = threading.RLock()
        self._refresh_lock = threading.Lock()
        self._loaded = False
        self._refreshing = False
        self._refreshed_at = 0
        self._by_name: dict[str, list[dict[str, Any]]] = {}

    def snapshot(self, *, refresh_if_missing: bool = True) -> dict[str, list[str]]:
        """Return a filename-to-path snapshot without rescanning per lookup."""

        self._ensure_loaded()
        if refresh_if_missing and not self._refreshed_at:
            self.refresh()
        elif self._is_stale():
            self.refresh_async()
        with self._lock:
            return {
                name: [str(item["path"]) for item in entries]
                for name, entries in self._by_name.items()
            }

    def lookup(self, names: Iterable[str]) -> dict[str, list[str]]:
        requested = {str(name) for name in names if str(name)}
        if not requested:
            return {}
        snapshot = self.snapshot()
        return {name: snapshot.get(name, []) for name in requested}

    def warm_async(self) -> None:
        """Load a persisted snapshot and reconcile only when missing or stale."""

        self._ensure_loaded()
        with self._lock:
            needs_refresh = not self._refreshed_at or self._is_stale()
        if needs_refresh:
            self.refresh_async()

    def refresh_async(self) -> None:
        with self._lock:
            if self._refreshing:
                return
            self._refreshing = True

        def run() -> None:
            try:
                self.refresh()
            finally:
                with self._lock:
                    self._refreshing = False

        threading.Thread(
            target=run,
            name=f"file-reference-index-{self.index_path.stem[:24]}",
            daemon=True,
        ).start()

    def refresh(self) -> None:
        with self._refresh_lock:
            self._refresh_unlocked()

    def _refresh_unlocked(self) -> None:
        entries: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        for scan_root in self.scan_roots:
            if not self._within_workspace(scan_root) or not scan_root.is_dir():
                continue
            for directory, dirnames, filenames in os.walk(scan_root):
                dirnames[:] = [name for name in dirnames if not name.startswith(".")]
                parent = Path(directory)
                for filename in filenames:
                    if filename.startswith("."):
                        continue
                    path = parent / filename
                    try:
                        if not self.is_visible(path):
                            continue
                        stat = path.stat()
                        relative_path = str(path.resolve().relative_to(self.workspace_root))
                    except (OSError, RuntimeError, ValueError):
                        continue
                    if relative_path in seen_paths:
                        continue
                    seen_paths.add(relative_path)
                    entries.append(
                        {
                            "path": relative_path,
                            "name": path.name,
                            "size": int(stat.st_size),
                            "mtime_ns": int(stat.st_mtime_ns),
                        }
                    )
        entries.sort(key=lambda item: (int(item["mtime_ns"]), str(item["path"])), reverse=True)
        by_name = self._group_entries(entries)
        refreshed_at = int(time.time())
        payload = {
            "schema_version": FILE_REFERENCE_INDEX_SCHEMA_VERSION,
            "workspace_root": str(self.workspace_root),
            "refreshed_at": refreshed_at,
            "files": entries,
        }
        self._write_payload(payload)
        with self._lock:
            self._by_name = by_name
            self._refreshed_at = refreshed_at
            self._loaded = True

    def upsert(self, path: str | Path) -> None:
        """Immediately reflect a file created through a known application path."""

        candidate = Path(path).resolve()
        if not self._within_workspace(candidate):
            return
        self._ensure_loaded()
        with self._lock:
            had_complete_snapshot = self._refreshed_at > 0
            entries = [
                item
                for values in self._by_name.values()
                for item in values
                if str(item.get("path") or "") != self._relative(candidate)
            ]
        try:
            visible = candidate.is_file() and self.is_visible(candidate)
            stat = candidate.stat() if visible else None
        except OSError:
            visible = False
            stat = None
        if visible and stat is not None:
            entries.append(
                {
                    "path": self._relative(candidate),
                    "name": candidate.name,
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                }
            )
        entries.sort(key=lambda item: (int(item["mtime_ns"]), str(item["path"])), reverse=True)
        refreshed_at = max(self._refreshed_at, int(time.time())) if had_complete_snapshot else 0
        payload = {
            "schema_version": FILE_REFERENCE_INDEX_SCHEMA_VERSION,
            "workspace_root": str(self.workspace_root),
            "refreshed_at": refreshed_at,
            "files": entries,
        }
        self._write_payload(payload)
        with self._lock:
            self._by_name = self._group_entries(entries)
            self._refreshed_at = refreshed_at
            self._loaded = True
        if not had_complete_snapshot:
            self.refresh_async()

    def remove(self, path: str | Path) -> None:
        candidate = Path(path).resolve()
        if not self._within_workspace(candidate):
            return
        self._ensure_loaded()
        relative_path = self._relative(candidate)
        with self._lock:
            had_complete_snapshot = self._refreshed_at > 0
            entries = [
                item
                for values in self._by_name.values()
                for item in values
                if str(item.get("path") or "") != relative_path
            ]
        refreshed_at = max(self._refreshed_at, int(time.time())) if had_complete_snapshot else 0
        payload = {
            "schema_version": FILE_REFERENCE_INDEX_SCHEMA_VERSION,
            "workspace_root": str(self.workspace_root),
            "refreshed_at": refreshed_at,
            "files": entries,
        }
        self._write_payload(payload)
        with self._lock:
            self._by_name = self._group_entries(entries)
            self._refreshed_at = refreshed_at
            self._loaded = True
        if not had_complete_snapshot:
            self.refresh_async()

    def _ensure_loaded(self) -> None:
        with self._lock:
            if self._loaded:
                return
            payload = self._read_payload()
            files = payload.get("files") if isinstance(payload, dict) else []
            valid_files = []
            if (
                isinstance(payload, dict)
                and int(payload.get("schema_version") or 0) == FILE_REFERENCE_INDEX_SCHEMA_VERSION
                and str(payload.get("workspace_root") or "") == str(self.workspace_root)
                and isinstance(files, list)
            ):
                valid_files = [item for item in files if self._valid_entry(item)]
                self._refreshed_at = max(0, int(payload.get("refreshed_at") or 0))
            self._by_name = self._group_entries(valid_files)
            self._loaded = True

    def _is_stale(self) -> bool:
        with self._lock:
            return bool(
                self._refreshed_at
                and time.time() - self._refreshed_at >= self.refresh_interval_seconds
            )

    def _read_payload(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _write_payload(self, payload: dict[str, Any]) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.index_path.with_name(f".{self.index_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            temporary.replace(self.index_path)
        finally:
            temporary.unlink(missing_ok=True)

    def _within_workspace(self, path: Path) -> bool:
        return self.workspace_root in (path, *path.parents)

    def _relative(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.workspace_root))
        except ValueError:
            return ""

    @staticmethod
    def _valid_entry(item: Any) -> bool:
        return bool(
            isinstance(item, dict)
            and str(item.get("path") or "")
            and str(item.get("name") or "")
        )

    @staticmethod
    def _group_entries(entries: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        by_name: dict[str, list[dict[str, Any]]] = {}
        for item in entries:
            name = str(item.get("name") or "")
            if name:
                by_name.setdefault(name, []).append(dict(item))
        return by_name


_INDEX_CACHE: dict[tuple[str, str], PersistentFileReferenceIndex] = {}
_INDEX_CACHE_LOCK = threading.RLock()


def get_persistent_file_reference_index(
    *,
    workspace_root: str | Path,
    index_path: str | Path,
    scan_roots: Iterable[str | Path],
    is_visible: Callable[[Path], bool],
    refresh_interval_seconds: int = DEFAULT_REFRESH_INTERVAL_SECONDS,
) -> PersistentFileReferenceIndex:
    root = Path(workspace_root).resolve()
    persistent_path = Path(index_path).resolve()
    key = (str(root), str(persistent_path))
    with _INDEX_CACHE_LOCK:
        index = _INDEX_CACHE.get(key)
        if index is None:
            index = PersistentFileReferenceIndex(
                workspace_root=root,
                index_path=persistent_path,
                scan_roots=scan_roots,
                is_visible=is_visible,
                refresh_interval_seconds=refresh_interval_seconds,
            )
            _INDEX_CACHE[key] = index
        return index
