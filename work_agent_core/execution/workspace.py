from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import difflib
import hashlib
import json
import os
import shutil
import tempfile
import time
import uuid

from .errors import failure
from .models import ChangeSet, FileChange
from .policy import contract_digest


DEFAULT_EXCLUDED_NAMES = {
    ".git",
    ".env",
    ".ssh",
    ".DS_Store",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    ".venv_agent",
    "meet_files",
    "conversation_history",
    "model_cache",
    ".execution-test-data",
}
MAX_TEXT_DIFF_BYTES = 1_000_000
DEFAULT_MAX_SNAPSHOT_FILES = 25_000
DEFAULT_MAX_SNAPSHOT_BYTES = 512 * 1024 * 1024
EXCLUDED_INPUT_SUFFIXES = {
    ".aac",
    ".aiff",
    ".flac",
    ".m4a",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".ogg",
    ".wav",
    ".webm",
}


@dataclass(frozen=True)
class WorkspaceSnapshot:
    snapshot_id: str
    account_id: str
    project_id: str
    source_root: Path
    root: Path
    manifest_path: Path
    created_at_ms: int
    file_count: int
    total_bytes: int

    @property
    def workspace_path(self) -> Path:
        return self.root / "workspace"


class WorkspaceManager:
    """Creates private execution copies and applies verified change sets atomically."""

    def __init__(
        self,
        execution_root: str | Path,
        *,
        max_snapshot_files: int = DEFAULT_MAX_SNAPSHOT_FILES,
        max_snapshot_bytes: int = DEFAULT_MAX_SNAPSHOT_BYTES,
    ) -> None:
        self.execution_root = Path(execution_root).resolve()
        self.snapshots_root = self.execution_root / "snapshots"
        self.change_root = self.execution_root / "changes"
        self.max_snapshot_files = max(1, int(max_snapshot_files))
        self.max_snapshot_bytes = max(1, int(max_snapshot_bytes))
        self.snapshots_root.mkdir(parents=True, exist_ok=True)
        self.change_root.mkdir(parents=True, exist_ok=True)

    def create_snapshot(
        self,
        *,
        source_root: str | Path,
        account_id: str,
        project_id: str,
        excludes: Iterable[str] = DEFAULT_EXCLUDED_NAMES,
    ) -> WorkspaceSnapshot:
        source = Path(source_root).resolve()
        if not source.is_dir():
            raise failure("WORKSPACE_UNAVAILABLE", "当前账户工作区不存在或不可访问。", phase="preparing")
        snapshot_id = f"snap_{uuid.uuid4().hex}"
        root = self.snapshots_root / snapshot_id
        target = root / "workspace"
        root.mkdir(parents=True, exist_ok=False)
        try:
            manifest_entries = self._copy_tree(source, target, excludes={str(item) for item in excludes})
            manifest_path = root / "manifest.json"
            manifest_payload = {
                "schema_version": 1,
                "snapshot_id": snapshot_id,
                "account_id": account_id,
                "project_id": project_id,
                "source_root": str(source),
                "created_at_ms": int(time.time() * 1000),
                "entries": manifest_entries,
            }
            _atomic_write_json(manifest_path, manifest_payload)
            total_bytes = sum(int(item.get("size") or 0) for item in manifest_entries if item.get("type") == "file")
            return WorkspaceSnapshot(
                snapshot_id=snapshot_id,
                account_id=account_id,
                project_id=project_id,
                source_root=source,
                root=root,
                manifest_path=manifest_path,
                created_at_ms=int(manifest_payload["created_at_ms"]),
                file_count=sum(1 for item in manifest_entries if item.get("type") == "file"),
                total_bytes=total_bytes,
            )
        except Exception:
            shutil.rmtree(root, ignore_errors=True)
            raise

    def load_snapshot(self, snapshot_id: str) -> WorkspaceSnapshot:
        root = (self.snapshots_root / snapshot_id).resolve()
        if self.snapshots_root not in (root, *root.parents):
            raise ValueError("非法快照标识。")
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"执行快照不存在：{snapshot_id}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = payload.get("entries") if isinstance(payload.get("entries"), list) else []
        return WorkspaceSnapshot(
            snapshot_id=str(payload.get("snapshot_id") or snapshot_id),
            account_id=str(payload.get("account_id") or ""),
            project_id=str(payload.get("project_id") or ""),
            source_root=Path(str(payload.get("source_root") or "")).resolve(),
            root=root,
            manifest_path=manifest_path,
            created_at_ms=int(payload.get("created_at_ms") or 0),
            file_count=sum(1 for item in entries if isinstance(item, dict) and item.get("type") == "file"),
            total_bytes=sum(int(item.get("size") or 0) for item in entries if isinstance(item, dict) and item.get("type") == "file"),
        )

    def capture_changes(self, *, execution_id: str, snapshot: WorkspaceSnapshot) -> ChangeSet:
        baseline = self._manifest_index(snapshot.manifest_path)
        current = self._tree_index(snapshot.workspace_path)
        all_paths = sorted(set(baseline) | set(current))
        changes: list[FileChange] = []
        change_id = f"chg_{uuid.uuid4().hex}"
        change_dir = self.change_root / change_id
        change_dir.mkdir(parents=True, exist_ok=False)
        for relative_path in all_paths:
            before = baseline.get(relative_path)
            after = current.get(relative_path)
            if before == after:
                continue
            if before is None:
                change_type = "directory_added" if after and after.get("type") == "directory" else "added"
            elif after is None:
                change_type = "directory_deleted" if before.get("type") == "directory" else "deleted"
            elif before.get("type") != after.get("type"):
                change_type = "replaced"
            elif before.get("hash") != after.get("hash") or before.get("mode") != after.get("mode"):
                change_type = "modified" if before.get("mode") == after.get("mode") else "mode_changed"
            else:
                continue
            binary = bool((before or after or {}).get("binary"))
            diff_ref = self._write_diff(change_dir, relative_path, snapshot, before, after, binary)
            changes.append(
                FileChange(
                    path=relative_path,
                    change_type=change_type,
                    base_hash=str(before.get("hash")) if before and before.get("hash") else None,
                    result_hash=str(after.get("hash")) if after and after.get("hash") else None,
                    size_before=int(before.get("size")) if before and before.get("size") is not None else None,
                    size_after=int(after.get("size")) if after and after.get("size") is not None else None,
                    binary=binary,
                    diff_ref=diff_ref,
                )
            )
        payload = {
            "change_set_id": change_id,
            "execution_id": execution_id,
            "snapshot_id": snapshot.snapshot_id,
            "generated_at_ms": int(time.time() * 1000),
            "changes": [
                {
                    "path": item.path,
                    "change_type": item.change_type,
                    "base_hash": item.base_hash,
                    "result_hash": item.result_hash,
                    "size_before": item.size_before,
                    "size_after": item.size_after,
                    "binary": item.binary,
                    "diff_ref": item.diff_ref,
                }
                for item in changes
            ],
        }
        digest = contract_digest(payload)
        payload["digest"] = digest
        _atomic_write_json(change_dir / "change_set.json", payload)
        return ChangeSet(
            change_set_id=change_id,
            execution_id=execution_id,
            snapshot_id=snapshot.snapshot_id,
            changes=tuple(changes),
            generated_at_ms=int(payload["generated_at_ms"]),
            digest=digest,
        )

    def apply_changes(
        self,
        *,
        snapshot: WorkspaceSnapshot,
        change_set: ChangeSet,
        selected_paths: tuple[str, ...] | None = None,
    ) -> list[str]:
        allowed = set(selected_paths) if selected_paths is not None else None
        selected = [change for change in change_set.changes if allowed is None or change.path in allowed]
        if allowed is not None and {change.path for change in selected} != allowed:
            raise failure("WORKSPACE_CONFLICT", "待写回的文件列表与变更集不一致。", phase="applying")
        self._verify_apply_baseline(snapshot, selected)
        applied: list[str] = []
        additions = sorted(
            (change for change in selected if not _is_deletion(change)),
            key=lambda change: (len(Path(change.path).parts), change.path),
        )
        deletions = sorted(
            (change for change in selected if _is_deletion(change)),
            key=lambda change: (-len(Path(change.path).parts), change.path),
        )
        for change in additions:
            destination = self._safe_destination(snapshot.source_root, change.path)
            source = self._safe_destination(snapshot.workspace_path, change.path)
            if change.change_type == "replaced":
                raise failure("WORKSPACE_CONFLICT", f"不允许原子写回文件类型替换：{change.path}", phase="applying")
            if _is_directory_addition(change):
                if not source.is_dir() or source.is_symlink():
                    raise failure("APPLY_FAILED", f"隔离环境中的目录产物不存在或不安全：{change.path}", phase="applying")
                destination.mkdir(parents=True, exist_ok=False)
                applied.append(change.path)
                continue
            if not source.is_file() or source.is_symlink():
                raise failure("APPLY_FAILED", f"隔离环境中的产物不存在或不是普通文件：{change.path}", phase="applying")
            destination.parent.mkdir(parents=True, exist_ok=True)
            _atomic_copy(source, destination)
            applied.append(change.path)
        for change in deletions:
            destination = self._safe_destination(snapshot.source_root, change.path)
            if _is_directory_deletion(change):
                if destination.is_symlink() or not destination.is_dir():
                    raise failure("WORKSPACE_CONFLICT", f"写回前目录已被删除或替换：{change.path}", phase="applying")
                try:
                    destination.rmdir()
                except OSError as error:
                    raise failure("WORKSPACE_CONFLICT", f"写回前目录不为空，拒绝删除：{change.path}", phase="applying") from error
            elif destination.exists() or destination.is_symlink():
                destination.unlink()
            applied.append(change.path)
        self._verify_after_apply(snapshot, selected)
        return applied

    def load_change_set(self, change_set_id: str) -> ChangeSet:
        root = (self.change_root / change_set_id).resolve()
        if self.change_root not in (root, *root.parents):
            raise ValueError("非法变更集标识。")
        path = root / "change_set.json"
        if not path.is_file():
            raise FileNotFoundError(f"变更集不存在：{change_set_id}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        changes = tuple(
            FileChange(
                path=str(item.get("path") or ""),
                change_type=str(item.get("change_type") or "modified"),
                base_hash=str(item["base_hash"]) if item.get("base_hash") else None,
                result_hash=str(item["result_hash"]) if item.get("result_hash") else None,
                size_before=int(item["size_before"]) if item.get("size_before") is not None else None,
                size_after=int(item["size_after"]) if item.get("size_after") is not None else None,
                binary=bool(item.get("binary")),
                diff_ref=str(item["diff_ref"]) if item.get("diff_ref") else None,
            )
            for item in payload.get("changes") or ()
            if isinstance(item, dict) and str(item.get("path") or "")
        )
        return ChangeSet(
            change_set_id=str(payload.get("change_set_id") or change_set_id),
            execution_id=str(payload.get("execution_id") or ""),
            snapshot_id=str(payload.get("snapshot_id") or ""),
            changes=changes,
            generated_at_ms=int(payload.get("generated_at_ms") or 0),
            digest=str(payload.get("digest") or ""),
        )

    def _copy_tree(self, source: Path, target: Path, *, excludes: set[str]) -> list[dict[str, Any]]:
        directories: list[tuple[Path, Path]] = []
        files_to_copy: list[tuple[Path, Path]] = []
        total_bytes = 0
        for root, dirs, files in os.walk(source, topdown=True, followlinks=False):
            current = Path(root)
            relative_parent = current.relative_to(source)
            filtered_dirs: list[str] = []
            for name in dirs:
                candidate = current / name
                if self._should_exclude_candidate(candidate, source, excludes):
                    continue
                if candidate.is_symlink():
                    self._assert_safe_symlink(candidate, source)
                    raise failure("WORKSPACE_SYMLINK_UNSUPPORTED", f"执行快照不支持目录符号链接：{candidate.relative_to(source)}", phase="preparing")
                filtered_dirs.append(name)
                directories.append((relative_parent / name, candidate))
            dirs[:] = filtered_dirs
            for name in files:
                candidate = current / name
                if self._should_exclude_candidate(candidate, source, excludes):
                    continue
                relative = relative_parent / name
                if candidate.is_symlink():
                    self._assert_safe_symlink(candidate, source)
                    raise failure("WORKSPACE_SYMLINK_UNSUPPORTED", f"执行快照不支持文件符号链接：{relative}", phase="preparing")
                if not candidate.is_file():
                    continue
                total_bytes += candidate.stat().st_size
                files_to_copy.append((relative, candidate))
                if len(files_to_copy) > self.max_snapshot_files:
                    raise failure(
                        "SNAPSHOT_LIMIT_EXCEEDED",
                        f"安全执行工作区文件数超过上限（{self.max_snapshot_files}），请把 cwd 收敛到具体项目目录。",
                        phase="preparing",
                        user_action="change_scope",
                    )
                if total_bytes > self.max_snapshot_bytes:
                    raise failure(
                        "SNAPSHOT_LIMIT_EXCEEDED",
                        f"安全执行工作区超过 {self.max_snapshot_bytes // (1024 * 1024)}MB 上限；会议原始资料应通过受管能力处理，请把 cwd 收敛到具体项目目录。",
                        phase="preparing",
                        user_action="change_scope",
                    )
        target.mkdir(parents=True, exist_ok=True)
        entries: list[dict[str, Any]] = []
        for relative, directory in directories:
            destination_dir = target / relative
            destination_dir.mkdir(parents=True, exist_ok=True)
            entries.append({"path": str(relative.as_posix()), "type": "directory", "mode": directory.stat().st_mode & 0o777})
        for relative, candidate in files_to_copy:
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, destination, follow_symlinks=False)
            entries.append(_file_entry(relative, candidate))
        return sorted(entries, key=lambda item: str(item["path"]))

    def _tree_index(self, root: Path) -> dict[str, dict[str, Any]]:
        if not root.is_dir():
            raise failure("WORKSPACE_UNAVAILABLE", "执行工作区不存在。", phase="validating")
        index: dict[str, dict[str, Any]] = {}
        for path in root.rglob("*"):
            relative = path.relative_to(root)
            if relative.parts and relative.parts[0] in {".work-agent-home", ".work-agent-tmp"}:
                continue
            if path.is_symlink():
                raise failure("WORKSPACE_SYMLINK_UNSUPPORTED", f"执行结果包含不允许的符号链接：{relative}", phase="validating")
            if path.is_dir():
                index[str(relative.as_posix())] = {"type": "directory", "mode": path.stat().st_mode & 0o777}
            elif path.is_file():
                index[str(relative.as_posix())] = _file_entry(relative, path)
        return index

    def _manifest_index(self, manifest_path: Path) -> dict[str, dict[str, Any]]:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = payload.get("entries") if isinstance(payload.get("entries"), list) else []
        return {
            str(item.get("path") or ""): dict(item)
            for item in entries
            if isinstance(item, dict) and str(item.get("path") or "")
        }

    def _write_diff(
        self,
        change_dir: Path,
        relative_path: str,
        snapshot: WorkspaceSnapshot,
        before: dict[str, Any] | None,
        after: dict[str, Any] | None,
        binary: bool,
    ) -> str | None:
        if binary:
            return None
        source = self._safe_destination(snapshot.source_root, relative_path)
        target = self._safe_destination(snapshot.workspace_path, relative_path)
        before_text = _read_text_for_diff(source) if before and source.is_file() else ""
        after_text = _read_text_for_diff(target) if after and target.is_file() else ""
        if before_text is None or after_text is None:
            return None
        lines = difflib.unified_diff(
            before_text.splitlines(keepends=True),
            after_text.splitlines(keepends=True),
            fromfile=f"a/{relative_path}",
            tofile=f"b/{relative_path}",
        )
        diff_path = change_dir / "diffs" / (hashlib.sha256(relative_path.encode("utf-8")).hexdigest() + ".diff")
        diff_path.parent.mkdir(parents=True, exist_ok=True)
        diff_path.write_text("".join(lines), encoding="utf-8")
        return str(diff_path.relative_to(self.execution_root))

    def _verify_apply_baseline(self, snapshot: WorkspaceSnapshot, changes: list[FileChange]) -> None:
        for change in changes:
            destination = self._safe_destination(snapshot.source_root, change.path)
            if _is_directory_addition(change):
                if destination.exists() or destination.is_symlink():
                    raise failure("WORKSPACE_CONFLICT", f"写回前目标目录已存在：{change.path}", phase="applying")
                continue
            if _is_directory_deletion(change):
                if destination.is_symlink() or not destination.is_dir():
                    raise failure("WORKSPACE_CONFLICT", f"写回前目录已被删除或替换：{change.path}", phase="applying")
                continue
            if change.change_type == "replaced":
                raise failure("WORKSPACE_CONFLICT", f"不允许原子写回文件类型替换：{change.path}", phase="applying")
            if change.base_hash is None:
                if destination.exists() or destination.is_symlink():
                    raise failure("WORKSPACE_CONFLICT", f"写回前目标文件已存在：{change.path}", phase="applying")
                continue
            if not destination.is_file() or destination.is_symlink():
                raise failure("WORKSPACE_CONFLICT", f"写回前目标文件已被删除或替换：{change.path}", phase="applying")
            if hash_file(destination) != change.base_hash:
                raise failure("WORKSPACE_CONFLICT", f"写回前文件已被其他操作修改：{change.path}", phase="applying")

    def _verify_after_apply(self, snapshot: WorkspaceSnapshot, changes: list[FileChange]) -> None:
        for change in changes:
            destination = self._safe_destination(snapshot.source_root, change.path)
            if _is_directory_deletion(change):
                if destination.exists() or destination.is_symlink():
                    raise failure("APPLY_FAILED", f"目录删除后仍然存在：{change.path}", phase="applying")
                continue
            if _is_directory_addition(change):
                if destination.is_symlink() or not destination.is_dir():
                    raise failure("APPLY_FAILED", f"目录写回后校验失败：{change.path}", phase="applying")
                continue
            if change.result_hash is None:
                if destination.exists() or destination.is_symlink():
                    raise failure("APPLY_FAILED", f"文件删除后仍然存在：{change.path}", phase="applying")
                continue
            if not destination.is_file() or hash_file(destination) != change.result_hash:
                raise failure("APPLY_FAILED", f"文件写回后校验失败：{change.path}", phase="applying")

    def _safe_destination(self, root: Path, relative_path: str) -> Path:
        candidate = (root / relative_path).resolve(strict=False)
        if root.resolve() not in (candidate, *candidate.parents):
            raise failure("WORKSPACE_CONFLICT", "变更集包含越出工作区的路径。", phase="applying")
        current = candidate.parent
        while current != root.resolve() and current != current.parent:
            if current.is_symlink():
                raise failure("WORKSPACE_CONFLICT", f"目标路径经过符号链接：{relative_path}", phase="applying")
            current = current.parent
        return candidate

    def _assert_safe_symlink(self, path: Path, root: Path) -> None:
        target = path.resolve(strict=False)
        if root not in (target, *target.parents):
            raise failure("WORKSPACE_CONFLICT", f"工作区符号链接越出项目范围：{path.relative_to(root)}", phase="preparing")

    @staticmethod
    def _is_sensitive_name(name: str) -> bool:
        lower = name.lower()
        return lower in {".env", ".ssh", "id_rsa", "id_ed25519", "secrets", "private_key"}

    def _should_exclude_candidate(self, candidate: Path, source_root: Path, excludes: set[str]) -> bool:
        if candidate.name in excludes or self._is_sensitive_name(candidate.name):
            return True
        if candidate.is_file() and candidate.suffix.lower() in EXCLUDED_INPUT_SUFFIXES:
            return True
        resolved = candidate.resolve(strict=False)
        if self.execution_root in (resolved, *resolved.parents):
            return True
        # A snapshot must never recursively include itself even when a caller
        # chooses an execution root under a nested workspace path.
        return self.snapshots_root in (resolved, *resolved.parents)


def _file_entry(relative: Path, path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "path": str(relative.as_posix()),
        "type": "file",
        "size": len(raw),
        "hash": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "mode": path.stat().st_mode & 0o777,
        "binary": b"\0" in raw[:8192],
    }


def _is_directory_addition(change: FileChange) -> bool:
    return change.change_type == "directory_added"


def _is_directory_deletion(change: FileChange) -> bool:
    return change.change_type == "directory_deleted"


def _is_deletion(change: FileChange) -> bool:
    return change.change_type in {"deleted", "directory_deleted"}


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_text_for_diff(path: Path) -> str | None:
    if not path.is_file() or path.stat().st_size > MAX_TEXT_DIFF_BYTES:
        return None
    raw = path.read_bytes()
    if b"\0" in raw[:8192]:
        return None
    return raw.decode("utf-8", errors="replace")


def _atomic_copy(source: Path, destination: Path) -> None:
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".work-agent-", delete=False) as handle:
        temporary = Path(handle.name)
        with source.open("rb") as input_handle:
            shutil.copyfileobj(input_handle, handle)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.chmod(temporary, source.stat().st_mode & 0o777)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".work-agent-", delete=False, mode="w", encoding="utf-8") as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)
