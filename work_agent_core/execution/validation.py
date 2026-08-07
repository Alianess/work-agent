from __future__ import annotations

from pathlib import Path
from typing import Iterable
import json
import uuid

from .models import ValidationOutcome, ValidationSpec


class ValidationService:
    def validate(self, workspace_root: Path, specs: Iterable[ValidationSpec]) -> tuple[ValidationOutcome, ...]:
        results: list[ValidationOutcome] = []
        for spec in specs:
            target = self._resolve(workspace_root, spec.target)
            status = "passed"
            detail = ""
            try:
                if spec.kind == "file_exists":
                    if not target.exists():
                        raise FileNotFoundError("文件不存在")
                    detail = "文件存在"
                elif spec.kind == "file_readable":
                    if not target.is_file():
                        raise FileNotFoundError("文件不存在或不是普通文件")
                    target.read_bytes()
                    detail = "文件可读取"
                elif spec.kind == "json_schema":
                    payload = json.loads(target.read_text(encoding="utf-8"))
                    required = tuple(str(item) for item in spec.options.get("required_keys") or ())
                    if not isinstance(payload, dict) or any(key not in payload for key in required):
                        raise ValueError("JSON 未满足所需字段")
                    detail = "JSON 结构通过基础校验"
                elif spec.kind == "mime":
                    expected_suffix = str(spec.options.get("suffix") or "").lower()
                    if expected_suffix and target.suffix.lower() != expected_suffix:
                        raise ValueError(f"文件扩展名不是 {expected_suffix}")
                    detail = "文件类型通过基础校验"
                else:
                    status = "skipped"
                    detail = f"当前验证器尚未执行 {spec.kind}，保留给专用 Validator。"
            except Exception as error:
                status = "failed"
                detail = str(error)
            results.append(
                ValidationOutcome(
                    validation_id=f"val_{uuid.uuid4().hex}",
                    kind=spec.kind,
                    target=spec.target,
                    status=status,
                    detail=detail,
                )
            )
        return tuple(results)

    @staticmethod
    def _resolve(root: Path, raw: str) -> Path:
        candidate = (root / raw).resolve(strict=False)
        if root.resolve() not in (candidate, *candidate.parents):
            raise ValueError("验证目标越出执行工作区")
        return candidate
