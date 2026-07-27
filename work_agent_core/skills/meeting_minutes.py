from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import os
import re
import shutil
import subprocess
import sys
import time

from ..config import ModelProfile
from ..audio_metadata import probe_audio_metadata, recording_metadata_summary
from ..llm import OpenAICompatibleClient
from ..progress import run_logged_process
from ..runtime_env import (
    find_runtime_executable,
    project_agent_python,
    runtime_search_path,
)
from ..tools import Tool, ToolRegistry, WorkspaceFiles


SPEC_PATH = Path("meeting_audio_minutes/meeting_minutes_spec.md")
SKILL_SPEC_PATH = Path("meeting_audio_minutes/skills/meeting-minutes/SKILL.md")
ASR_SETTINGS_PATH = Path("config/asr_settings.json")
AGENT_SETTINGS_PATH = Path("config/agent_settings.json")
QWEN3_MLX_REMOTE_MODEL_ID = "mlx-community/Qwen3-ASR-1.7B-8bit"
QWEN3_MLX_LOCAL_MODEL_ID = "meeting_audio_minutes/model_cache/mlx-community/Qwen3-ASR-1.7B-8bit"
TEXT_INPUT_EXTENSIONS = {".md", ".txt", ".srt", ".vtt"}
AUDIO_INPUT_EXTENSIONS = {
    ".m4a",
    ".mp3",
    ".wav",
    ".aac",
    ".flac",
    ".ogg",
    ".opus",
    ".wma",
    ".amr",
    ".aiff",
    ".aif",
    ".caf",
    ".webm",
    ".mp4",
}
DEFAULT_ASR_HOTWORDS = (
    "会议 客户 产品 项目 技术 方案 数据 模型 合作 产业 平台 机器人 "
    "具身智能 智能座舱 工业 教育"
)
DEFAULT_ASR_SETTINGS: dict[str, Any] = {
    "profile": "qwen3-asr-mlx-8bit",
    "model_id": QWEN3_MLX_LOCAL_MODEL_ID,
    "backend": "mlx",
    "hotwords": DEFAULT_ASR_HOTWORDS,
}


@dataclass(frozen=True)
class MeetingMinutesOutputs:
    asr_markdown_path: Path
    asr_text_path: Path
    internal_path: Path
    work_markdown_path: Path
    work_docx_path: Path
    manifest_path: Path


class MeetingMinutesSkill:
    def __init__(
        self,
        *,
        workspace_root: str | Path,
        client: OpenAICompatibleClient,
        profile: ModelProfile,
    ) -> None:
        self.workspace = WorkspaceFiles(workspace_root)
        self.client = client
        self.profile = profile

    def run(self, args: dict[str, Any]) -> str:
        input_path = self._resolve_input_path(args)
        output_dir = self.workspace.resolve(str(args.get("output_dir") or "meet_files"))
        meeting_name = str(args.get("meeting_name") or input_path.stem)
        confirmed_info = str(args.get("confirmed_info") or "")
        external_research = str(args.get("external_research") or args.get("research_notes") or "")
        supplemental_paths = [str(path) for path in args.get("supplemental_paths") or []]
        recording_metadata = probe_audio_metadata(input_path)
        recording_context = recording_metadata_summary(recording_metadata)

        transcript_path, processing_note = self._ensure_transcript(input_path, args)
        if recording_context:
            processing_note = f"{processing_note}\n{recording_context}"
        transcript = transcript_path.read_text(encoding="utf-8")
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_name = sanitize_filename(meeting_name)
        archive_dir = output_dir / "会议项目" / safe_name
        archive_dir.mkdir(parents=True, exist_ok=True)
        asr_markdown_path, asr_text_path = self._write_public_asr_outputs(
            output_dir=archive_dir,
            safe_name=safe_name,
            meeting_name=meeting_name,
            transcript_path=transcript_path,
            transcript=transcript,
        )
        spec = self._load_minutes_spec(args)
        supplements = self._load_supplements(supplemental_paths)

        internal = self._generate_internal_minutes(
            meeting_name=meeting_name,
            transcript=transcript,
            confirmed_info=confirmed_info,
            external_research=external_research,
            supplements=supplements,
            spec=spec,
            processing_note=processing_note,
            recording_context=recording_context,
        )
        work = self._generate_work_minutes(
            meeting_name=meeting_name,
            internal_minutes=internal,
            confirmed_info=confirmed_info,
            spec=spec,
            recording_context=recording_context,
        )

        internal_path = archive_dir / f"{safe_name}_会议沟通内容整理_内部留档版.md"
        work_markdown_path = archive_dir / f"{safe_name}_会议纪要_工作提交版.md"
        work_docx_path = archive_dir / f"{safe_name}会议纪要.docx"
        self.workspace.write_text({"path": str(internal_path), "content": internal})
        self.workspace.write_text({"path": str(work_markdown_path), "content": work})
        self._export_work_docx(
            markdown_path=work_markdown_path,
            output_path=work_docx_path,
            title=f"{meeting_name}会议纪要",
        )
        manifest_path = self._write_archive_manifest(
            archive_dir=archive_dir,
            safe_name=safe_name,
            meeting_name=meeting_name,
            source_path=input_path,
            transcript_path=transcript_path,
            asr_markdown_path=asr_markdown_path,
            asr_text_path=asr_text_path,
            internal_path=internal_path,
            work_markdown_path=work_markdown_path,
            work_docx_path=work_docx_path,
            processing_note=processing_note,
            supplemental_paths=supplemental_paths,
            recording_metadata=recording_metadata,
        )

        result = MeetingMinutesOutputs(
            asr_markdown_path=asr_markdown_path,
            asr_text_path=asr_text_path,
            internal_path=internal_path,
            work_markdown_path=work_markdown_path,
            work_docx_path=work_docx_path,
            manifest_path=manifest_path,
        )
        return json.dumps(
            {
                "archive_dir": workspace_relative_path(archive_dir, self.workspace.workspace_root),
                "manifest_path": workspace_relative_path(result.manifest_path, self.workspace.workspace_root),
                "source_path": str(input_path),
                "transcript_path": str(transcript_path),
                "asr_transcript_path": str(result.asr_markdown_path),
                "asr_markdown_path": str(result.asr_markdown_path),
                "asr_text_path": str(result.asr_text_path),
                "internal_path": str(result.internal_path),
                "work_path": str(result.work_docx_path),
                "work_markdown_path": str(result.work_markdown_path),
                "work_docx_path": str(result.work_docx_path),
                "processing_note": processing_note,
                "recording_metadata": recording_metadata,
            },
            ensure_ascii=False,
            indent=2,
        )

    def transcribe_audio(self, args: dict[str, Any]) -> str:
        input_path = self._resolve_input_path(args)
        output_dir = self.workspace.resolve(str(args.get("output_dir") or "meet_files"))
        meeting_name = str(args.get("meeting_name") or input_path.stem)
        transcript_path, processing_note = self._ensure_transcript(input_path, args)
        recording_metadata = probe_audio_metadata(input_path)
        recording_context = recording_metadata_summary(recording_metadata)
        if recording_context:
            processing_note = f"{processing_note}\n{recording_context}"
        transcript = transcript_path.read_text(encoding="utf-8")
        safe_name = sanitize_filename(meeting_name)
        asr_markdown_path, asr_text_path = self._write_public_asr_outputs(
            output_dir=output_dir,
            safe_name=safe_name,
            meeting_name=meeting_name,
            transcript_path=transcript_path,
            transcript=transcript,
        )
        return json.dumps(
            {
                "source_path": str(input_path),
                "transcript_path": str(transcript_path),
                "asr_transcript_path": str(asr_markdown_path),
                "asr_markdown_path": str(asr_markdown_path),
                "asr_text_path": str(asr_text_path),
                "processing_note": processing_note,
                "recording_metadata": recording_metadata,
            },
            ensure_ascii=False,
            indent=2,
        )

    def check_asr_progress(self, args: dict[str, Any]) -> str:
        raw_path = str(args.get("input_path") or args.get("asr_output_dir") or "").strip()
        if not raw_path:
            raise ValueError("缺少 input_path 或 asr_output_dir。")
        target = self.workspace.resolve(raw_path)
        if not target.exists():
            raise FileNotFoundError(f"断点检查路径不存在：{raw_path}")
        script_path = (
            self.workspace.workspace_root
            / "meeting_audio_minutes"
            / "scripts"
            / "check_asr_progress.py"
        )
        python = project_agent_python(self.workspace.workspace_root) or Path(sys.executable)
        result = run_logged_process(
            [str(python), str(script_path), str(target)],
            cwd=self.workspace.workspace_root,
            timeout_seconds=120,
            label="ASR断点检查",
            check=False,
        )
        return json.dumps(
            {
                "ok": result.returncode == 0,
                "input_path": str(target),
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
            ensure_ascii=False,
            indent=2,
        )

    def _resolve_input_path(self, args: dict[str, Any]) -> Path:
        raw_path = str(
            args.get("input_path")
            or args.get("transcript_path")
            or args.get("audio_path")
            or ""
        ).strip()
        if not raw_path:
            raise ValueError(
                "缺少 input_path。请提供拖入后的录音/转写文本路径，或先让用户拖入会议录音。"
            )
        path = self.workspace.resolve(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"输入文件不存在：{raw_path}")
        return path

    def _ensure_transcript(self, input_path: Path, args: dict[str, Any]) -> tuple[Path, str]:
        extension = input_path.suffix.lower()
        if extension in TEXT_INPUT_EXTENSIONS:
            return input_path, "已使用现有转写文本，未重新进行音频转写。"
        if extension not in AUDIO_INPUT_EXTENSIONS:
            raise ValueError(
                f"暂不支持的会议输入类型：{extension or '无扩展名'}。请提供常见音频或 md/txt 转写文本。"
            )

        asr_root = self.workspace.resolve(
            str(args.get("asr_output_dir") or f"meet_files/asr_full/{sanitize_filename(input_path.stem)}")
        )
        prepared_audio, preprocess_note = self._prepare_audio(input_path, asr_root, args)
        transcript_path = self._transcribe_with_qwen3(prepared_audio, asr_root, args)
        note = (
            "已按本地链路处理音频：FFmpeg 降噪/标准化 -> VAD 边界分块 -> Qwen3-ASR 中文识别 -> 双版本会议纪要生成。"
            f"\n预处理：{preprocess_note}"
            f"\n转写文本：{transcript_path}"
        )
        return transcript_path, note

    def _prepare_audio(self, input_path: Path, asr_root: Path, args: dict[str, Any]) -> tuple[Path, str]:
        denoise_backend = str(args.get("denoise_backend") or "ffmpeg")
        # The managed Python 3.12 environment intentionally excludes
        # DeepFilterNet's incompatible Python 3.10 / torch stack. Normalize
        # legacy saved calls to the supported FFmpeg path.
        if denoise_backend not in {"ffmpeg", "none"}:
            denoise_backend = "ffmpeg"
        audio_work_dir = asr_root / "audio"
        meeting_audio_root = self.workspace.workspace_root / "meeting_audio_minutes"
        sys.path.insert(0, str(meeting_audio_root))
        old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = runtime_search_path(
            self.workspace.workspace_root, old_path
        )
        try:
            from meeting_minutes.audio import preprocess_audio

            prepared = preprocess_audio(
                input_path,
                audio_work_dir,
                denoise_backend=denoise_backend,
                use_postfilter=bool(args.get("use_denoise_postfilter", True)),
                sample_rate=int(args.get("sample_rate") or 16000),
                deepfilter_timeout_seconds=int(args.get("deepfilter_timeout_seconds") or 600),
                ffmpeg_timeout_seconds=int(args.get("audio_preprocess_timeout_seconds") or 1200),
            )
            note = f"{prepared.denoise_backend}; {prepared.filter_chain}"
            if prepared.warning:
                note = f"{note}; {prepared.warning}"
            return prepared.path, note
        except Exception as error:
            if denoise_backend == "deepfilter":
                raise
            fallback_path = audio_work_dir / f"{input_path.stem}.standardized_16k.wav"
            audio_work_dir.mkdir(parents=True, exist_ok=True)
            self._run_process(
                [
                    self._require_executable("ffmpeg", "FFmpeg 未安装，无法预处理会议录音。"),
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(input_path),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-af",
                    "highpass=f=80,lowpass=f=7800,afftdn=nf=-25,dynaudnorm=f=150:g=15,loudnorm=I=-16:LRA=11:TP=-1.5",
                    str(fallback_path),
                ],
                timeout_seconds=600,
                label="音频预处理",
            )
            return fallback_path, f"ffmpeg-fallback; DeepFilter/预处理模块不可用：{compact_error(error)}"
        finally:
            os.environ["PATH"] = old_path
            try:
                sys.path.remove(str(meeting_audio_root))
            except ValueError:
                pass

    def _write_public_asr_outputs(
        self,
        *,
        output_dir: Path,
        safe_name: str,
        meeting_name: str,
        transcript_path: Path,
        transcript: str,
    ) -> tuple[Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        asr_markdown_path = output_dir / f"{safe_name}_会议沟通内容整理_ASR转写稿_Qwen3.md"
        asr_text_path = output_dir / f"{safe_name}_会议沟通内容整理_ASR转写稿_Qwen3.txt"

        if transcript_path.resolve() != asr_markdown_path.resolve():
            markdown = self._render_public_asr_markdown(
                meeting_name=meeting_name,
                transcript_path=transcript_path,
                transcript=transcript,
            )
            self.workspace.write_text({"path": str(asr_markdown_path), "content": markdown})
        else:
            asr_markdown_path = transcript_path

        if transcript_path.resolve() != asr_text_path.resolve():
            self.workspace.write_text({"path": str(asr_text_path), "content": transcript.strip() + "\n"})
        else:
            asr_text_path = transcript_path
        return asr_markdown_path, asr_text_path

    def _render_public_asr_markdown(
        self,
        *,
        meeting_name: str,
        transcript_path: Path,
        transcript: str,
    ) -> str:
        markdown_source_path = transcript_path
        if transcript_path.suffix.lower() != ".md":
            sibling_markdown = transcript_path.with_suffix(".md")
            if sibling_markdown.is_file():
                markdown_source_path = sibling_markdown

        if markdown_source_path.suffix.lower() == ".md" and markdown_source_path.is_file():
            body = markdown_source_path.read_text(encoding="utf-8", errors="replace").strip()
        else:
            body = transcript.strip()

        if not body:
            body = "（转写正文为空，需检查 ASR 输出。）"

        source_lines = [f"- 原始转写路径：`{transcript_path}`"]
        if markdown_source_path != transcript_path:
            source_lines.append(f"- 完整Markdown来源：`{markdown_source_path}`")
        source_lines.append(
            "- 说明：本文件为录音转文本的标准命名完整副本，供前端文件库和后续会议纪要整理直接读取。"
        )
        return (
            f"# {meeting_name} ASR转写稿（Qwen3）\n\n"
            + "\n".join(source_lines)
            + "\n\n---\n\n"
            + body
            + "\n"
        )

    def _write_archive_manifest(
        self,
        *,
        archive_dir: Path,
        safe_name: str,
        meeting_name: str,
        source_path: Path,
        transcript_path: Path,
        asr_markdown_path: Path,
        asr_text_path: Path,
        internal_path: Path,
        work_markdown_path: Path,
        work_docx_path: Path,
        processing_note: str,
        supplemental_paths: list[str],
        recording_metadata: dict[str, Any],
    ) -> Path:
        workspace_root = self.workspace.workspace_root
        manifest_path = archive_dir / "manifest.json"
        now = int(time.time())
        created_at = now
        meeting_time: dict[str, Any] | None = None
        if manifest_path.is_file():
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
                created_at = int(existing.get("created_at") or now)
                if isinstance(existing.get("meeting_time"), dict):
                    meeting_time = existing["meeting_time"]
            except Exception:
                created_at = now
        manifest = {
            "schema_version": 1,
            "meeting_id": safe_name,
            "title": meeting_name,
            "archive_dir": workspace_relative_path(archive_dir, workspace_root),
            "created_at": created_at,
            "updated_at": now,
            "source_path": workspace_relative_path(source_path, workspace_root),
            "transcript_path": workspace_relative_path(transcript_path, workspace_root),
            "processing_note": processing_note,
            "recording_metadata": recording_metadata,
            "canonical_outputs": {
                "asr": workspace_relative_path(asr_markdown_path, workspace_root),
                "internal": workspace_relative_path(internal_path, workspace_root),
                "work_md": workspace_relative_path(work_markdown_path, workspace_root),
                "work_docx": workspace_relative_path(work_docx_path, workspace_root),
            },
            "supporting_outputs": {
                "asr_text": workspace_relative_path(asr_text_path, workspace_root),
            },
            "supplemental_paths": [
                workspace_relative_path(self.workspace.resolve(path), workspace_root)
                for path in supplemental_paths
                if str(path).strip()
            ],
        }
        if meeting_time is not None:
            manifest["meeting_time"] = meeting_time
        self.workspace.write_text(
            {
                "path": str(manifest_path),
                "content": json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            }
        )
        return manifest_path

    def _transcribe_with_qwen3(self, audio_path: Path, asr_root: Path, args: dict[str, Any]) -> Path:
        output_root = asr_root / "qwen3"
        script_path = self.workspace.workspace_root / "meeting_audio_minutes" / "scripts" / "transcribe_qwen3_asr_chunked.py"
        if not script_path.is_file():
            raise FileNotFoundError(f"未找到本地 Qwen3-ASR 脚本：{script_path}")
        cache_dir = self.workspace.workspace_root / "meeting_audio_minutes" / "model_cache"
        asr_settings = load_asr_settings(self.workspace.workspace_root)
        asr_backend = "mlx"
        asr_model_id = str(
            args.get("asr_model_id")
            or asr_settings.get("model_id")
            or default_asr_model_id("qwen3-asr-mlx-8bit", workspace_root=self.workspace.workspace_root)
        )
        command = [
            str(self._local_asr_python()),
            str(script_path),
            str(audio_path),
            "--output-dir",
            str(output_root),
            "--backend",
            asr_backend,
            "--model-id",
            asr_model_id,
            "--cache-dir",
            str(cache_dir),
            "--device",
            "mlx-metal",
            "--language",
            str(args.get("asr_language") or "Chinese"),
            "--chunk-mode",
            "vad",
            "--chunk-seconds",
            str(int(args.get("chunk_seconds") or 120)),
            "--max-new-tokens",
            str(int(args.get("max_new_tokens") or 2048)),
            "--workers",
            "1",
            "--skip-existing",
        ]
        self._run_process(
            command,
            timeout_seconds=int(args.get("transcription_timeout_seconds") or 14400),
            label="本地 Qwen3-ASR 转写",
        )
        candidates = sorted(
            output_root.rglob("transcript.txt"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError("Qwen3-ASR 已运行，但没有生成 transcript.txt。")
        return candidates[0]

    def _local_asr_python(self) -> Path:
        candidate = project_agent_python(self.workspace.workspace_root)
        if candidate is not None:
            return candidate
        raise FileNotFoundError(
            "项目唯一 Python 环境不存在。请运行：scripts/runtime_env.sh bootstrap"
        )

    def _require_executable(self, name: str, message: str) -> str:
        executable = find_runtime_executable(name, self.workspace.workspace_root)
        if not executable:
            raise FileNotFoundError(message)
        return executable

    def _run_process(self, command: list[str], *, timeout_seconds: int, label: str) -> None:
        try:
            run_logged_process(
                command,
                cwd=self.workspace.workspace_root,
                timeout_seconds=timeout_seconds,
                label=label,
                check=True,
            )
        except subprocess.TimeoutExpired as error:
            raise TimeoutError(f"{label}超时，请缩短录音或稍后重试。") from error
        except subprocess.CalledProcessError as error:
            output = "\n".join(part for part in (error.stdout, error.stderr) if part).strip()
            output = output[-2000:] if output else "无详细输出"
            raise RuntimeError(f"{label}失败：{output}") from error

    def _export_work_docx(self, *, markdown_path: Path, output_path: Path, title: str) -> None:
        script_path = self.workspace.workspace_root / "work_agent_core" / "docx_exporter.py"
        self._run_process(
            [
                str(self._office_python()),
                str(script_path),
                "--markdown-path",
                str(markdown_path),
                "--output-path",
                str(output_path),
                "--title",
                title,
            ],
            timeout_seconds=300,
            label="工作提交版DOCX生成",
        )

    def _office_python(self) -> Path:
        agent_python = project_agent_python(self.workspace.workspace_root)
        candidates = [
            os.getenv("WORK_AGENT_OFFICE_PYTHON"),
            str(agent_python) if agent_python else None,
            str(Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"),
            sys.executable,
        ]
        for candidate in candidates:
            if candidate and Path(candidate).is_file():
                return Path(candidate)
        return Path(sys.executable)

    def _load_supplements(self, paths: list[str]) -> str:
        chunks = []
        for raw_path in paths:
            path = self.workspace.resolve(raw_path)
            if path.is_file():
                text = path.read_text(encoding="utf-8", errors="replace")
                chunks.append(f"## {path.name}\n\n{text[:12000]}")
        return "\n\n".join(chunks)

    def _load_minutes_spec(self, args: dict[str, Any]) -> str:
        base_spec = self.workspace.resolve(str(args.get("spec_path") or SPEC_PATH)).read_text(encoding="utf-8")
        skill_path = self.workspace.resolve(str(args.get("skill_spec_path") or SKILL_SPEC_PATH))
        sections = [base_spec]
        if skill_path.is_file():
            skill_spec = skill_path.read_text(encoding="utf-8", errors="replace")
            sections.append(f"# meeting-minutes skill instructions\n\n{skill_spec}")
        custom_instructions = str(args.get("custom_instructions") or "").strip()
        if custom_instructions:
            sections.append(f"# 用户会议纪要设置\n\n{custom_instructions}")
        return "\n\n".join(sections)

    def _generate_internal_minutes(
        self,
        *,
        meeting_name: str,
        transcript: str,
        confirmed_info: str,
        external_research: str,
        supplements: str,
        spec: str,
        processing_note: str,
        recording_context: str,
    ) -> str:
        prompt = (
            "请根据规范生成“内部留档版”会议沟通内容整理。"
            "这版给用户自己看，必须比提交版更完整，便于后续复盘、追问和回听校对。"
            "可以记录不确定内容，但必须标清确定性；不要把不确定金额、人名、模型参数写成确定事实。"
            "如提供了公开检索/外部资料，只能作为背景核验和专名纠错参考，不要覆盖会议事实。"
            "公开资料需单独成节，说明来源支持了什么、哪些会议内数字或合作事项仍属公开渠道未核验。"
            "输出为Markdown，建议包含：会议基本信息、对方情况、交流要点、合作线索、"
            "公开资料核验与背景补充、待核实信息、转写不确定点。"
            "内部留档版可以出现“ASR/转写不确定”等处理痕迹。\n\n"
            f"会议名称：{meeting_name}\n\n"
            f"用户确认信息：\n{confirmed_info or '无'}\n\n"
            f"公开检索/外部资料：\n{external_research or '无'}\n\n"
            f"补充材料：\n{supplements or '无'}\n\n"
            f"本地处理记录：\n{processing_note or '无'}\n\n"
            f"录音时间元数据：\n{recording_context or '未读取到内嵌录音开始时间'}\n\n"
            f"规范：\n{spec[:16000]}\n\n"
            f"ASR转写文本：\n{transcript[:50000]}"
        )
        return self._chat_markdown(prompt)

    def _generate_work_minutes(
        self,
        *,
        meeting_name: str,
        internal_minutes: str,
        confirmed_info: str,
        spec: str,
        recording_context: str,
    ) -> str:
        prompt = (
            "请根据规范和内部留档版，生成“工作提交版”会议纪要Markdown，随后会被导出为Word。"
            "该Word应遵循用户配置的正式会议纪要写法："
            "正式、克制、像人工会后整理，不像逐字稿、尽调报告或AI分析。"
            "结构固定为：标题、一个开篇概述段、四个中文编号章节："
            "一、会议基本情况；二、对方单位基本情况；三、双方交流情况；四、初步研判意见。"
            "第二节标题要按已确认的会议对象改写，例如“二、合作单位基本情况”。"
            "正文使用连续正式段落，不使用表格，不使用项目符号，不写“待核实”“需补充”“ASR显示”“转写文本显示”。"
            "用户是新入职旁听人员，不是主持人，不要写责任方、任务分派、要求对方补材料。"
            "对于未经确认的人名、金额、具体承诺、投资事项、时间节点，宁可不写或弱化为“会议中提到/围绕...进行了交流”。"
            "媒体 creation_time 只证明录音文件记录的开始时间，不必然等于会议正式开始时间。"
            "当用户确认信息或转写明确给出会议时间时，以其为准；否则只能保守写录音日期，或明确写作录音开始时间。"
            "最后一节可以给出审慎初步研判，末句可以用Markdown加粗突出建议，但不能冒进。"
            "Markdown中标题用一级标题，四个章节用二级标题；不要输出代码块。\n\n"
            f"会议名称：{meeting_name}\n\n"
            f"用户确认信息：\n{confirmed_info or '无'}\n\n"
            f"录音时间元数据：\n{recording_context or '未读取到内嵌录音开始时间'}\n\n"
            f"规范：\n{spec[:16000]}\n\n"
            f"内部留档版：\n{internal_minutes[:50000]}"
        )
        return self._chat_markdown(prompt)

    def _chat_markdown(self, prompt: str) -> str:
        response = self.client.chat(
            [
                {
                    "role": "system",
                    "content": meeting_minutes_system_prompt(self.workspace.workspace_root),
                },
                {"role": "user", "content": prompt},
            ],
            profile=self.profile,
        )
        text = response.content.strip()
        if not text:
            raise RuntimeError("Meeting minutes skill returned empty content.")
        return text + "\n"


def register_meeting_minutes_skill(
    registry: ToolRegistry,
    *,
    workspace_root: str | Path,
    client: OpenAICompatibleClient,
    profile: ModelProfile,
) -> None:
    skill = MeetingMinutesSkill(workspace_root=workspace_root, client=client, profile=profile)
    registry.register(
        Tool(
            name="check_meeting_asr_progress",
            description=(
                "只读检查会议录音或ASR输出目录的分块断点。路径作为结构化参数传入，"
                "支持文件名中的空格，不需要 shell_exec 或终端审批。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": "会议录音路径或已有ASR输出目录。",
                    },
                    "asr_output_dir": {
                        "type": "string",
                        "description": "input_path 的兼容别名。",
                    },
                },
            },
            handler=skill.check_asr_progress,
        )
    )
    registry.register(
        Tool(
            name="transcribe_meeting_audio",
            description=(
                "底层会议音频转写工具。输入拖入后的会议录音路径，执行本地音频预处理/降噪、VAD分块和Qwen3-ASR中文识别，"
                "只返回转写文本路径和处理记录，不生成会议纪要。长录音中断后可再次调用；底层Qwen3命令带 --skip-existing，"
                "会复用已完成的 items/chunk_* 结果并只补缺失分块。调用前优先调用 "
                "check_meeting_asr_progress 查询断点，不要为此运行 shell_exec。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": "拖入附件后的音频路径，或已有md/txt转写文本路径。",
                    },
                    "output_dir": {"type": "string", "default": "meet_files"},
                    "asr_output_dir": {"type": "string"},
                    "denoise_backend": {
                        "type": "string",
                        "enum": ["ffmpeg", "none"],
                        "default": "ffmpeg",
                    },
                    "asr_profile": {
                        "type": "string",
                        "enum": ["qwen3-asr-mlx-8bit"],
                        "default": "qwen3-asr-mlx-8bit",
                    },
                    "asr_model_id": {
                        "type": "string",
                        "default": QWEN3_MLX_LOCAL_MODEL_ID,
                        "description": "可选的 Qwen3-ASR 模型ID或本地快照路径。默认使用项目本地 MLX 8bit。",
                    },
                    "asr_device": {
                        "type": "string",
                        "default": "mlx-metal",
                        "description": "ASR设备。项目统一使用 MLX Metal。",
                    },
                    "chunk_seconds": {
                        "type": "integer",
                        "default": 120,
                        "description": "Qwen3-ASR 的 VAD 合并分块目标长度，默认 120 秒。",
                    },
                    "asr_workers": {
                        "type": "integer",
                        "default": 1,
                        "description": "Qwen3-ASR MLX 后端固定为1个 Metal worker。",
                    },
                    "asr_backend": {
                        "type": "string",
                        "enum": ["mlx"],
                        "default": "mlx",
                        "description": "Qwen3-ASR 推理后端。项目统一使用 mlx 8bit。",
                    },
                    "hotword": {
                        "type": "string",
                        "description": "空格分隔的ASR热词，例如公司名、人名、项目名、行业术语。",
                    },
                    "hotwords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "ASR热词列表，只放用户明确提供或转写/材料中已出现的公司名、人名、项目名、行业术语。"
                            "不要根据录音文件名编造参会方、会议对象或合作关系。"
                        ),
                    },
                    "confirmed_info": {
                        "type": "string",
                        "description": "用户确认过的会议信息，可从中抽取热词并用于后续纪要。",
                    },
                    "deepfilter_timeout_seconds": {
                        "type": "integer",
                        "default": 600,
                        "description": "DeepFilterNet 自动降噪最长等待秒数。auto 模式超时后会降级到 ffmpeg 预处理。",
                    },
                    "audio_preprocess_timeout_seconds": {
                        "type": "integer",
                        "default": 1200,
                        "description": "ffmpeg 音频转换/预处理最长等待秒数。",
                    },
                },
                "required": ["input_path"],
            },
            handler=skill.transcribe_audio,
        )
    )
    registry.register(
        Tool(
            name="generate_meeting_minutes",
            description=(
                "兼容旧接口的一键会议纪要工具（保留既有调用，不作为模块化新流程）。输入可以是拖入后的会议录音路径或ASR转写文本路径；"
                "音频会先走本地预处理/降噪、VAD分块和Qwen3-ASR中文识别，再生成内部留档版Markdown、工作提交版Markdown和工作提交版DOCX。"
                "音频转写阶段会使用 --skip-existing 复用已有分块结果，适合中断后续跑。"
                "新流程按 meeting-minutes 技能形成内容，再按需打开 official-document，最终由完整 docx 技能生成和验收 Word。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": "拖入附件后的音频路径，或已有md/txt转写文本路径。",
                    },
                    "transcript_path": {
                        "type": "string",
                        "description": "兼容旧参数：已有ASR转写文本路径。",
                    },
                    "audio_path": {
                        "type": "string",
                        "description": "兼容旧参数：会议录音路径。",
                    },
                    "output_dir": {"type": "string", "default": "meet_files"},
                    "meeting_name": {"type": "string"},
                    "confirmed_info": {"type": "string"},
                    "external_research": {
                        "type": "string",
                        "description": (
                            "公开检索/AnySearch核验笔记。只用于内部留档版的背景补充和专名纠错，"
                            "不要把搜索推断直接写进工作提交版DOCX。"
                        ),
                    },
                    "research_notes": {
                        "type": "string",
                        "description": "external_research 的兼容别名。",
                    },
                    "hotword": {
                        "type": "string",
                        "description": "空格分隔的ASR热词，例如公司名、人名、项目名、行业术语。",
                    },
                    "hotwords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "ASR热词列表，只放用户明确提供或转写/材料中已出现的公司名、人名、项目名、行业术语。"
                            "不要根据录音文件名编造参会方、会议对象或合作关系。"
                        ),
                    },
                    "supplemental_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "denoise_backend": {
                        "type": "string",
                        "enum": ["ffmpeg", "none"],
                        "default": "ffmpeg",
                    },
                    "asr_profile": {
                        "type": "string",
                        "enum": ["qwen3-asr-mlx-8bit"],
                        "default": "qwen3-asr-mlx-8bit",
                    },
                    "asr_model_id": {
                        "type": "string",
                        "default": QWEN3_MLX_LOCAL_MODEL_ID,
                        "description": "可选的 Qwen3-ASR 模型ID或本地快照路径。默认使用项目本地 MLX 8bit。",
                    },
                    "chunk_seconds": {
                        "type": "integer",
                        "default": 120,
                        "description": "Qwen3-ASR 的 VAD 合并分块目标长度，默认 120 秒。",
                    },
                    "asr_workers": {
                        "type": "integer",
                        "default": 1,
                        "description": "Qwen3-ASR MLX 后端固定为1个 Metal worker。",
                    },
                    "asr_backend": {
                        "type": "string",
                        "enum": ["mlx"],
                        "default": "mlx",
                        "description": "Qwen3-ASR 推理后端。项目统一使用 mlx 8bit。",
                    },
                    "deepfilter_timeout_seconds": {
                        "type": "integer",
                        "default": 600,
                        "description": "DeepFilterNet 自动降噪最长等待秒数。auto 模式超时后会降级到 ffmpeg 预处理。",
                    },
                    "audio_preprocess_timeout_seconds": {
                        "type": "integer",
                        "default": 1200,
                        "description": "ffmpeg 音频转换/预处理最长等待秒数。",
                    },
                },
                "required": ["input_path"],
            },
            handler=skill.run,
        )
    )


def sanitize_filename(name: str) -> str:
    cleaned = "".join(char if char not in '/\\:*?"<>|' else "_" for char in name).strip()
    return cleaned or "meeting"


def workspace_relative_path(path: Path, workspace_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(workspace_root.resolve()))
    except ValueError:
        return str(path.resolve())


def compact_error(error: Exception, *, limit: int = 220) -> str:
    text = " ".join(str(error).split())
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}..."


def load_asr_settings(workspace_root: Path) -> dict[str, Any]:
    path = workspace_root / ASR_SETTINGS_PATH
    data: dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except json.JSONDecodeError:
            data = {}
    settings = {**DEFAULT_ASR_SETTINGS, **data}
    settings["profile"] = normalize_asr_profile(str(settings.get("profile") or "qwen3-asr"))
    settings["model_id"] = str(
        settings.get("model_id")
        or default_asr_model_id(settings["profile"], workspace_root=workspace_root)
    )
    settings["backend"] = "mlx"
    settings["hotwords"] = str(settings.get("hotwords") or DEFAULT_ASR_HOTWORDS)
    settings["agent_work_background"] = str(load_agent_settings(workspace_root).get("work_background") or "")
    return settings


def load_agent_settings(workspace_root: Path) -> dict[str, Any]:
    path = workspace_root / AGENT_SETTINGS_PATH
    data: dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except json.JSONDecodeError:
            data = {}
    return {"work_background": str(data.get("work_background") or "").strip()}


def meeting_minutes_system_prompt(workspace_root: Path) -> str:
    work_background = str(load_agent_settings(workspace_root).get("work_background") or "").strip()
    prompt = (
        "你是严谨的中文会议纪要整理助手。只输出Markdown正文。"
        "公开检索资料只能用于内部留档版的背景核验和专名纠错；"
        "工作提交版必须以用户确认信息、会议转写和附件材料为主，保持保守。"
    )
    if work_background:
        prompt += (
            "\n\n用户长期工作背景/常用系统提示词：\n"
            f"{work_background}\n\n"
            "使用规则：该背景用于默认工作语境、称谓口径、专名纠错和文档写作风格；"
            "它不是某一场会议已经发生或对方已经确认的事实。正式材料仍必须以用户确认信息、会议转写和附件材料为依据。"
        )
    return prompt


def normalize_asr_profile(profile: str) -> str:
    profile = profile.strip().lower()
    if profile in {"qwen3-asr-mlx-8bit", "qwen3-mlx", "mlx", "mlx-8bit", "qwen3-asr", "qwen3"}:
        return "qwen3-asr-mlx-8bit"
    return "qwen3-asr-mlx-8bit"


def default_asr_model_id(profile: str, *, workspace_root: Path | None = None) -> str:
    root = workspace_root or Path.cwd()
    local_candidate = root / QWEN3_MLX_LOCAL_MODEL_ID
    if local_candidate.exists():
        return QWEN3_MLX_LOCAL_MODEL_ID
    return QWEN3_MLX_REMOTE_MODEL_ID


def build_asr_hotwords(args: dict[str, Any], *, settings: dict[str, Any] | None = None) -> str:
    terms: list[str] = []
    if settings:
        terms.extend(str(settings.get("hotwords") or "").replace("\n", " ").split())
        terms.extend(extract_domain_terms(str(settings.get("agent_work_background") or "")))
    else:
        terms.extend(DEFAULT_ASR_HOTWORDS.split())
    raw_hotword = str(args.get("hotword") or "").strip()
    if raw_hotword:
        terms.extend(raw_hotword.split())
    raw_hotwords = args.get("hotwords") or []
    if isinstance(raw_hotwords, list):
        terms.extend(str(term).strip() for term in raw_hotwords)
    for key in ("meeting_name", "confirmed_info"):
        terms.extend(extract_domain_terms(str(args.get(key) or "")))

    seen: set[str] = set()
    cleaned: list[str] = []
    for term in terms:
        normalized = term.strip(" ，。、“”‘’（）()[]【】;；:：\n\t")
        if len(normalized) < 2 or normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(normalized)
    return " ".join(cleaned[:160])


def extract_domain_terms(text: str) -> list[str]:
    if not text.strip():
        return []
    terms: list[str] = []
    for match in re.findall(r"[A-Za-z][A-Za-z0-9_+.-]{1,31}", text):
        terms.append(match)
    for match in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,24}", text):
        if len(match) <= 1:
            continue
        if match in {"这个", "那个", "然后", "就是", "我们", "他们", "你们", "进行", "相关"}:
            continue
        terms.append(match)
    return terms
