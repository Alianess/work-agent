from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from http.cookies import SimpleCookie
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, quote, urlparse
import urllib.error
import urllib.request
import atexit
import argparse
import array
import base64
import hashlib
import json
import math
import mimetypes
import os
import re
import select
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
import wave

from .auth import AuthStore, AuthUser, SESSION_TTL_SECONDS
from .cli import (
    add_model_profile,
    build_default_tools,
    delete_model_profile,
    set_default_profile,
    update_model_profile,
    update_model_profile_api_key_env,
)
from .audio_metadata import probe_audio_metadata
from .config import (
    DEFAULT_CONFIG_PATH,
    ModelProfile,
    ModelRegistry,
    api_key_env_for_profile,
    delete_env_value,
    save_env_value,
)
from .debug_trace import DebugTrace, list_debug_traces
from .cross_chat_memory import CrossChatMemoryStore
from .llm import OpenAICompatibleClient, chat_completions_endpoint, normalize_reasoning_effort
from .memory import (
    CHAT_SUMMARY_TRIGGER_TOKENS,
    prepare_session_memory,
)
from .history_recall import render_history_recall_system_context
from .office_preview import OFFICE_TO_PDF_EXTENSIONS, convert_office_to_pdf
from .react import (
    DEFAULT_MAX_STEPS,
    AgentCancelled,
    ReActAgent,
    contains_tool_call_markup,
    strip_tool_call_markup,
)
from .runtime_env import (
    find_runtime_executable,
    project_agent_python,
    runtime_contract_status,
)
from .session_store import SessionStore, repair_runtime_message_sequence, sanitize_conversation_id
from .skills.meeting_minutes import MeetingMinutesSkill
from .skill_runtime import load_skill_manifests
from .tools import WorkspaceFiles
from .turn_runtime import TurnCancelled, TurnRuntime
from .turn_store import TurnStore, sanitize_turn_id


WORKSPACE_ROOT = Path.cwd().resolve()
CONFIG_PATH = DEFAULT_CONFIG_PATH
STATIC_DIR = Path("web_frontend/dist")
ASR_SETTINGS_PATH = Path("config/asr_settings.json")
AGENT_SETTINGS_PATH = Path("config/agent_settings.json")
AUTH_DATABASE_PATH = Path("config/auth.sqlite3")
AUTH_COOKIE_NAME = "work_agent_session"
USER_DATA_ROOT = Path("meet_files/users")
QWEN3_MLX_REMOTE_MODEL_ID = "mlx-community/Qwen3-ASR-1.7B-8bit"
QWEN3_MLX_LOCAL_MODEL_ID = "meeting_audio_minutes/model_cache/mlx-community/Qwen3-ASR-1.7B-8bit"
ASR_WORKER: LocalQwen3ASRWorker | None = None
ASR_WORKER_LIFECYCLE_LOCK = threading.Lock()
ASR_WORKER_IDLE_TIMER: threading.Timer | None = None
ASR_ACTIVE_REQUESTS = 0
VAD_WORKER: LocalWebRtcVadWorker | None = None
CLIENT_DISCONNECT_ERRORS = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)
ACTIVE_CHAT_CONVERSATIONS: set[tuple[int, str]] = set()
ACTIVE_CHAT_CONVERSATIONS_LOCK = threading.Lock()
PENDING_CONVERSATION_TITLE = "待命名对话"
CONVERSATION_HISTORY_PATH = Path("meet_files/conversation_history/conversations.json")
PROJECTS_ROOT = Path("meet_files/projects")
PROJECT_ID_PATTERN = re.compile(r"^project-[a-f0-9]{12}$")
DEFAULT_ASR_SETTINGS: dict[str, Any] = {
    "profile": "qwen3-asr-mlx-8bit",
    "model_id": QWEN3_MLX_LOCAL_MODEL_ID,
    "backend": "mlx",
    "hotwords": (
        "会议 客户 产品 项目 技术 方案 数据 模型 合作 产业 平台 机器人 "
        "具身智能 智能座舱 工业 教育 搬运 上料 SMT 料盘 断纱 接纱 "
        "同执合 橡豫智能 国先控股 中科灵鉴"
    ),
}
DEFAULT_WORK_BACKGROUND = (
    "我方为合肥国先控股有限公司机器人事业部，日常对接国先中心（合肥）/"
    "国际先进技术应用推进中心（合肥）相关工作。国先控股是合肥市产业投资控股（集团）"
    "有限公司体系内全资企业，承担国先中心（合肥）相关项目的市场化运营、产业资源导入、"
    "场景拓展、平台建设运营和项目落地推进等工作。\n"
    "会议纪要中，“我方”“平台方”通常指合肥国先控股有限公司、国先中心（合肥）及相关"
    "机器人事业部工作口径。正式材料中应优先使用“国先控股”“合肥国先控股”“国先中心（合肥）”"
    "等确认名称，不要将 ASR 相近词误写为国研、国元、国轩、国信、国现等。\n"
    "当前重点工作围绕合肥具身智能和智能机器人产业推进展开，包括智能机器人公共服务平台、"
    "具身智能机器人共性平台服务公司、机器人场景训练、零部件生产加工、产品测试验证、"
    "真实场景应用、场景招商、场景实验室、产业生态协同、解决方案团队储备、二次开发团队对接、"
    "平台化运营和项目落地等。\n"
    "整理会议纪要时，内部留档版可以保留 ASR 不确定项、口述信息、待核验内容和材料差异；"
    "工作提交版 DOCX 必须保守、简洁、正式，只写高置信和已确认内容，不写 ASR 痕迹、不写待确认事项、"
    "不写不确定人名、地点、金额、股权、参数或合作条款。会议地点表述优先使用“赴X开展座谈沟通”"
    "或“与X开展座谈沟通”，其中 X 应为确认的会议对象或项目名称。"
)
DEFAULT_AGENT_SETTINGS: dict[str, Any] = {
    "work_background": DEFAULT_WORK_BACKGROUND,
    "company_document_format": (
        "页面设置：上3.5厘米、下3.1厘米、左2.65厘米、右2.65厘米\n"
        "行间距：固定值29.6磅\n"
        "标题：2号字，方正小标宋简体\n"
        "正文：3号字，仿宋_GB2312\n"
        "一级标题：黑体\n"
        "二级标题：楷体_GB2312"
    ),
}
DEFAULT_DISABLED_SKILL_IDS = {"baidu-web-search", "tavily-search", "edge-browser"}

AUTH_STORE: AuthStore | None = None
REQUEST_AUTH = threading.local()
USER_STORES_LOCK = threading.RLock()
USER_SESSION_STORES: dict[int, SessionStore] = {}
USER_TURN_STORES: dict[int, TurnStore] = {}
TEMP_SYNC_LOCK = threading.RLock()
TEMP_SYNC_FILE_TTL_SECONDS = 60 * 60
TEMP_SYNC_MAX_FILE_BYTES = 100 * 1024 * 1024
TEMP_SYNC_MAX_TEXT_CHARS = 200_000
TEMP_SYNC_FILE_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")


def get_auth_store() -> AuthStore:
    global AUTH_STORE
    if AUTH_STORE is None:
        store = AuthStore(WORKSPACE_ROOT / AUTH_DATABASE_PATH)
        admin = store.ensure_admin(
            os.getenv("WORK_AGENT_ADMIN_USERNAME", "admin"),
            os.getenv("WORK_AGENT_ADMIN_PASSWORD", "admin123"),
        )
        migrate_legacy_admin_data(admin)
        AUTH_STORE = store
    return AUTH_STORE


def current_auth_user() -> AuthUser:
    user = getattr(REQUEST_AUTH, "user", None)
    if isinstance(user, AuthUser):
        return user
    admin = get_auth_store().get_user(os.getenv("WORK_AGENT_ADMIN_USERNAME", "admin"))
    if admin is None:
        raise RuntimeError("管理员账户尚未初始化")
    return admin


def user_data_dir(user: AuthUser | None = None) -> Path:
    account = user or current_auth_user()
    path = WORKSPACE_ROOT / USER_DATA_ROOT / f"u{account.id}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def user_conversation_dir(user: AuthUser | None = None) -> Path:
    return user_data_dir(user) / "conversation_history"


def user_agent_settings_path(user: AuthUser | None = None) -> Path:
    return user_data_dir(user) / "agent_settings.json"


def user_conversation_history_path(user: AuthUser | None = None) -> Path:
    return user_conversation_dir(user) / "conversations.json"


def account_workspace_root(user: AuthUser | None = None) -> Path:
    account = user or current_auth_user()
    if account.role == "admin":
        return WORKSPACE_ROOT
    root = user_data_dir(account) / "workspace"
    (root / "meet_files").mkdir(parents=True, exist_ok=True)
    return root


def account_relative_path(path: Path) -> str:
    return str(path.relative_to(account_workspace_root()))


def migrate_legacy_admin_data(admin: AuthUser) -> None:
    destination = WORKSPACE_ROOT / USER_DATA_ROOT / f"u{admin.id}"
    marker = destination / ".legacy_migrated"
    if marker.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    legacy_conversations = WORKSPACE_ROOT / "meet_files" / "conversation_history"
    target_conversations = destination / "conversation_history"
    if legacy_conversations.exists():
        shutil.copytree(legacy_conversations, target_conversations, dirs_exist_ok=True)
    legacy_agent_settings = WORKSPACE_ROOT / AGENT_SETTINGS_PATH
    target_agent_settings = destination / "agent_settings.json"
    if legacy_agent_settings.is_file() and not target_agent_settings.exists():
        shutil.copy2(legacy_agent_settings, target_agent_settings)
    marker.write_text(str(int(time.time())), encoding="utf-8")


def get_session_store() -> SessionStore:
    user = current_auth_user()
    with USER_STORES_LOCK:
        return USER_SESSION_STORES.setdefault(
            user.id,
            SessionStore(WORKSPACE_ROOT, session_dir=user_conversation_dir(user) / "sessions"),
        )


def get_turn_store() -> TurnStore:
    user = current_auth_user()
    with USER_STORES_LOCK:
        return USER_TURN_STORES.setdefault(
            user.id,
            TurnStore(WORKSPACE_ROOT, turn_dir=user_conversation_dir(user) / "turns"),
        )


class LocalQwen3ASRWorker:
    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root
        self.process: subprocess.Popen[str] | None = None
        self.lock = threading.Lock()
        self.request_id = 0
        self.ready = False

    def warm(self) -> None:
        with self.lock:
            self._ensure_ready()

    def transcribe(self, audio_path: Path, *, timeout_seconds: int = 180) -> dict[str, Any]:
        with self.lock:
            self._ensure_ready()
            self.request_id += 1
            request_id = f"speech-{self.request_id}"
            duration_seconds = max(0.0, wav_signal_stats(audio_path).get("duration_ms", 0) / 1000)
            job = {
                "index": self.request_id,
                "request_id": request_id,
                "chunk_path": str(audio_path),
                "source_audio": str(audio_path),
                "start_seconds": 0.0,
                "end_seconds": round(duration_seconds, 3),
                "start": "00:00:00",
                "end": format_seconds(duration_seconds),
                "chunk_mode": "realtime",
                "vad_speech_segment_count": 0,
                "vad_speech_segments_ms": [],
            }
            process = self._process_or_raise()
            assert process.stdin is not None
            process.stdin.write(json.dumps({"event": "transcribe", "job": job}, ensure_ascii=False) + "\n")
            process.stdin.flush()

            deadline = time.monotonic() + timeout_seconds
            while True:
                event = self._read_event(deadline)
                if event.get("event") != "result":
                    continue
                if not event.get("ok"):
                    raise RuntimeError(str(event.get("error") or "本地 Qwen3-ASR worker 识别失败"))
                row = event.get("row") or {}
                return {
                    "event": "result",
                    "id": request_id,
                    "ok": True,
                    "text": str(row.get("transcription") or "").strip(),
                    "row": row,
                    "elapsed_ms": int(float(row.get("infer_seconds") or 0) * 1000),
                }

    def stop(self) -> None:
        with self.lock:
            process = self.process
            self.process = None
            self.ready = False
            if not process or process.poll() is not None:
                return
            try:
                if process.stdin:
                    process.stdin.write(json.dumps({"event": "shutdown"}) + "\n")
                    process.stdin.flush()
            except Exception:
                pass
            try:
                process.terminate()
                process.wait(timeout=2)
            except Exception:
                process.kill()

    def _ensure_ready(self) -> None:
        process = self.process
        if process is not None and process.poll() is None and self.ready:
            return

        self._start_process()
        deadline = time.monotonic() + 120
        while True:
            event = self._read_event(deadline)
            if event.get("event") == "ready":
                if not event.get("ok", True):
                    raise RuntimeError(str(event.get("error") or "Qwen3-ASR worker 启动失败"))
                self.ready = True
                return

    def _start_process(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        python_path = local_asr_python()
        script_path = self.workspace_root / "meeting_audio_minutes" / "scripts" / "qwen3_asr_worker.py"
        if not script_path.is_file():
            raise FileNotFoundError(f"未找到本地 Qwen3-ASR worker：{script_path}")
        settings = load_asr_settings()
        self.process = subprocess.Popen(
            [
                str(python_path),
                str(script_path),
                "--model-id",
                str(settings["model_id"]),
                "--cache-dir",
                str(self.workspace_root / "meeting_audio_minutes" / "model_cache"),
                "--device",
                "mlx-metal",
                "--language",
                "zh",
                "--max-new-tokens",
                "1024",
                "--worker-id",
                "1",
            ],
            cwd=self.workspace_root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        self.ready = False

    def _process_or_raise(self) -> subprocess.Popen[str]:
        process = self.process
        if process is None or process.poll() is not None:
            raise RuntimeError("本地 Qwen3-ASR worker 未运行。")
        return process

    def _read_event(self, deadline: float) -> dict[str, Any]:
        process = self._process_or_raise()
        assert process.stdout is not None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("本地 Qwen3-ASR worker 等待超时。")
            ready, _, _ = select.select([process.stdout], [], [], min(remaining, 0.5))
            if not ready:
                if process.poll() is not None:
                    raise RuntimeError("本地 Qwen3-ASR worker 已退出。")
                continue
            line = process.stdout.readline()
            if not line:
                raise RuntimeError("本地 Qwen3-ASR worker 没有返回数据。")
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                return event


class LocalWebRtcVadWorker:
    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root
        self.process: subprocess.Popen[str] | None = None
        self.lock = threading.Lock()
        self.request_id = 0
        self.ready = False
        self.provider = "webrtcvad"

    def classify(
        self,
        frames_base64: list[str],
        *,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        aggressiveness: int = 3,
        timeout_seconds: int = 5,
    ) -> dict[str, Any]:
        with self.lock:
            self._ensure_ready()
            self.request_id += 1
            request_id = f"vad-{self.request_id}"
            request = {
                "id": request_id,
                "sample_rate": sample_rate,
                "frame_ms": frame_ms,
                "aggressiveness": aggressiveness,
                "frames_base64": frames_base64,
            }
            process = self._process_or_raise()
            assert process.stdin is not None
            process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            process.stdin.flush()

            deadline = time.monotonic() + timeout_seconds
            while True:
                event = self._read_event(deadline)
                if event.get("event") != "result":
                    continue
                if event.get("id") != request_id:
                    continue
                if not event.get("ok"):
                    raise RuntimeError(str(event.get("error") or "WebRTC VAD worker failed"))
                return event

    def stop(self) -> None:
        with self.lock:
            process = self.process
            self.process = None
            self.ready = False
            if not process or process.poll() is not None:
                return
            try:
                if process.stdin:
                    process.stdin.write(json.dumps({"event": "shutdown"}) + "\n")
                    process.stdin.flush()
            except Exception:
                pass
            try:
                process.terminate()
                process.wait(timeout=2)
            except Exception:
                process.kill()

    def _ensure_ready(self) -> None:
        process = self.process
        if process is not None and process.poll() is None and self.ready:
            return

        self._start_process()
        deadline = time.monotonic() + 10
        while True:
            event = self._read_event(deadline)
            if event.get("event") == "ready":
                self.ready = True
                self.provider = str(event.get("provider") or "webrtcvad")
                return
            if event.get("event") == "error":
                raise RuntimeError(str(event.get("error") or "WebRTC VAD worker failed to start"))

    def _start_process(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        python_path = local_vad_python()
        script_path = self.workspace_root / "work_agent_core" / "vad_worker.py"
        if not script_path.is_file():
            raise FileNotFoundError(f"未找到本地 VAD worker：{script_path}")
        self.process = subprocess.Popen(
            [str(python_path), "-u", str(script_path)],
            cwd=self.workspace_root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        self.ready = False

    def _process_or_raise(self) -> subprocess.Popen[str]:
        process = self.process
        if process is None or process.poll() is not None:
            raise RuntimeError("本地 WebRTC VAD worker 未运行。")
        return process

    def _read_event(self, deadline: float) -> dict[str, Any]:
        process = self._process_or_raise()
        assert process.stdout is not None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("本地 WebRTC VAD worker 等待超时。")
            ready, _, _ = select.select([process.stdout], [], [], min(remaining, 0.2))
            if not ready:
                if process.poll() is not None:
                    raise RuntimeError("本地 WebRTC VAD worker 已退出。")
                continue
            line = process.stdout.readline()
            if not line:
                raise RuntimeError("本地 WebRTC VAD worker 没有返回数据。")
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                return event


class WorkAgentHandler(SimpleHTTPRequestHandler):
    server_version = "WorkAgentWeb/0.1"

    def do_OPTIONS(self) -> None:
        self._send_empty(204)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/health":
                self._send_json(
                    {
                        "ok": True,
                        "auth_enabled": True,
                        "workspace": str(WORKSPACE_ROOT),
                        "config": str((WORKSPACE_ROOT / CONFIG_PATH).resolve()),
                        "runtime": runtime_contract_status(WORKSPACE_ROOT),
                    }
                )
                return
            if parsed.path == "/api/auth/me":
                user = self._authenticated_user()
                self._send_json({"authenticated": bool(user), "user": user.to_payload() if user else None})
                return
            if parsed.path.startswith("/api/") and not self._require_auth():
                return
            if parsed.path == "/api/models":
                self._send_json(models_payload())
                return
            if parsed.path == "/api/settings/asr":
                self._send_json(asr_settings_payload())
                return
            if parsed.path == "/api/settings/agent":
                self._send_json(agent_settings_payload())
                return
            if parsed.path == "/api/memories":
                params = parse_qs(parsed.query)
                project_values = params.get("project_id")
                project_id = first(params, "project_id", "") if project_values is not None else None
                self._send_json(
                    cross_chat_memories_payload(
                        project_id=project_id,
                        query=first(params, "query", ""),
                    )
                )
                return
            if parsed.path == "/api/tools":
                self._send_json(tools_payload())
                return
            if parsed.path == "/api/debug/traces":
                if not self._require_admin():
                    return
                params = parse_qs(parsed.query)
                limit = int(first(params, "limit", "200"))
                self._send_json(
                    list_debug_traces(
                        WORKSPACE_ROOT,
                        conversation_id=first(params, "conversation_id", "") or None,
                        trace_id=first(params, "trace_id", "") or None,
                        limit=limit,
                    )
                )
                return
            turn_route = parse_turn_route(parsed.path)
            if turn_route:
                turn_id, suffix = turn_route
                params = parse_qs(parsed.query)
                if suffix == "":
                    self._send_json(turn_payload(turn_id))
                    return
                if suffix == "events":
                    after = int(first(params, "after", "-1"))
                    self._send_json(turn_events_payload(turn_id, after=after))
                    return
            if parsed.path == "/api/skills":
                self._send_json(skill_catalog_payload())
                return
            if parsed.path == "/api/files":
                params = parse_qs(parsed.query)
                root = first(params, "path", "meet_files")
                limit = int(first(params, "limit", "200"))
                self._send_json(list_files_payload(root, limit=limit))
                return
            if parsed.path == "/api/meeting-archives":
                self._send_json(meeting_archives_payload())
                return
            if parsed.path == "/api/conversations":
                self._send_json(load_conversations_payload())
                return
            if parsed.path == "/api/projects":
                self._send_json(list_projects_payload())
                return
            if parsed.path == "/api/temporary-sync":
                self._send_json(temporary_sync_payload())
                return
            temporary_sync_file_id = parse_temporary_sync_file_route(parsed.path)
            if temporary_sync_file_id:
                self._send_temporary_sync_file(temporary_sync_file_id)
                return
            project_route = parse_project_route(parsed.path)
            if project_route:
                project_id, suffix = project_route
                if suffix == "":
                    self._send_json(project_detail_payload(project_id))
                    return
            if parsed.path == "/api/file":
                params = parse_qs(parsed.query)
                path = first(params, "path", "")
                max_chars = int(first(params, "max_chars", "50000"))
                self._send_json(read_file_payload(path, max_chars=max_chars))
                return
            if parsed.path == "/api/file/raw":
                params = parse_qs(parsed.query)
                path = first(params, "path", "")
                self._send_workspace_file(path)
                return
            if self._try_static(parsed.path):
                return
            self._send_json({"error": "Not found", "path": parsed.path}, status=404)
        except CLIENT_DISCONNECT_ERRORS:
            return
        except Exception as error:
            self._send_error(error)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
            if parsed.path == "/api/auth/login":
                user = get_auth_store().authenticate(
                    required_string(payload, "username"),
                    required_string(payload, "password"),
                )
                if user is None:
                    self._send_json({"error": "用户名或密码不正确"}, status=401)
                    return
                token = get_auth_store().create_session(user.id)
                self._send_auth_json(user, token)
                return
            if parsed.path == "/api/auth/register":
                user = get_auth_store().register(
                    required_string(payload, "username"),
                    required_string(payload, "password"),
                )
                token = get_auth_store().create_session(user.id)
                self._send_auth_json(user, token, status=201)
                return
            if parsed.path == "/api/auth/logout":
                get_auth_store().revoke_session(self._session_token())
                self._send_json(
                    {"ok": True},
                    headers={"Set-Cookie": self._expired_auth_cookie()},
                )
                return
            if not self._require_auth():
                return
            if parsed.path == "/api/auth/password":
                user = current_auth_user()
                get_auth_store().change_password(
                    user.id,
                    required_string(payload, "current_password"),
                    required_string(payload, "new_password"),
                )
                self._send_json({"ok": True, "message": "密码已更新"})
                return
            if parsed.path == "/api/models/use":
                if not self._require_admin():
                    return
                name = required_string(payload, "name")
                set_default_profile(WORKSPACE_ROOT / CONFIG_PATH, name)
                self._send_json(models_payload(message=f"Default model profile set to {name}"))
                return
            if parsed.path == "/api/models/key":
                if not self._require_admin():
                    return
                name = required_string(payload, "name")
                if name not in load_registry().names():
                    raise ValueError(f"模型配置不存在：{name}")
                api_key_env = api_key_env_for_profile(name)
                save_env_value(
                    WORKSPACE_ROOT / ".env",
                    api_key_env,
                    required_string(payload, "api_key"),
                )
                update_model_profile_api_key_env(
                    WORKSPACE_ROOT / CONFIG_PATH,
                    name,
                    api_key_env,
                )
                self._send_json(models_payload(message=f"{name} 的 API 密钥已保存"))
                return
            if parsed.path == "/api/models/add":
                if not self._require_admin():
                    return
                name = required_string(payload, "name")
                if name in load_registry().names():
                    raise ValueError(f"模型配置已存在：{name}")
                source_name = str(payload.get("source_name") or "").strip()
                api_key = str(payload.get("api_key") or "").strip()
                if api_key:
                    api_key_env = api_key_env_for_profile(name)
                    save_env_value(WORKSPACE_ROOT / ".env", api_key_env, api_key)
                elif source_name:
                    api_key_env = load_registry().get(source_name).api_key_env
                else:
                    raise ValueError("API 密钥不能为空。")
                profile_data = validated_model_profile_data(
                    payload,
                    name=name,
                    api_key_env=api_key_env,
                )
                add_model_profile(
                    WORKSPACE_ROOT / CONFIG_PATH,
                    profile_data,
                    set_default=bool(payload.get("set_default")),
                )
                self._send_json(models_payload(message="模型配置已添加，API 密钥已安全保存"))
                return
            if parsed.path == "/api/models/update":
                if not self._require_admin():
                    return
                name = required_string(payload, "name")
                registry = load_registry()
                existing = registry.get(name)
                api_key_env = existing.api_key_env
                api_key = str(payload.get("api_key") or "").strip()
                if api_key:
                    api_key_env = api_key_env_for_profile(name)
                    save_env_value(WORKSPACE_ROOT / ".env", api_key_env, api_key)
                profile_data = validated_model_profile_data(
                    payload,
                    name=name,
                    api_key_env=api_key_env,
                )
                update_model_profile(WORKSPACE_ROOT / CONFIG_PATH, name, profile_data)
                if bool(payload.get("set_default")):
                    set_default_profile(WORKSPACE_ROOT / CONFIG_PATH, name)
                self._send_json(models_payload(message=f"{name} 的配置已更新"))
                return
            if parsed.path == "/api/models/delete":
                if not self._require_admin():
                    return
                name = required_string(payload, "name")
                removed = delete_model_profile(WORKSPACE_ROOT / CONFIG_PATH, name)
                removed_key_env = str(removed.get("api_key_env") or "")
                if (
                    removed_key_env.startswith("WORK_AGENT_MODEL_")
                    and not model_api_key_env_in_use(removed_key_env)
                ):
                    delete_env_value(WORKSPACE_ROOT / ".env", removed_key_env)
                self._send_json(models_payload(message=f"{name} 已删除"))
                return
            if parsed.path == "/api/models/test":
                if not self._require_admin():
                    return
                self._send_json(test_model_connection_payload(payload))
                return
            if parsed.path == "/api/models/discover":
                if not self._require_admin():
                    return
                self._send_json(discover_models_payload(payload))
                return
            if parsed.path == "/api/settings/asr":
                if not self._require_admin():
                    return
                self._send_json(save_asr_settings_payload(payload))
                return
            if parsed.path == "/api/settings/agent":
                self._send_json(save_agent_settings_payload(payload))
                return
            if parsed.path == "/api/memories/update":
                self._send_json(update_cross_chat_memory_payload(payload))
                return
            if parsed.path == "/api/memories/delete":
                self._send_json(delete_cross_chat_memory_payload(payload))
                return
            if parsed.path == "/api/agent/run":
                self._send_json(run_agent_payload(payload))
                return
            if parsed.path == "/api/agent/chat":
                self._send_json(run_agent_chat_payload(payload))
                return
            if parsed.path == "/api/agent/chat-stream":
                self._send_sse(run_agent_chat_events(payload))
                return
            turn_route = parse_turn_route(parsed.path)
            if turn_route:
                turn_id, suffix = turn_route
                if suffix == "cancel":
                    self._send_json(cancel_turn_payload(turn_id))
                    return
                if suffix == "approve":
                    self._send_sse(approve_turn_events(turn_id, payload))
                    return
            if parsed.path == "/api/agent/title":
                self._send_json(generate_chat_title_payload(payload))
                return
            if parsed.path == "/api/skills/meeting-minutes":
                if not self._require_admin():
                    return
                self._send_json(run_meeting_minutes_payload(payload))
                return
            if parsed.path == "/api/skills/settings":
                self._send_json(save_skill_enabled_payload(payload))
                return
            if parsed.path == "/api/speech/transcribe":
                self._send_json(transcribe_speech_payload(payload))
                return
            if parsed.path == "/api/speech/vad":
                self._send_json(speech_vad_payload(payload))
                return
            if parsed.path == "/api/realtime-transcript/save":
                self._send_json(save_realtime_transcript_payload(payload))
                return
            if parsed.path == "/api/attachments/add":
                self._send_json(add_attachment_payload(payload))
                return
            if parsed.path == "/api/temporary-sync/text":
                self._send_json(save_temporary_sync_text_payload(payload))
                return
            if parsed.path == "/api/temporary-sync/files/add":
                self._send_json(add_temporary_sync_file_payload(payload), status=201)
                return
            if parsed.path == "/api/temporary-sync/files/delete":
                self._send_json(delete_temporary_sync_file_payload(payload))
                return
            if parsed.path == "/api/conversations/save":
                self._send_json(save_conversations_payload(payload))
                return
            if parsed.path == "/api/projects/create":
                self._send_json(create_project_payload(payload), status=201)
                return
            project_route = parse_project_route(parsed.path)
            if project_route:
                project_id, suffix = project_route
                if suffix == "settings":
                    self._send_json(update_project_payload(project_id, payload))
                    return
                if suffix == "files/add":
                    self._send_json(add_project_file_payload(project_id, payload), status=201)
                    return
                if suffix == "files/delete":
                    self._send_json(delete_project_file_payload(project_id, payload))
                    return
                if suffix == "sync-meeting":
                    self._send_json(sync_meeting_to_project_payload(project_id, payload))
                    return
            if parsed.path == "/api/file/open":
                if not self._require_admin():
                    return
                self._send_json(open_workspace_file_payload(payload, reveal=False))
                return
            if parsed.path == "/api/file/reveal":
                if not self._require_admin():
                    return
                self._send_json(open_workspace_file_payload(payload, reveal=True))
                return
            self._send_json({"error": "Not found", "path": parsed.path}, status=404)
        except CLIENT_DISCONNECT_ERRORS:
            return
        except Exception as error:
            self._send_error(error)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[web] {self.address_string()} - {format % args}")

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("Request JSON body must be an object.")
        return data

    def _session_token(self) -> str:
        raw_cookie = self.headers.get("Cookie", "")
        if not raw_cookie:
            return ""
        cookie = SimpleCookie()
        try:
            cookie.load(raw_cookie)
        except Exception:
            return ""
        morsel = cookie.get(AUTH_COOKIE_NAME)
        return morsel.value if morsel else ""

    def _authenticated_user(self) -> AuthUser | None:
        user = get_auth_store().user_for_session(self._session_token())
        if user is not None:
            REQUEST_AUTH.user = user
        return user

    def _require_auth(self) -> bool:
        user = self._authenticated_user()
        if user is not None:
            return True
        self._send_json({"error": "请先登录", "type": "AuthenticationRequired"}, status=401)
        return False

    def _require_admin(self) -> bool:
        if current_auth_user().role == "admin":
            return True
        self._send_json({"error": "此操作仅限管理员", "type": "AdminRequired"}, status=403)
        return False

    def _auth_cookie(self, token: str) -> str:
        return (
            f"{AUTH_COOKIE_NAME}={token}; Path=/; HttpOnly; SameSite=Strict; "
            f"Max-Age={SESSION_TTL_SECONDS}"
        )

    def _expired_auth_cookie(self) -> str:
        return f"{AUTH_COOKIE_NAME}=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0"

    def _send_auth_json(self, user: AuthUser, token: str, *, status: int = 200) -> None:
        self._send_json(
            {"authenticated": True, "user": user.to_payload()},
            status=status,
            headers={"Set-Cookie": self._auth_cookie(token)},
        )

    def _send_json(
        self,
        payload: dict[str, Any],
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self._send_common_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _send_empty(self, status: int) -> None:
        self.send_response(status)
        self._send_common_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _send_common_headers(self) -> None:
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store")

    def _send_sse(self, events: Iterable[dict[str, Any]]) -> None:
        self.send_response(200)
        self._send_common_headers()
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        client_disconnected = False
        try:
            for event in events:
                self._write_sse_event(event)
        except CLIENT_DISCONNECT_ERRORS:
            client_disconnected = True
        except Exception as error:
            trace_lines = traceback.format_exc().splitlines()
            print("[web] SSE handler failed:")
            print("\n".join(trace_lines))
            try:
                self._write_sse_event(
                    {
                        "event": "error",
                        "message": friendly_error_message(error),
                        "type": type(error).__name__,
                        "detail": str(error),
                        "trace": trace_lines[-12:],
                    }
                )
            except CLIENT_DISCONNECT_ERRORS:
                client_disconnected = True
        finally:
            if not client_disconnected:
                try:
                    self._write_sse_event({"event": "done"})
                except CLIENT_DISCONNECT_ERRORS:
                    pass

    def _send_workspace_file(self, path: str) -> None:
        workspace = WorkspaceFiles(account_workspace_root())
        file_path = workspace.resolve(path)
        if not file_path.is_file():
            raise ValueError(f"Not a file: {path}")
        stat = file_path.stat()
        content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        self.send_response(200)
        self._send_common_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(stat.st_size))
        self.send_header("Content-Disposition", f"inline; filename*=UTF-8''{quote(file_path.name)}")
        self.end_headers()
        with file_path.open("rb") as handle:
            shutil.copyfileobj(handle, self.wfile)

    def _send_temporary_sync_file(self, file_id: str) -> None:
        file_path, metadata = resolve_temporary_sync_file(file_id)
        self.send_response(200)
        self._send_common_headers()
        self.send_header(
            "Content-Type",
            str(metadata.get("mime_type") or "application/octet-stream"),
        )
        self.send_header("Content-Length", str(file_path.stat().st_size))
        self.send_header(
            "Content-Disposition",
            f"attachment; filename*=UTF-8''{quote(str(metadata.get('name') or 'download'))}",
        )
        self.end_headers()
        with file_path.open("rb") as handle:
            shutil.copyfileobj(handle, self.wfile)

    def _write_sse_event(self, payload: dict[str, Any]) -> None:
        event_name = str(payload.get("event") or "message")
        data = json.dumps(payload, ensure_ascii=False)
        self.wfile.write(f"event: {event_name}\n".encode("utf-8"))
        for line in data.splitlines() or ["{}"]:
            self.wfile.write(f"data: {line}\n".encode("utf-8"))
        self.wfile.write(b"\n")
        self.wfile.flush()

    def _send_error(self, error: Exception) -> None:
        trace_lines = traceback.format_exc().splitlines()
        print("[web] request failed:")
        print("\n".join(trace_lines))
        try:
            self._send_json(
                {
                    "error": friendly_error_message(error),
                    "type": type(error).__name__,
                    "detail": str(error),
                    "trace": trace_lines[-12:],
                },
                status=400 if isinstance(error, ValueError) else 500,
            )
        except CLIENT_DISCONNECT_ERRORS:
            return

    def _try_static(self, request_path: str) -> bool:
        root = (WORKSPACE_ROOT / STATIC_DIR).resolve()
        if not root.exists():
            return False
        normalized = "index.html" if request_path in ("", "/") else request_path.lstrip("/")
        candidate = (root / normalized).resolve()
        if root not in (candidate, *candidate.parents) or not candidate.exists():
            candidate = root / "index.html"
        if not candidate.is_file():
            return False
        body = candidate.read_bytes()
        content_type = mimetypes.guess_type(str(candidate))[0] or "application/octet-stream"
        self.send_response(200)
        self._send_common_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        return True


def first(params: dict[str, list[str]], key: str, default: str) -> str:
    values = params.get(key)
    if not values:
        return default
    return values[0]


def parse_turn_route(path: str) -> tuple[str, str] | None:
    match = re.fullmatch(r"/api/agent/turns/([^/]+)(?:/([^/]+))?", path)
    if not match:
        return None
    turn_id = sanitize_turn_id(match.group(1))
    suffix = str(match.group(2) or "")
    if not turn_id:
        return None
    if suffix not in {"", "events", "cancel", "approve"}:
        return None
    return turn_id, suffix


def friendly_error_message(error: Exception) -> str:
    text = str(error) or type(error).__name__
    lower = text.lower()
    if "llm" in lower and "timed out" in lower:
        return (
            f"{text}\n\n建议：先检查当前模型的网络连接后重试。"
            "如需更换模型，请在设置中手动选择；系统不会自动切换模型。"
        )
    if "read operation timed out" in lower or "timed out" in lower:
        return (
            f"{text}\n\n这次更像是上游模型/网络读超时，不是文件库不存在。"
            "可以重试一次；如需更换模型，请在设置中手动选择。"
        )
    return text


def required_string(payload: dict[str, Any], key: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise ValueError(f"Missing required field: {key}")
    return value


def load_registry() -> ModelRegistry:
    return ModelRegistry.load(WORKSPACE_ROOT / CONFIG_PATH)


def profile_payload(profile: ModelProfile, default_profile: str) -> dict[str, Any]:
    data = asdict(profile)
    data["default"] = profile.name == default_profile
    data["api_key_configured"] = bool(os.getenv(profile.api_key_env))
    return data


def models_payload(message: str | None = None) -> dict[str, Any]:
    registry = load_registry()
    payload = {
        "default_profile": registry.default_profile,
        "env_override": os.getenv("WORK_AGENT_MODEL_PROFILE"),
        "profiles": [
            profile_payload(registry.get(name), registry.default_profile)
            for name in registry.names()
        ],
    }
    if message:
        payload["message"] = message
    return payload


def validated_model_profile_data(
    payload: dict[str, Any],
    *,
    name: str,
    api_key_env: str,
) -> dict[str, Any]:
    base_url = required_string(payload, "base_url").rstrip("/")
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("接口地址必须是完整的 http:// 或 https:// 地址。")
    temperature = float(
        payload["temperature"] if payload.get("temperature") is not None else 0.6
    )
    max_tokens = int(
        payload["max_tokens"] if payload.get("max_tokens") is not None else 16384
    )
    timeout_seconds = int(
        payload["timeout_seconds"] if payload.get("timeout_seconds") is not None else 180
    )
    if not 0 <= temperature <= 2:
        raise ValueError("温度必须在 0 到 2 之间。")
    if not 1 <= max_tokens <= 262144:
        raise ValueError("最大输出必须在 1 到 262144 之间。")
    if not 10 <= timeout_seconds <= 3600:
        raise ValueError("超时时间必须在 10 到 3600 秒之间。")
    return {
        "name": name,
        "provider": str(payload.get("provider") or "openai-compatible").strip(),
        "base_url": base_url,
        "model": required_string(payload, "model"),
        "api_key_env": api_key_env,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout_seconds": timeout_seconds,
    }


def model_api_key_env_in_use(api_key_env: str) -> bool:
    path = WORKSPACE_ROOT / CONFIG_PATH
    data = json.loads(path.read_text(encoding="utf-8"))
    return any(
        str(profile.get("api_key_env") or "") == api_key_env
        for profile in data.get("profiles", [])
    )


def model_access_fields(payload: dict[str, Any]) -> tuple[str, str, str, int]:
    name = str(payload.get("name") or "").strip()
    source_name = str(payload.get("source_name") or "").strip()
    existing: ModelProfile | None = None
    if name or source_name:
        try:
            existing = load_registry().get(name or source_name)
        except KeyError:
            if source_name:
                existing = load_registry().get(source_name)
    base_url = str(payload.get("base_url") or (existing.base_url if existing else "")).strip()
    model = str(payload.get("model") or (existing.model if existing else "")).strip()
    api_key = str(payload.get("api_key") or "").strip()
    if not api_key and existing is not None:
        api_key = os.getenv(existing.api_key_env, "").strip()
    timeout_seconds = int(
        payload.get("timeout_seconds")
        or (existing.timeout_seconds if existing else 30)
    )
    if not base_url:
        raise ValueError("请先填写接口地址。")
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("接口地址必须是完整的 http:// 或 https:// 地址。")
    if not model:
        raise ValueError("请先填写模型名称。")
    if not api_key:
        raise ValueError("请先填写 API 密钥。")
    return base_url.rstrip("/"), model, api_key, min(max(timeout_seconds, 5), 60)


def model_route_profile(base_url: str, model: str, timeout_seconds: int) -> ModelProfile:
    host = (urlparse(base_url).hostname or "").lower()
    return ModelProfile(
        name=model,
        provider="deepseek" if host in {"api.deepseek.com", "api.deepseek.cn"} else "openai-compatible",
        base_url=base_url,
        model=model,
        api_key_env="",
        timeout_seconds=timeout_seconds,
    )


def test_model_connection_payload(payload: dict[str, Any]) -> dict[str, Any]:
    base_url, model, api_key, timeout_seconds = model_access_fields(payload)
    endpoint = chat_completions_endpoint(base_url)
    request_body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "max_tokens": 2,
            "stream": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=request_body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started_at = time.monotonic()
    client = OpenAICompatibleClient()
    route_profile = model_route_profile(base_url, model, timeout_seconds)
    try:
        with client._open_request(
            request,
            profile=route_profile,
            timeout=timeout_seconds,
        ) as response:
            raw = response.read().decode("utf-8", errors="replace")
            status = int(getattr(response, "status", 200))
    except urllib.error.HTTPError as error:
        detail = error.read(2000).decode("utf-8", errors="replace")
        raise ValueError(f"连接测试失败：HTTP {error.code} · {compact_error_detail(detail)}") from error
    except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as error:
        raise ValueError(f"连接测试失败：{error}") from error
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("连接成功，但接口返回的不是有效 JSON。") from error
    if not isinstance(parsed, dict) or not isinstance(parsed.get("choices"), list):
        raise ValueError("连接成功，但响应不是 OpenAI Chat Completions 格式。")
    return {
        "ok": True,
        "status": status,
        "latency_ms": max(1, round((time.monotonic() - started_at) * 1000)),
        "endpoint": endpoint,
        "model": model,
        "message": "连接可用",
    }


def discover_models_payload(payload: dict[str, Any]) -> dict[str, Any]:
    base_url, model, api_key, timeout_seconds = model_access_fields(payload)
    endpoint = models_endpoint_for_base_url(base_url)
    request = urllib.request.Request(
        endpoint,
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        method="GET",
    )
    started_at = time.monotonic()
    client = OpenAICompatibleClient()
    route_profile = model_route_profile(base_url, model, timeout_seconds)
    try:
        with client._open_request(
            request,
            profile=route_profile,
            timeout=timeout_seconds,
        ) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        detail = error.read(2000).decode("utf-8", errors="replace")
        raise ValueError(f"获取模型失败：HTTP {error.code} · {compact_error_detail(detail)}") from error
    except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as error:
        raise ValueError(f"获取模型失败：{error}") from error
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("接口返回的模型列表不是有效 JSON。") from error
    raw_models = parsed.get("data") if isinstance(parsed, dict) else None
    if not isinstance(raw_models, list):
        raise ValueError("接口未返回 OpenAI-compatible 的 data 模型列表。")
    models = sorted(
        {
            str(item.get("id") or "").strip()
            for item in raw_models
            if isinstance(item, dict) and str(item.get("id") or "").strip()
        }
    )
    return {
        "ok": True,
        "models": models[:500],
        "count": len(models),
        "endpoint": endpoint,
        "latency_ms": max(1, round((time.monotonic() - started_at) * 1000)),
        "message": f"已获取 {len(models)} 个模型",
    }


def models_endpoint_for_base_url(base_url: str) -> str:
    endpoint = str(base_url).rstrip("/")
    for suffix in ("/chat/completions", "/responses"):
        if endpoint.endswith(suffix):
            endpoint = endpoint[: -len(suffix)]
            break
    return f"{endpoint.rstrip('/')}/models"


def compact_error_detail(detail: str) -> str:
    text = re.sub(r"\s+", " ", str(detail or "")).strip()
    return text[:500] or "上游未返回错误详情"


def load_asr_settings() -> dict[str, Any]:
    path = WORKSPACE_ROOT / ASR_SETTINGS_PATH
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
    settings["model_id"] = str(settings.get("model_id") or default_asr_model_id(settings["profile"]))
    settings["backend"] = "mlx"
    settings["hotwords"] = normalize_multiline_text(str(settings.get("hotwords") or ""))
    return settings


def load_agent_settings() -> dict[str, Any]:
    path = user_agent_settings_path()
    data: dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except json.JSONDecodeError:
            data = {}
    settings = {**DEFAULT_AGENT_SETTINGS, **data}
    settings["work_background"] = normalize_multiline_text(str(settings.get("work_background") or ""))
    settings["company_document_format"] = normalize_multiline_text(
        str(settings.get("company_document_format") or "")
    )
    return settings


def enabled_skill_ids() -> set[str]:
    overrides = load_agent_settings().get("skill_enabled")
    if not isinstance(overrides, dict):
        overrides = {}
    known_ids = {"meeting-minutes"}
    known_ids.update(manifest.id for manifest in load_skill_manifests(WORKSPACE_ROOT))
    return {
        skill_id
        for skill_id in known_ids
        if bool(overrides.get(skill_id, skill_id not in DEFAULT_DISABLED_SKILL_IDS))
    }


def save_skill_enabled_payload(payload: dict[str, Any]) -> dict[str, Any]:
    skill_id = required_string(payload, "skill_id")
    enabled = payload.get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError("enabled 必须是布尔值。")
    known_ids = {"meeting-minutes"}
    known_ids.update(manifest.id for manifest in load_skill_manifests(WORKSPACE_ROOT))
    if skill_id not in known_ids:
        raise ValueError(f"未找到技能：{skill_id}")
    current = load_agent_settings()
    overrides = current.get("skill_enabled")
    if not isinstance(overrides, dict):
        overrides = {}
    overrides = {str(key): bool(value) for key, value in overrides.items() if str(key) in known_ids}
    overrides[skill_id] = enabled
    current["skill_enabled"] = overrides
    path = user_agent_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return skill_catalog_payload(message=f"{skill_id} 已{'启用' if enabled else '关闭'}")


def agent_settings_payload(message: str | None = None) -> dict[str, Any]:
    payload = load_agent_settings()
    if message:
        payload["message"] = message
    return payload


def save_agent_settings_payload(payload: dict[str, Any]) -> dict[str, Any]:
    current = load_agent_settings()
    settings = {
        "work_background": normalize_multiline_text(
            str(payload.get("work_background", current.get("work_background")) or "")
        ),
        "company_document_format": normalize_multiline_text(
            str(payload.get("company_document_format", current.get("company_document_format")) or "")
        ),
        "skill_enabled": (
            {str(key): bool(value) for key, value in current.get("skill_enabled", {}).items()}
            if isinstance(current.get("skill_enabled"), dict)
            else {}
        ),
    }
    path = user_agent_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return agent_settings_payload(message="Agent settings saved")


def cross_chat_memories_payload(*, project_id: str | None = None, query: str = "") -> dict[str, Any]:
    store = CrossChatMemoryStore(get_session_store())
    memories = store.list(project_id=project_id, query=query)
    return {
        "memories": memories,
        "count": len(memories),
        "automatic": True,
        "source": "conversation_summaries",
        "project_id": project_id,
    }


def update_cross_chat_memory_payload(payload: dict[str, Any]) -> dict[str, Any]:
    memory = CrossChatMemoryStore(get_session_store()).update(
        required_string(payload, "id"),
        required_string(payload, "content"),
    )
    return {"ok": True, "memory": memory, "message": "记忆已纠正"}


def delete_cross_chat_memory_payload(payload: dict[str, Any]) -> dict[str, Any]:
    CrossChatMemoryStore(get_session_store()).delete(required_string(payload, "id"))
    return {"ok": True, "message": "记忆已删除"}


def asr_settings_payload(message: str | None = None) -> dict[str, Any]:
    payload = load_asr_settings()
    payload["available_profiles"] = [
        {
            "name": "qwen3-asr-mlx-8bit",
            "label": "Qwen3-ASR MLX 8bit（Mac推荐）",
            "default_model_id": default_asr_model_id("qwen3-asr-mlx-8bit"),
        }
    ]
    if message:
        payload["message"] = message
    return payload


def save_asr_settings_payload(payload: dict[str, Any]) -> dict[str, Any]:
    profile = normalize_asr_profile(str(payload.get("profile") or DEFAULT_ASR_SETTINGS["profile"]))
    settings = {
        "profile": profile,
        "model_id": str(payload.get("model_id") or default_asr_model_id(profile)).strip(),
        "backend": "mlx",
        "hotwords": normalize_multiline_text(str(payload.get("hotwords") or "")),
    }
    path = WORKSPACE_ROOT / ASR_SETTINGS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    stop_asr_worker()
    return asr_settings_payload(message="ASR settings saved")


def normalize_asr_profile(profile: str) -> str:
    profile = profile.strip().lower()
    if profile in {"qwen3-asr-mlx-8bit", "qwen3-mlx", "mlx", "mlx-8bit", "qwen3-asr", "qwen3"}:
        return "qwen3-asr-mlx-8bit"
    return "qwen3-asr-mlx-8bit"


def default_asr_model_id(profile: str) -> str:
    if (WORKSPACE_ROOT / QWEN3_MLX_LOCAL_MODEL_ID).exists():
        return QWEN3_MLX_LOCAL_MODEL_ID
    return QWEN3_MLX_REMOTE_MODEL_ID


def normalize_multiline_text(text: str) -> str:
    lines = [line.strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return "\n".join(line for line in lines if line).strip()


def build_runtime_hotwords(settings: dict[str, Any], *extra_texts: str) -> str:
    terms: list[str] = []
    terms.extend(str(settings.get("hotwords") or "").replace("\n", " ").split())
    terms.extend(extract_hotword_terms(str(load_agent_settings().get("work_background") or "")))
    for text in extra_texts:
        terms.extend(extract_hotword_terms(text))
    seen: set[str] = set()
    cleaned: list[str] = []
    for term in terms:
        normalized = term.strip(" ，。、“”‘’（）()[]【】;；:：\n\t")
        if len(normalized) < 2 or normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(normalized)
    return " ".join(cleaned[:180])


def extract_hotword_terms(text: str) -> list[str]:
    if not text.strip():
        return []
    terms: list[str] = []
    terms.extend(re.findall(r"[A-Za-z][A-Za-z0-9_+.-]{1,31}", text))
    for match in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,24}", text):
        if match in {"这个", "那个", "然后", "就是", "我们", "他们", "你们", "进行", "相关"}:
            continue
        terms.append(match)
    return terms


def tools_payload() -> dict[str, Any]:
    registry = load_registry()
    profile = registry.get()
    client = OpenAICompatibleClient()
    tools = build_default_tools(
        WORKSPACE_ROOT,
        client,
        profile,
        data_workspace=account_workspace_root(),
        include_shared_tools=current_auth_user().role == "admin",
        enabled_skill_ids=enabled_skill_ids(),
    )
    return {
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "provider_id": getattr(tool, "provider_id", "local"),
                "provider_kind": getattr(tool, "provider_kind", "local"),
            }
            for tool in tools.list_model_tools()
        ],
        "providers": tools.providers() if hasattr(tools, "providers") else [],
        "hidden_skill_tool_count": max(0, len(tools.list()) - len(tools.list_model_tools())),
        "default_max_steps": DEFAULT_MAX_STEPS,
    }


def agent_system_context() -> str:
    settings = load_agent_settings()
    work_background = str(settings.get("work_background") or "").strip()
    company_document_format = str(settings.get("company_document_format") or "").strip()
    if not work_background and not company_document_format:
        return ""
    blocks: list[str] = []
    if work_background:
        blocks.append(
            "用户长期工作背景/常用系统提示词：\n"
            f"{work_background}\n\n"
            "使用规则：该背景用于默认工作语境、称谓口径、专名纠错和文档写作风格；"
            "它不是某一场会议已经发生或对方已经确认的事实。"
            "生成正式材料时，仍必须以用户确认信息、会议转写和附件材料为依据。"
        )
    if company_document_format:
        blocks.append(
            "公司标准文件格式（纯文字设置）：\n"
            f"{company_document_format}\n\n"
            "使用规则：这是公司级 Word 排版覆盖，不是正文内容。"
            "普通正式文字材料可直接交给 docx 技能按此生成；"
            "请示、报告、通知、通报、函、批复、意见、决定、公告、通告、纪要等明显公文内容，"
            "先打开 official-document 技能确定文种、要素和规范，再打开 docx 技能生成最终文件。"
            "不得为套格式虚构红头、文号、签发人、印章、密级或日期。"
        )
    return "\n\n".join(blocks)



def skill_catalog_payload(message: str | None = None) -> dict[str, Any]:
    skills = [
        {
            "id": "meeting-minutes",
            "label": "会议纪要",
            "mention": "@会议纪要",
            "description": "接收会议录音、转写文本和补充材料，按技能流程生成内部留档版Markdown与工作提交版Word纪要。",
            "when_to_use": "处理会议录音、ASR转写、旁听会议记录、工作纪要提交稿。",
            "tool_name": "transcribe_meeting_audio",
            "outputs": [
                "会议沟通内容整理_ASR转写稿_Qwen3.md",
                "会议沟通内容整理_内部留档版.md",
                "会议纪要.docx",
            ],
        },
    ]
    existing_ids = {str(item["id"]) for item in skills}
    for manifest in load_skill_manifests(WORKSPACE_ROOT):
        if manifest.id in existing_ids:
            continue
        skills.append(manifest.to_payload())
        existing_ids.add(manifest.id)
    active_ids = enabled_skill_ids()
    for skill in skills:
        skill["enabled"] = str(skill["id"]) in active_ids
    payload: dict[str, Any] = {"skills": skills}
    if message:
        payload["message"] = message
    return payload


TEXT_PREVIEW_EXTENSIONS = {
    ".md",
    ".txt",
    ".json",
    ".yml",
    ".yaml",
    ".csv",
    ".log",
    ".srt",
    ".vtt",
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".css",
    ".html",
    ".xml",
    ".toml",
    ".ini",
    ".sh",
    ".sql",
}
MARKDOWN_PREVIEW_EXTENSIONS = {".md"}
PDF_PREVIEW_EXTENSIONS = {".pdf"}
IMAGE_PREVIEW_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".heic", ".tif", ".tiff"}
AUDIO_PREVIEW_EXTENSIONS = {".m4a", ".mp3", ".wav", ".aac", ".flac", ".ogg", ".opus", ".aiff", ".aif", ".caf"}
VIDEO_PREVIEW_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"}


def list_files_payload(root: str, *, limit: int) -> dict[str, Any]:
    storage_root = account_workspace_root()
    workspace = WorkspaceFiles(storage_root)
    directory = workspace.resolve(root)
    if not directory.exists():
        return {"root": root, "files": []}
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {root}")
    files: list[dict[str, Any]] = []
    for item in directory.rglob("*"):
        if not item.is_file():
            continue
        if not is_file_library_visible(item):
            continue
        files.append(file_item_payload(item))
    files.sort(key=lambda item: (int(item["modified"]), str(item["path"])), reverse=True)
    attachment_index = load_attachment_index(storage_root / "meet_files" / "attachments")
    files = dedupe_file_library_items(files, attachment_index)[:limit]
    return {"root": str(directory.relative_to(storage_root)), "files": files}


def meeting_archives_payload() -> dict[str, Any]:
    storage_root = account_workspace_root()
    archive_root = storage_root / "meet_files" / "会议项目"
    if not archive_root.exists():
        return {"root": "meet_files/会议项目", "meetings": []}
    if not archive_root.is_dir():
        raise ValueError("会议项目归档路径不是目录。")

    meetings: list[dict[str, Any]] = []
    for manifest_path in archive_root.glob("*/manifest.json"):
        try:
            meeting = meeting_archive_from_manifest(manifest_path)
        except Exception as error:
            print(f"[web] failed to read meeting archive manifest {manifest_path}: {error}")
            continue
        meetings.append(meeting)
    meetings.sort(key=lambda item: int(item.get("updated_at") or 0), reverse=True)
    return {"root": str(archive_root.relative_to(storage_root)), "meetings": meetings}


def meeting_archive_from_manifest(manifest_path: Path) -> dict[str, Any]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("manifest.json must be an object.")
    canonical = data.get("canonical_outputs") if isinstance(data.get("canonical_outputs"), dict) else {}
    outputs = {
        "asr": meeting_archive_output_payload(str(canonical.get("asr") or "")),
        "internal": meeting_archive_output_payload(str(canonical.get("internal") or "")),
        "work_md": meeting_archive_output_payload(str(canonical.get("work_md") or "")),
        "work_docx": meeting_archive_output_payload(str(canonical.get("work_docx") or "")),
    }
    return {
        "schema_version": int(data.get("schema_version") or 1),
        "meeting_id": str(data.get("meeting_id") or manifest_path.parent.name),
        "title": str(data.get("title") or manifest_path.parent.name),
        "archive_dir": str(data.get("archive_dir") or manifest_path.parent.relative_to(WORKSPACE_ROOT)),
        "manifest_path": str(manifest_path.relative_to(WORKSPACE_ROOT)),
        "meeting_time": meeting_time_from_manifest(data),
        "created_at": int(data.get("created_at") or manifest_path.stat().st_ctime),
        "updated_at": int(data.get("updated_at") or manifest_path.stat().st_mtime),
        "source_path": str(data.get("source_path") or ""),
        "transcript_path": str(data.get("transcript_path") or ""),
        "outputs": outputs,
    }


def meeting_time_from_manifest(data: dict[str, Any]) -> dict[str, Any] | None:
    explicit = data.get("meeting_time")
    if isinstance(explicit, dict) and str(explicit.get("display") or "").strip():
        return explicit

    recording = data.get("recording_metadata")
    if not isinstance(recording, dict):
        return None
    start_raw = str(recording.get("recording_started_at") or "").strip()
    if not start_raw:
        return None
    try:
        started_at = datetime.fromisoformat(start_raw)
    except ValueError:
        return None

    display = f"{started_at.year}年{started_at.month}月{started_at.day}日"
    fallback: dict[str, Any] = {
        "display": display,
        "start": start_raw,
        "source": "recording_metadata_fallback",
    }
    end_raw = str(recording.get("recording_ended_at") or "").strip()
    if end_raw:
        fallback["end"] = end_raw
    return fallback


def meeting_archive_output_payload(raw_path: str) -> dict[str, Any] | None:
    if not raw_path:
        return None
    workspace = WorkspaceFiles(WORKSPACE_ROOT)
    try:
        path = workspace.resolve(raw_path)
    except Exception:
        return {
            "path": raw_path,
            "name": Path(raw_path).name,
            "exists": False,
            "size": 0,
            "modified": 0,
            "extension": Path(raw_path).suffix.lower(),
            "mime_type": "application/octet-stream",
            "kind": "file",
            "previewable": False,
        }
    if not path.is_file():
        return {
            "path": raw_path,
            "name": path.name,
            "exists": False,
            "size": 0,
            "modified": 0,
            "extension": path.suffix.lower(),
            "mime_type": mimetypes.guess_type(str(path))[0] or "application/octet-stream",
            "kind": classify_attachment(path.suffix.lower(), mimetypes.guess_type(str(path))[0] or ""),
            "previewable": False,
        }
    item = file_item_payload(path)
    item["exists"] = True
    return item


def load_conversations_payload() -> dict[str, Any]:
    path = user_conversation_history_path()
    if not path.is_file():
        return {"items": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return {"items": sanitize_conversation_archive_items(data["items"])}
    if isinstance(data, list):
        return {"items": sanitize_conversation_archive_items(data)}
    return {"items": []}


def save_conversations_payload(payload: dict[str, Any]) -> dict[str, Any]:
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("items must be a list.")
    sanitized_items = sanitize_conversation_archive_items(items)
    path = user_conversation_history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"items": sanitized_items, "saved_at": int(time.time())}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"ok": True, "path": str(path.relative_to(WORKSPACE_ROOT)), "count": len(sanitized_items)}


def projects_root() -> Path:
    root = (account_workspace_root() / PROJECTS_ROOT).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def sanitize_project_id(raw_value: Any) -> str:
    project_id = str(raw_value or "").strip().lower()
    if not PROJECT_ID_PATTERN.fullmatch(project_id):
        raise ValueError("项目标识无效。")
    return project_id


def project_dir(project_id: str) -> Path:
    clean_id = sanitize_project_id(project_id)
    directory = (projects_root() / clean_id).resolve()
    try:
        directory.relative_to(projects_root())
    except ValueError as error:
        raise ValueError("项目路径无效。") from error
    return directory


def project_manifest_path(project_id: str) -> Path:
    return project_dir(project_id) / "project.json"


def parse_project_route(path: str) -> tuple[str, str] | None:
    match = re.fullmatch(r"/api/projects/(project-[a-f0-9]{12})(?:/(.*))?", path)
    if not match:
        return None
    return match.group(1), str(match.group(2) or "")


def read_project(project_id: str) -> dict[str, Any]:
    manifest = project_manifest_path(project_id)
    if not manifest.is_file():
        raise ValueError("项目不存在或当前账户无权访问。")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("项目配置损坏。")
    return data


def write_project(data: dict[str, Any]) -> None:
    project_id = sanitize_project_id(data.get("id"))
    manifest = project_manifest_path(project_id)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temp = manifest.with_suffix(".json.tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(manifest)


def project_files(project_id: str) -> list[dict[str, Any]]:
    sources = project_dir(project_id) / "sources"
    if not sources.is_dir():
        return []
    items = [file_item_payload(path) for path in sources.rglob("*") if path.is_file()]
    items.sort(key=lambda item: (int(item["modified"]), str(item["name"])), reverse=True)
    return items


def project_payload(data: dict[str, Any], *, include_files: bool = True) -> dict[str, Any]:
    project_id = sanitize_project_id(data.get("id"))
    files = project_files(project_id) if include_files else []
    return {
        "id": project_id,
        "name": str(data.get("name") or "未命名项目"),
        "instructions": str(data.get("instructions") or ""),
        "memory_scope": "project_only",
        "created_at": int(data.get("created_at") or 0),
        "updated_at": int(data.get("updated_at") or 0),
        "root": account_relative_path(project_dir(project_id)),
        "file_count": len(files) if include_files else int(data.get("file_count") or 0),
        "files": files if include_files else None,
    }


def list_projects_payload() -> dict[str, Any]:
    projects: list[dict[str, Any]] = []
    for manifest in projects_root().glob("project-*/project.json"):
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                continue
            payload = project_payload(data, include_files=True)
            payload.pop("files", None)
            projects.append(payload)
        except Exception as error:
            print(f"[web] failed to read project manifest {manifest}: {error}")
    projects.sort(key=lambda item: int(item.get("updated_at") or 0), reverse=True)
    return {"projects": projects}


def project_detail_payload(project_id: str) -> dict[str, Any]:
    return {"project": project_payload(read_project(project_id), include_files=True)}


def normalize_project_name(raw_value: Any) -> str:
    name = re.sub(r"\s+", " ", str(raw_value or "")).strip()
    if not name:
        raise ValueError("请输入项目名称。")
    if len(name) > 80:
        raise ValueError("项目名称不能超过 80 个字符。")
    return name


def normalize_project_instructions(raw_value: Any) -> str:
    instructions = normalize_multiline_text(str(raw_value or ""))
    if len(instructions) > 12000:
        raise ValueError("项目指令不能超过 12000 个字符。")
    return instructions


def create_project_payload(payload: dict[str, Any]) -> dict[str, Any]:
    now = int(time.time())
    project_id = f"project-{uuid.uuid4().hex[:12]}"
    data = {
        "schema_version": 1,
        "id": project_id,
        "name": normalize_project_name(payload.get("name")),
        "instructions": normalize_project_instructions(payload.get("instructions")),
        "memory_scope": "project_only",
        "created_at": now,
        "updated_at": now,
    }
    directory = project_dir(project_id)
    (directory / "sources").mkdir(parents=True, exist_ok=False)
    write_project(data)
    return {"project": project_payload(data, include_files=True)}


def update_project_payload(project_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = read_project(project_id)
    if "name" in payload:
        data["name"] = normalize_project_name(payload.get("name"))
    if "instructions" in payload:
        data["instructions"] = normalize_project_instructions(payload.get("instructions"))
    data["updated_at"] = int(time.time())
    write_project(data)
    return {"project": project_payload(data, include_files=True)}


def decode_uploaded_file(payload: dict[str, Any]) -> tuple[str, str, bytes]:
    name = sanitize_filename(required_string(payload, "name"))
    mime_type = str(payload.get("mime_type") or "application/octet-stream")
    try:
        data = base64.b64decode(required_string(payload, "content_base64"), validate=True)
    except Exception as error:
        raise ValueError("文件内容无效。") from error
    if not data:
        raise ValueError("文件不能为空。")
    return name, mime_type, data


def add_project_file_payload(project_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    project = read_project(project_id)
    name, mime_type, data = decode_uploaded_file(payload)
    sources = project_dir(project_id) / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    target = unique_path(sources / name)
    target.write_bytes(data)
    project["updated_at"] = int(time.time())
    write_project(project)
    attachment = attachment_payload_from_path(
        target,
        display_name=name,
        mime_type=mime_type,
        deduplicated=False,
    )
    return {"file": file_item_payload(target), "attachment": attachment, "project": project_payload(project)}


def delete_project_file_payload(project_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    project = read_project(project_id)
    raw_path = required_string(payload, "path")
    file_path = WorkspaceFiles(account_workspace_root()).resolve(raw_path)
    sources = (project_dir(project_id) / "sources").resolve()
    try:
        file_path.relative_to(sources)
    except ValueError as error:
        raise ValueError("只能删除当前项目资料目录中的文件。") from error
    if not file_path.is_file():
        raise ValueError("项目文件不存在。")
    file_path.unlink()
    project["updated_at"] = int(time.time())
    write_project(project)
    return {"ok": True, "project": project_payload(project)}


def sync_meeting_to_project_payload(project_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    project = read_project(project_id)
    storage_root = account_workspace_root()
    manifest_path = WorkspaceFiles(storage_root).resolve(required_string(payload, "manifest_path"))
    archive_root = (storage_root / "meet_files" / "会议项目").resolve()
    try:
        manifest_path.relative_to(archive_root)
    except ValueError as error:
        raise ValueError("只能同步当前账户会议归档中的纪要。") from error
    if manifest_path.name != "manifest.json" or not manifest_path.is_file():
        raise ValueError("会议归档清单不存在。")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("会议归档清单格式无效。")
    title = re.sub(r"\s+", " ", str(manifest.get("title") or manifest_path.parent.name)).strip()
    title = title[:120] or "未命名会议"
    canonical = manifest.get("canonical_outputs")
    if not isinstance(canonical, dict):
        raise ValueError("会议归档还没有可同步的正式产出。")

    source_paths: list[Path] = []
    seen: set[Path] = set()
    for key in ("asr", "internal", "work_md", "work_docx"):
        raw_path = str(canonical.get(key) or "").strip()
        if not raw_path:
            continue
        source = WorkspaceFiles(storage_root).resolve(raw_path)
        if not source.is_file() or source in seen:
            continue
        seen.add(source)
        source_paths.append(source)
    if not source_paths:
        raise ValueError("该会议暂无可同步的纪要文件。")

    target_dir = project_dir(project_id) / "sources" / "会议纪要" / sanitize_filename(title)
    target_dir.mkdir(parents=True, exist_ok=True)
    synced_files: list[dict[str, Any]] = []
    copied_count = 0
    unchanged_count = 0
    for source in source_paths:
        target = target_dir / source.name
        if target.is_file() and files_have_same_content(source, target):
            unchanged_count += 1
        else:
            shutil.copy2(source, target)
            copied_count += 1
        synced_files.append(file_item_payload(target))

    now = int(time.time())
    meeting_syncs = project.get("meeting_syncs")
    if not isinstance(meeting_syncs, dict):
        meeting_syncs = {}
    meeting_syncs[str(manifest_path.relative_to(storage_root))] = {
        "title": title,
        "synced_at": now,
        "files": [item["path"] for item in synced_files],
    }
    project["meeting_syncs"] = meeting_syncs
    project["updated_at"] = now
    write_project(project)
    return {
        "ok": True,
        "meeting_title": title,
        "copied_count": copied_count,
        "unchanged_count": unchanged_count,
        "files": synced_files,
        "project": project_payload(project),
    }


def files_have_same_content(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    left_hash = hashlib.sha256()
    right_hash = hashlib.sha256()
    with left.open("rb") as left_stream, right.open("rb") as right_stream:
        for chunk in iter(lambda: left_stream.read(1024 * 1024), b""):
            left_hash.update(chunk)
        for chunk in iter(lambda: right_stream.read(1024 * 1024), b""):
            right_hash.update(chunk)
    return left_hash.digest() == right_hash.digest()


def sanitize_conversation_archive_items(items: list[Any]) -> list[Any]:
    return [sanitize_conversation_archive_item(item) for item in items]


def sanitize_conversation_archive_item(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    clean = dict(item)
    messages = clean.get("messages")
    if isinstance(messages, list):
        clean["messages"] = [sanitize_conversation_message(message) for message in messages]
    activities = clean.get("activities")
    if isinstance(activities, dict):
        clean["activities"] = {
            key: sanitize_activity_record(record) for key, record in activities.items()
        }
    return clean


def sanitize_conversation_message(message: Any) -> Any:
    if not isinstance(message, dict):
        return message
    clean = dict(message)
    content = clean.get("content")
    if isinstance(content, str) and contains_tool_call_markup(content):
        cleaned_content = strip_tool_call_markup(content).strip()
        clean["content"] = cleaned_content or "工具调用过程已隐藏。请重新发送上一条请求继续。"
    return clean


def sanitize_activity_record(record: Any) -> Any:
    if not isinstance(record, dict):
        return record
    clean = dict(record)
    events = clean.get("events")
    if isinstance(events, list):
        clean["events"] = [sanitize_activity_event(event) for event in events]
    return clean


def sanitize_activity_event(event: Any) -> Any:
    if not isinstance(event, dict):
        return event
    clean = dict(event)
    for key in ("detail", "content"):
        value = clean.get(key)
        if isinstance(value, str) and contains_tool_call_markup(value):
            cleaned_value = strip_tool_call_markup(value).strip()
            clean[key] = cleaned_value or "工具调用过程已隐藏。"
    return clean


def file_item_payload(path: Path) -> dict[str, Any]:
    stat = path.stat()
    extension = path.suffix.lower()
    mime_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    preview_mode = preview_mode_for_file(extension, mime_type)
    return {
        "path": account_relative_path(path),
        "name": path.name,
        "size": stat.st_size,
        "modified": int(stat.st_mtime),
        "extension": extension,
        "mime_type": mime_type,
        "kind": classify_attachment(extension, mime_type),
        "previewable": preview_mode != "none",
        "preview_mode": preview_mode,
    }


def dedupe_file_library_items(
    files: list[dict[str, Any]],
    attachment_index: dict[str, Any],
) -> list[dict[str, Any]]:
    path_fingerprints = attachment_path_fingerprints(attachment_index)
    seen: set[str] = set()
    visible: list[dict[str, Any]] = []
    for item in files:
        key = file_library_dedupe_key(item, path_fingerprints)
        if key:
            if key in seen:
                continue
            seen.add(key)
        visible.append(item)
    return visible


def attachment_path_fingerprints(index: dict[str, Any]) -> dict[str, str]:
    fingerprints: dict[str, str] = {}
    for fingerprint, item in index.items():
        if not isinstance(item, dict):
            continue
        rel_path = str(item.get("path") or "")
        if rel_path:
            fingerprints[rel_path] = str(fingerprint)
    return fingerprints


def file_library_dedupe_key(item: dict[str, Any], path_fingerprints: dict[str, str]) -> str:
    rel_path = str(item.get("path") or "")
    if not rel_path:
        return ""
    path = (account_workspace_root() / rel_path).resolve()
    if not is_attachment_file(path):
        return ""
    if rel_path in path_fingerprints:
        return f"attachment:{path_fingerprints[rel_path]}"
    original_name = legacy_attachment_original_name(Path(rel_path).name)
    if original_name:
        return f"legacy-attachment:{original_name}|size:{int(item.get('size') or 0)}"
    return ""


def legacy_attachment_original_name(name: str) -> str:
    match = re.match(r"^\d{8}-\d{6}-(.+)$", name)
    if not match:
        return ""
    return match.group(1)


def is_file_library_visible(path: Path) -> bool:
    """Only expose user-added files and final office artifacts in the file library."""

    try:
        relative = path.relative_to(account_workspace_root())
    except ValueError:
        return False

    parts = relative.parts
    if any(part.startswith(".") for part in parts):
        return False
    if is_attachment_file(path):
        return True
    if len(parts) >= 3 and parts[0] == "meet_files" and parts[1] == "office_extracts":
        return path.suffix.lower() == ".md"
    if len(parts) >= 3 and parts[0] == "meet_files" and parts[1] == "realtime_transcripts":
        return path.suffix.lower() == ".md"
    if len(parts) >= 3 and parts[0] == "meet_files" and parts[1] == "资料项目":
        return path.suffix.lower() in FILE_LIBRARY_OUTPUT_EXTENSIONS
    if len(parts) >= 2 and parts[0] == "meet_files" and parts[1] == "file_previews":
        return False
    if path.suffix.lower() not in FILE_LIBRARY_OUTPUT_EXTENSIONS:
        return False
    if parts[0] != "meet_files":
        return False
    return any(
        marker in path.name
        for marker in FILE_LIBRARY_OUTPUT_MARKERS
    )


FILE_LIBRARY_OUTPUT_EXTENSIONS = {
    ".md",
    ".docx",
    ".pdf",
    ".txt",
    ".xlsx",
    ".pptx",
}
FILE_LIBRARY_OUTPUT_MARKERS = (
    "内部留档版",
    "工作提交版",
    "完善版",
    "会议纪要",
    "会议沟通内容整理",
    "座谈沟通会议纪要",
    "实时转写",
)


def is_attachment_file(path: Path) -> bool:
    try:
        relative = path.relative_to(account_workspace_root())
    except ValueError:
        return False
    parts = relative.parts
    return len(parts) >= 3 and parts[0] == "meet_files" and parts[1] == "attachments"


def read_file_payload(path: str, *, max_chars: int) -> dict[str, Any]:
    storage_root = account_workspace_root()
    workspace = WorkspaceFiles(storage_root)
    file_path = workspace.resolve(path)
    if not file_path.is_file():
        raise ValueError(f"Not a file: {path}")
    stat = file_path.stat()
    extension = file_path.suffix.lower()
    mime_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
    preview_mode = preview_mode_for_file(extension, mime_type)
    rendered_path = ""
    rendered_url = ""
    if extension in OFFICE_TO_PDF_EXTENSIONS:
        rendered_file = ensure_office_pdf_preview(file_path)
        rendered_path = str(rendered_file.relative_to(storage_root))
        rendered_url = raw_file_url(rendered_path)
        preview_mode = "pdf"
    previewable = preview_mode != "none"
    text = ""
    truncated = False
    if extension in TEXT_PREVIEW_EXTENSIONS:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        truncated = len(text) > max_chars
        if truncated:
            text = text[:max_chars]
    source_url = raw_file_url(str(file_path.relative_to(storage_root)))
    return {
        "path": str(file_path.relative_to(storage_root)),
        "name": file_path.name,
        "content": text,
        "truncated": truncated,
        "chars": len(text),
        "size": stat.st_size,
        "modified": int(stat.st_mtime),
        "extension": extension,
        "mime_type": mime_type,
        "kind": classify_attachment(extension, mime_type),
        "previewable": previewable,
        "preview_mode": preview_mode,
        "preview_url": rendered_url or (source_url if previewable and preview_mode not in {"text", "markdown"} else ""),
        "source_url": source_url,
        "rendered_path": rendered_path,
        "editable": extension in EDITABLE_FILE_EXTENSIONS,
    }


EDITABLE_FILE_EXTENSIONS = TEXT_PREVIEW_EXTENSIONS | {".doc", ".docx", ".odt"}


def preview_mode_for_file(extension: str, mime_type: str) -> str:
    if extension in MARKDOWN_PREVIEW_EXTENSIONS:
        return "markdown"
    if extension in TEXT_PREVIEW_EXTENSIONS:
        return "text"
    if extension in PDF_PREVIEW_EXTENSIONS:
        return "pdf"
    if extension in IMAGE_PREVIEW_EXTENSIONS or mime_type.startswith("image/"):
        return "image"
    if extension in AUDIO_PREVIEW_EXTENSIONS or mime_type.startswith("audio/"):
        return "audio"
    if extension in VIDEO_PREVIEW_EXTENSIONS or mime_type.startswith("video/"):
        return "video"
    if extension in OFFICE_TO_PDF_EXTENSIONS:
        return "pdf"
    return "none"


def ensure_office_pdf_preview(source_path: Path) -> Path:
    storage_root = account_workspace_root()
    rel = source_path.relative_to(storage_root)
    digest = hashlib.sha1(str(rel).encode("utf-8")).hexdigest()[:16]
    output_dir = storage_root / "meet_files" / "file_previews" / digest
    output_path = output_dir / f"{source_path.stem}.pdf"
    if output_path.is_file() and output_path.stat().st_mtime >= source_path.stat().st_mtime:
        return output_path
    return convert_office_to_pdf(source_path, output_dir=output_dir, workspace_root=storage_root)


def raw_file_url(path: str) -> str:
    return f"/api/file/raw?path={quote(path)}"


def open_workspace_file_payload(payload: dict[str, Any], *, reveal: bool) -> dict[str, Any]:
    workspace = WorkspaceFiles(account_workspace_root())
    file_path = workspace.resolve(required_string(payload, "path"))
    if not file_path.exists():
        raise ValueError(f"Path does not exist: {payload['path']}")
    command = ["open", "-R", str(file_path)] if reveal else ["open", str(file_path)]
    result = subprocess.run(
        command,
        cwd=WORKSPACE_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"打开文件失败：{detail}")
    return {"ok": True, "path": str(file_path.relative_to(WORKSPACE_ROOT)), "action": "reveal" if reveal else "open"}


def add_attachment_payload(payload: dict[str, Any]) -> dict[str, Any]:
    name = sanitize_filename(required_string(payload, "name"))
    encoded = required_string(payload, "content_base64")
    try:
        data = base64.b64decode(encoded, validate=True)
    except Exception as error:
        raise ValueError("Invalid attachment content_base64.") from error

    if not data:
        raise ValueError("Attachment is empty.")

    mime_type = str(payload.get("mime_type") or "application/octet-stream")
    storage_root = account_workspace_root()
    target_dir = (storage_root / "meet_files" / "attachments").resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    index = load_attachment_index(target_dir)
    fingerprint = attachment_fingerprint(payload, name=name, size=len(data))
    if fingerprint:
        existing_path = indexed_attachment_path(index, fingerprint, target_dir)
        if existing_path is not None:
            return {
                "attachment": attachment_payload_from_path(
                    existing_path,
                    display_name=name,
                    mime_type=mime_type,
                    deduplicated=True,
                )
            }

    uploaded_at = time.strftime("%Y%m%d-%H%M%S")
    target_path = unique_path(target_dir / f"{uploaded_at}-{name}")
    target_path.write_bytes(data)
    if fingerprint:
        index[fingerprint] = {
            "path": str(target_path.relative_to(storage_root)),
            "name": name,
            "size": len(data),
            "mime_type": mime_type,
            "source_path": clean_optional_string(payload.get("source_path")),
            "relative_path": clean_optional_string(payload.get("relative_path")),
            "last_modified": optional_int(payload.get("last_modified")),
            "saved_at": int(time.time()),
            "recording_metadata": probe_audio_metadata(target_path),
        }
        save_attachment_index(target_dir, index)
    return {
        "attachment": attachment_payload_from_path(
            target_path,
            display_name=name,
            mime_type=mime_type,
            deduplicated=False,
        )
    }


ATTACHMENT_INDEX_NAME = ".attachments_index.json"


def load_attachment_index(directory: Path) -> dict[str, Any]:
    index_path = directory / ATTACHMENT_INDEX_NAME
    if not index_path.is_file():
        return {}
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def save_attachment_index(directory: Path, index: dict[str, Any]) -> None:
    index_path = directory / ATTACHMENT_INDEX_NAME
    tmp_path = directory / f"{ATTACHMENT_INDEX_NAME}.tmp"
    tmp_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(index_path)


def indexed_attachment_path(index: dict[str, Any], fingerprint: str, target_dir: Path) -> Path | None:
    item = index.get(fingerprint)
    if not isinstance(item, dict):
        return None
    rel_path = str(item.get("path") or "")
    if not rel_path:
        return None
    candidate = (account_workspace_root() / rel_path).resolve()
    try:
        candidate.relative_to(target_dir)
    except ValueError:
        return None
    if candidate.is_file():
        return candidate
    return None


def attachment_fingerprint(payload: dict[str, Any], *, name: str, size: int) -> str:
    source_path = clean_optional_string(payload.get("source_path"))
    if source_path:
        return f"source_path:{source_path}"
    relative_path = clean_optional_string(payload.get("relative_path"))
    last_modified = optional_int(payload.get("last_modified"))
    if relative_path:
        return f"relative_path:{relative_path}|name:{name}|size:{size}|mtime:{last_modified or 0}"
    if last_modified is None:
        return ""
    return f"name:{name}|size:{size}|mtime:{last_modified}"


def clean_optional_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def attachment_payload_from_path(
    path: Path,
    *,
    display_name: str,
    mime_type: str,
    deduplicated: bool,
) -> dict[str, Any]:
    stat = path.stat()
    payload = {
        "name": display_name,
        "path": str(path.relative_to(account_workspace_root())),
        "size": stat.st_size,
        "mime_type": mime_type,
        "extension": path.suffix.lower(),
        "kind": classify_attachment(path.suffix.lower(), mime_type),
        "deduplicated": deduplicated,
    }
    recording_metadata = probe_audio_metadata(path)
    if recording_metadata:
        payload["recording_metadata"] = recording_metadata
    return payload


def speech_vad_payload(payload: dict[str, Any]) -> dict[str, Any]:
    frames_raw = payload.get("frames_base64")
    if not isinstance(frames_raw, list):
        raise ValueError("frames_base64 must be a list.")
    frames_base64 = [str(frame) for frame in frames_raw if str(frame)]
    if not frames_base64:
        raise ValueError("frames_base64 must contain at least one frame.")
    if len(frames_base64) > 100:
        raise ValueError("frames_base64 can contain at most 100 frames.")

    sample_rate = int(payload.get("sample_rate") or 16000)
    frame_ms = int(payload.get("frame_ms") or 30)
    aggressiveness = int(payload.get("aggressiveness") or 3)
    try:
        result = get_vad_worker().classify(
            frames_base64,
            sample_rate=sample_rate,
            frame_ms=frame_ms,
            aggressiveness=aggressiveness,
        )
        speech_frames = result.get("speech_frames") or []
        if not isinstance(speech_frames, list):
            speech_frames = []
        return {
            "available": True,
            "provider": str(result.get("provider") or "webrtcvad"),
            "sample_rate": sample_rate,
            "frame_ms": frame_ms,
            "speech_frames": [bool(item) for item in speech_frames],
            "speech_count": int(result.get("speech_count") or 0),
        }
    except Exception as error:
        return {
            "available": False,
            "provider": "webrtcvad",
            "sample_rate": sample_rate,
            "frame_ms": frame_ms,
            "speech_frames": [],
            "speech_count": 0,
            "error": str(error),
        }


def transcribe_speech_payload(payload: dict[str, Any]) -> dict[str, Any]:
    name = sanitize_filename(str(payload.get("name") or "voice-input.webm"))
    mime_type = str(payload.get("mime_type") or "audio/webm").strip() or "audio/webm"
    use_denoise = bool(payload.get("use_denoise"))
    skip_if_silent = bool(payload.get("skip_if_silent"))
    if not Path(name).suffix:
        name = f"{name}{extension_for_audio_mime(mime_type)}"

    encoded = required_string(payload, "content_base64")
    try:
        data = base64.b64decode(encoded, validate=True)
    except Exception as error:
        raise ValueError("Invalid speech content_base64.") from error

    if not data:
        raise ValueError("录音内容为空，请重新录制。")
    if len(data) > 100 * 1024 * 1024:
        raise ValueError("录音文件过大，请先缩短录音后再转写。")

    uploaded_at = time.strftime("%Y%m%d-%H%M%S")
    session_dir = (account_workspace_root() / "meet_files" / "voice_inputs" / uploaded_at).resolve()
    session_dir.mkdir(parents=True, exist_ok=True)

    source_path = unique_path(session_dir / name)
    source_path.write_bytes(data)

    wav_path = session_dir / "voice_16k.wav"
    filter_chain = realtime_audio_filter_chain(use_denoise=use_denoise)
    command = [
        require_executable("ffmpeg", "FFmpeg 未安装，无法把浏览器录音转换为本地 ASR 可用的 WAV。"),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
        "-ac",
        "1",
        "-ar",
        "16000",
    ]
    if filter_chain:
        command.extend(["-af", filter_chain])
    command.append(str(wav_path))
    run_process(command, timeout_seconds=60, label="音频格式转换")

    signal_stats = wav_signal_stats(wav_path)
    if skip_if_silent and not signal_stats["has_voice_like_signal"]:
        transcript_path = write_worker_transcript(
            session_dir,
            "",
            {
                "text": "",
                "skipped": True,
                "reason": "voice_gate",
                "signal": signal_stats,
            },
        )
        return {
            "text": "",
            "audio_path": str(source_path.relative_to(WORKSPACE_ROOT)),
            "wav_path": str(wav_path.relative_to(WORKSPACE_ROOT)),
            "transcript_path": str(transcript_path.relative_to(WORKSPACE_ROOT)),
            "engine": "voice-gate",
            "asr_elapsed_ms": 0,
            "filter_chain": filter_chain,
            "signal": signal_stats,
            "skipped": True,
        }

    asr_started_at = time.perf_counter()
    engine = "qwen3-asr-worker"
    try:
        worker_result = transcribe_with_asr_worker(wav_path, timeout_seconds=180)
        text = str(worker_result.get("text") or "").strip()
        transcript_path = write_worker_transcript(session_dir, text, worker_result)
        asr_elapsed_ms = int(worker_result.get("elapsed_ms") or ((time.perf_counter() - asr_started_at) * 1000))
    except Exception as worker_error:
        engine = "qwen3-asr-mlx-cli"
        transcript_path = run_local_qwen3_asr(wav_path, session_dir / "asr")
        text = transcript_path.read_text(encoding="utf-8", errors="replace").strip()
        asr_elapsed_ms = int((time.perf_counter() - asr_started_at) * 1000)
        print(f"[asr] worker unavailable; used MLX CLI: {worker_error}")
    return {
        "text": text,
        "audio_path": str(source_path.relative_to(WORKSPACE_ROOT)),
        "wav_path": str(wav_path.relative_to(WORKSPACE_ROOT)),
        "transcript_path": str(transcript_path.relative_to(WORKSPACE_ROOT)),
        "engine": engine,
        "asr_elapsed_ms": asr_elapsed_ms,
        "filter_chain": filter_chain,
        "signal": signal_stats,
        "skipped": False,
    }


def realtime_audio_filter_chain(*, use_denoise: bool) -> str:
    filters = [
        "highpass=f=80",
        "lowpass=f=7800",
    ]
    if use_denoise:
        filters.append("afftdn=nf=-25")
    filters.extend(
        [
            "dynaudnorm=f=150:g=15",
            "loudnorm=I=-16:LRA=11:TP=-1.5",
        ]
    )
    return ",".join(filters)


def wav_signal_stats(wav_path: Path) -> dict[str, Any]:
    try:
        with wave.open(str(wav_path), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            frame_count = wav.getnframes()
            data = wav.readframes(frame_count)
        if channels != 1 or sample_width != 2 or not data:
            return {
                "duration_ms": 0,
                "rms": 0,
                "max_rms": 0,
                "active_ratio": 0,
                "has_voice_like_signal": True,
            }
        samples = array.array("h")
        samples.frombytes(data)
        if samples.itemsize != 2:
            return {
                "duration_ms": 0,
                "rms": 0,
                "max_rms": 0,
                "active_ratio": 0,
                "has_voice_like_signal": True,
            }
        duration_ms = int(frame_count / sample_rate * 1000) if sample_rate else 0
        if not samples:
            return {
                "duration_ms": duration_ms,
                "rms": 0,
                "max_rms": 0,
                "active_ratio": 0,
                "has_voice_like_signal": False,
            }
        total_power = 0
        for sample in samples:
            total_power += int(sample) * int(sample)
        rms = math.sqrt(total_power / len(samples))
        frame_size = max(1, int(sample_rate * 0.03))
        active_frames = 0
        total_frames = 0
        max_rms = 0.0
        threshold = 360
        for start in range(0, len(samples), frame_size):
            frame = samples[start : start + frame_size]
            if len(frame) < frame_size // 2:
                continue
            frame_power = 0
            for sample in frame:
                frame_power += int(sample) * int(sample)
            frame_rms = math.sqrt(frame_power / len(frame))
            max_rms = max(max_rms, frame_rms)
            total_frames += 1
            if frame_rms >= threshold:
                active_frames += 1
        active_ratio = active_frames / total_frames if total_frames else 0
        has_voice_like_signal = duration_ms >= 350 and (max_rms >= 520 or active_ratio >= 0.08 or rms >= 260)
        return {
            "duration_ms": duration_ms,
            "rms": round(rms, 1),
            "max_rms": round(max_rms, 1),
            "active_ratio": round(active_ratio, 3),
            "has_voice_like_signal": has_voice_like_signal,
        }
    except Exception as error:
        return {
            "duration_ms": 0,
            "rms": 0,
            "max_rms": 0,
            "active_ratio": 0,
            "has_voice_like_signal": True,
            "warning": str(error),
        }


def format_seconds(seconds: float) -> str:
    total = int(round(max(0.0, seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def save_realtime_transcript_payload(payload: dict[str, Any]) -> dict[str, Any]:
    title = str(payload.get("title") or "实时会议转写").strip() or "实时会议转写"
    segments_raw = payload.get("segments")
    if not isinstance(segments_raw, list):
        raise ValueError("segments must be a list.")

    segments: list[dict[str, Any]] = []
    for raw in segments_raw:
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text") or "").strip()
        if not text:
            continue
        try:
            index = int(raw.get("index") or len(segments) + 1)
        except (TypeError, ValueError):
            index = len(segments) + 1
        try:
            started_at = int(raw.get("started_at") or raw.get("startedAt") or 0)
        except (TypeError, ValueError):
            started_at = 0
        try:
            finished_at = int(raw.get("finished_at") or raw.get("finishedAt") or 0)
        except (TypeError, ValueError):
            finished_at = 0
        segments.append(
            {
                "index": index,
                "text": text,
                "started_at": started_at,
                "finished_at": finished_at,
            }
        )

    if not segments:
        raise ValueError("没有可保存的转写文本。")

    saved_at = time.strftime("%Y-%m-%d %H:%M:%S")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    storage_root = account_workspace_root()
    output_dir = (storage_root / "meet_files" / "realtime_transcripts").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = unique_path(output_dir / f"{stamp}-{sanitize_filename(title)}.md")

    full_text = "\n\n".join(segment["text"] for segment in segments)
    lines = [
        f"# {title}",
        "",
        f"- 保存时间：{saved_at}",
        "- 来源：浏览器麦克风实时分片识别",
        f"- 片段数：{len(segments)}",
        "",
        "## 完整转写",
        "",
        full_text,
        "",
        "## 分片记录",
        "",
    ]
    for segment in segments:
        lines.extend(
            [
                f"### 片段 {segment['index']}",
                "",
                segment["text"],
                "",
            ]
        )

    content = "\n".join(lines).rstrip() + "\n"
    output_path.write_text(content, encoding="utf-8")
    return {
        "ok": True,
        "title": title,
        "path": str(output_path.relative_to(storage_root)),
        "segments": len(segments),
        "chars": len(content),
    }


def write_worker_transcript(session_dir: Path, text: str, worker_result: dict[str, Any]) -> Path:
    output_dir = session_dir / "asr" / "worker"
    output_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = output_dir / "transcript.txt"
    raw_path = output_dir / "raw_result.json"
    transcript_path.write_text(text.strip() + "\n", encoding="utf-8")
    raw_path.write_text(json.dumps(worker_result, ensure_ascii=False, indent=2), encoding="utf-8")
    return transcript_path


def get_asr_worker() -> LocalQwen3ASRWorker:
    global ASR_WORKER
    if ASR_WORKER is None or ASR_WORKER.workspace_root != WORKSPACE_ROOT:
        ASR_WORKER = LocalQwen3ASRWorker(WORKSPACE_ROOT)
    return ASR_WORKER


def asr_idle_timeout_seconds() -> float:
    raw_value = os.environ.get("WORK_AGENT_ASR_IDLE_TIMEOUT_SECONDS", "90")
    try:
        return max(0.0, float(raw_value))
    except ValueError:
        return 90.0


def _cancel_asr_idle_timer_locked() -> None:
    global ASR_WORKER_IDLE_TIMER
    if ASR_WORKER_IDLE_TIMER is not None:
        ASR_WORKER_IDLE_TIMER.cancel()
        ASR_WORKER_IDLE_TIMER = None


def _stop_idle_asr_worker() -> None:
    global ASR_WORKER, ASR_WORKER_IDLE_TIMER
    with ASR_WORKER_LIFECYCLE_LOCK:
        ASR_WORKER_IDLE_TIMER = None
        if ASR_ACTIVE_REQUESTS:
            return
        worker = ASR_WORKER
        ASR_WORKER = None
    if worker is not None:
        worker.stop()
        print("[asr] Qwen3-ASR worker unloaded after idle timeout")


def _schedule_asr_idle_shutdown_locked() -> None:
    global ASR_WORKER_IDLE_TIMER
    _cancel_asr_idle_timer_locked()
    timeout_seconds = asr_idle_timeout_seconds()
    if timeout_seconds <= 0:
        threading.Thread(target=_stop_idle_asr_worker, name="qwen3-asr-worker-unload", daemon=True).start()
        return
    ASR_WORKER_IDLE_TIMER = threading.Timer(timeout_seconds, _stop_idle_asr_worker)
    ASR_WORKER_IDLE_TIMER.name = "qwen3-asr-worker-idle-unload"
    ASR_WORKER_IDLE_TIMER.daemon = True
    ASR_WORKER_IDLE_TIMER.start()


def transcribe_with_asr_worker(audio_path: Path, *, timeout_seconds: int = 180) -> dict[str, Any]:
    global ASR_ACTIVE_REQUESTS
    with ASR_WORKER_LIFECYCLE_LOCK:
        _cancel_asr_idle_timer_locked()
        worker = get_asr_worker()
        ASR_ACTIVE_REQUESTS += 1
    try:
        return worker.transcribe(audio_path, timeout_seconds=timeout_seconds)
    finally:
        with ASR_WORKER_LIFECYCLE_LOCK:
            ASR_ACTIVE_REQUESTS = max(0, ASR_ACTIVE_REQUESTS - 1)
            if ASR_ACTIVE_REQUESTS == 0 and ASR_WORKER is worker:
                _schedule_asr_idle_shutdown_locked()


def stop_asr_worker() -> None:
    global ASR_WORKER
    with ASR_WORKER_LIFECYCLE_LOCK:
        _cancel_asr_idle_timer_locked()
        worker = ASR_WORKER
        ASR_WORKER = None
    if worker is not None:
        worker.stop()


def get_vad_worker() -> LocalWebRtcVadWorker:
    global VAD_WORKER
    if VAD_WORKER is None or VAD_WORKER.workspace_root != WORKSPACE_ROOT:
        VAD_WORKER = LocalWebRtcVadWorker(WORKSPACE_ROOT)
    return VAD_WORKER


def run_local_qwen3_asr(audio_path: Path, output_root: Path) -> Path:
    python_path = local_asr_python()
    script_path = WORKSPACE_ROOT / "meeting_audio_minutes" / "scripts" / "transcribe_qwen3_asr_chunked.py"
    if not script_path.is_file():
        raise FileNotFoundError(f"未找到本地 Qwen3-ASR 脚本：{script_path}")

    cache_dir = WORKSPACE_ROOT / "meeting_audio_minutes" / "model_cache"
    settings = load_asr_settings()
    command = [
        str(python_path),
        str(script_path),
        str(audio_path),
        "--output-dir",
        str(output_root),
        "--backend",
        "mlx",
        "--model-id",
        str(settings["model_id"]),
        "--cache-dir",
        str(cache_dir),
        "--device",
        "mlx-metal",
        "--language",
        "zh",
        "--chunk-mode",
        "fixed",
        "--chunk-seconds",
        "120",
        "--max-new-tokens",
        "1024",
        "--workers",
        "1",
    ]

    run_process(
        command,
        timeout_seconds=900,
        label="本地 Qwen3-ASR 转写",
    )

    candidates = sorted(
        output_root.rglob("transcript.txt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("本地 ASR 已运行，但没有生成 transcript.txt。")
    return candidates[0]


def local_asr_python() -> Path:
    candidate = project_agent_python(WORKSPACE_ROOT)
    if candidate is not None:
        return candidate
    raise FileNotFoundError(
        "项目唯一 Python 环境不存在。请运行：scripts/runtime_env.sh bootstrap"
    )


def local_vad_python() -> Path:
    env_python = os.getenv("WORK_AGENT_VAD_PYTHON", "").strip()
    candidates = [
        Path(env_python) if env_python else None,
        project_agent_python(WORKSPACE_ROOT),
    ]
    checked: list[str] = []
    for candidate in candidates:
        if candidate is None:
            continue
        if not candidate.exists():
            checked.append(str(candidate))
            continue
        if python_has_module(candidate, "webrtcvad"):
            return candidate
        checked.append(str(candidate))
    raise FileNotFoundError(
        "未找到可用的 WebRTC VAD Python 环境。请在项目 venv 中安装："
        "scripts/runtime_env.sh bootstrap。"
        f" 已检查：{', '.join(checked)}"
    )


def python_has_module(python_path: Path, module_name: str) -> bool:
    try:
        result = subprocess.run(
            [
                str(python_path),
                "-c",
                (
                    "import importlib.util, sys; "
                    f"sys.exit(0 if importlib.util.find_spec({module_name!r}) else 1)"
                ),
            ],
            cwd=WORKSPACE_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def require_executable(name: str, message: str) -> str:
    executable = find_runtime_executable(name, WORKSPACE_ROOT)
    if not executable:
        raise FileNotFoundError(message)
    return executable


def run_process(command: list[str], *, timeout_seconds: int, label: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=WORKSPACE_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=True,
        )
    except subprocess.TimeoutExpired as error:
        raise TimeoutError(f"{label}超时，请缩短录音或稍后重试。") from error
    except subprocess.CalledProcessError as error:
        output = "\n".join(part for part in (error.stdout, error.stderr) if part).strip()
        output = output[-2000:] if output else "无详细输出"
        raise RuntimeError(f"{label}失败：{output}") from error


def extension_for_audio_mime(mime_type: str) -> str:
    normalized = mime_type.split(";", 1)[0].strip().lower()
    mapping = {
        "audio/webm": ".webm",
        "audio/ogg": ".ogg",
        "audio/mp4": ".m4a",
        "audio/mpeg": ".mp3",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
    }
    return mapping.get(normalized, ".webm")


def run_agent_payload(payload: dict[str, Any]) -> dict[str, Any]:
    goal = required_string(payload, "goal")
    registry = load_registry()
    profile = registry.get(str(payload.get("profile") or registry.default_profile))
    client = OpenAICompatibleClient()
    tools = build_default_tools(
        WORKSPACE_ROOT,
        client,
        profile,
        data_workspace=account_workspace_root(),
        include_shared_tools=current_auth_user().role == "admin",
        enabled_skill_ids=enabled_skill_ids(),
    )
    max_steps = int(payload.get("max_steps") or DEFAULT_MAX_STEPS)
    max_steps = max(1, min(max_steps, 60))
    agent = ReActAgent(
        client=client,
        profile=profile,
        tools=tools,
        max_steps=max_steps,
        extra_system_context=agent_system_context(),
    )
    result = agent.run(goal)
    return {"result": asdict(result)}


def turn_payload(turn_id: str) -> dict[str, Any]:
    turn = get_turn_store().load(turn_id)
    return {
        "ok": True,
        "turn": turn.to_payload(),
    }


def turn_events_payload(turn_id: str, *, after: int = -1) -> dict[str, Any]:
    store = get_turn_store()
    turn = store.load(turn_id)
    return {
        "ok": True,
        "turn_id": turn.id,
        "conversation_id": turn.conversation_id,
        "status": turn.status,
        "cancel_requested": turn.cancel_requested,
        "latest_event_index": turn.latest_event_index,
        "events": store.events_after(turn.id, after=after),
    }


def cancel_turn_payload(turn_id: str) -> dict[str, Any]:
    turn = get_turn_store().request_cancel(turn_id)
    return {
        "ok": True,
        "turn_id": turn.id,
        "conversation_id": turn.conversation_id,
        "status": turn.status,
        "cancel_requested": turn.cancel_requested,
    }


def rewind_session_or_rebuild_from_display(
    store: SessionStore,
    session: Any,
    messages: list[dict[str, Any]],
    rewind_user_ordinal: int | None,
    debug_trace: DebugTrace | None = None,
) -> bool:
    if rewind_user_ordinal is None:
        return False
    if store.has_user_message_ordinal(session, rewind_user_ordinal):
        store.rewind_before_user_message(session, rewind_user_ordinal)
        return False
    bootstrapped = store.rebuild_from_display_messages(session, messages, exclude_last_user=True)
    if debug_trace is not None:
        debug_trace.emit(
            "session_rewind_rebuilt_from_display",
            rewind_user_ordinal=rewind_user_ordinal,
            incoming_message_count=len(messages),
            rebuilt_message_count=len(session.messages),
            bootstrapped=bootstrapped,
        )
    return bootstrapped


def run_agent_chat_payload(payload: dict[str, Any]) -> dict[str, Any]:
    registry = load_registry()
    profile = registry.get(str(payload.get("profile") or registry.default_profile))
    client = OpenAICompatibleClient()
    messages = sanitize_chat_messages(payload.get("messages"))
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("Chat messages must end with a user message.")
    context_file_paths = sanitize_context_file_paths(payload.get("context_file_paths"))
    context_file_paths, project_context, project_id = resolve_project_chat_context(
        payload.get("project_id"), context_file_paths
    )
    conversation_id = resolve_conversation_id(payload, messages)
    skill_hint = normalize_skill_hint(payload.get("skill_hint"))
    max_steps = int(payload.get("max_steps") or DEFAULT_MAX_STEPS)
    max_steps = max(1, min(max_steps, 60))
    reasoning_effort = normalize_reasoning_effort(payload.get("reasoning_effort"))
    debug_trace = DebugTrace(
        WORKSPACE_ROOT,
        conversation_id=conversation_id,
        route="/api/agent/chat",
        profile=profile.name,
        model=profile.model,
    )
    debug_trace.emit(
        "http_chat_start",
        incoming_message_count=len(messages),
        latest_user_chars=len(messages[-1]["content"]),
        skill_hint=skill_hint,
        context_file_path_count=len(context_file_paths),
        project_id=project_id,
        max_steps=max_steps,
    )

    store = get_session_store()
    session = store.load(conversation_id)
    rewind_user_ordinal = sanitize_rewind_user_message_ordinal(payload)
    if rewind_user_ordinal is not None:
        get_turn_store().discard_pending_for_conversation(conversation_id)
    bootstrapped = rewind_session_or_rebuild_from_display(
        store,
        session,
        messages,
        rewind_user_ordinal,
        debug_trace,
    )
    if not bootstrapped:
        bootstrapped = store.bootstrap_from_display_messages(session, messages, exclude_last_user=True)
    if not session.summary:
        session.summary = sanitize_conversation_summary(payload.get("conversation_summary"))
        session.summary_message_count = sanitize_summary_message_count(
            payload.get("conversation_summary_message_count")
        )
    session.metadata["project_id"] = project_id
    store.append_user_message(session, messages[-1]["content"])
    prepared_context = prepare_session_memory(client, profile, session)
    store.save(session)
    debug_trace.emit(
        "session_prepared",
        bootstrapped=bootstrapped,
        stored_message_count=len(session.messages),
        prepared_message_count=len(prepared_context.messages),
        summary_chars=len(session.summary),
        summary_message_count=session.summary_message_count,
        compacted=prepared_context.compacted,
        estimated_tokens=prepared_context.estimated_tokens,
    )

    tools = build_default_tools(
        WORKSPACE_ROOT,
        client,
        profile,
        data_workspace=account_workspace_root(),
        include_shared_tools=current_auth_user().role == "admin",
        session_store=store,
        conversation_id=conversation_id,
        project_id=project_id,
        enabled_skill_ids=enabled_skill_ids(),
    )
    agent = ReActAgent(
        client=client,
        profile=profile,
        tools=tools,
        max_steps=max_steps,
        debug_trace=debug_trace,
        extra_system_context=agent_system_context()
        + project_context
        + render_history_recall_system_context(session.summary_message_count, project_id=project_id)
        + build_chat_session_system_context(
            session.messages,
            skill_hint=skill_hint,
            context_file_paths=context_file_paths,
        ),
        reasoning_effort=reasoning_effort,
    )
    runtime_messages = list(prepared_context.messages)
    runtime_message_count_before_run = len(runtime_messages)
    result = agent.run_messages(
        runtime_messages,
        system_context=prepared_context.system_context,
    )
    session.messages.extend(runtime_messages[runtime_message_count_before_run:])
    final_content = result.final
    if contains_tool_call_markup(final_content):
        final_content = (
            "检测到模型把工具调用格式写入最终回复，后端已拦截，未展示原始工具JSON。"
            "请重试刚才的请求；如果仍出现，请检查模型是否支持原生 tool calling。"
        )
        if session.messages and session.messages[-1].get("role") == "assistant":
            session.messages[-1]["content"] = final_content
    store.save(session)
    debug_trace.emit(
        "http_chat_final",
        steps_used=result.steps_used,
        used_tools=result.used_tools,
        final_chars=len(final_content),
        stored_message_count=len(session.messages),
    )
    return {
        "message": {"role": "assistant", "content": final_content},
        "steps_used": result.steps_used,
        "model_profile": result.model_profile,
        "used_tools": result.used_tools,
        "selected_skill": skill_hint,
        "conversation_id": conversation_id,
        **debug_trace.context_payload(),
        "context_summary": session.summary,
        "context_summary_message_count": session.summary_message_count,
        "context_compacted": prepared_context.compacted,
        "context_estimated_tokens": prepared_context.estimated_tokens,
    }


def run_agent_chat_events(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    messages = sanitize_chat_messages(payload.get("messages"))
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("Chat messages must end with a user message.")
    conversation_id = resolve_conversation_id(payload, messages)
    active_key = (current_auth_user().id, conversation_id)
    with ACTIVE_CHAT_CONVERSATIONS_LOCK:
        if active_key in ACTIVE_CHAT_CONVERSATIONS:
            yield {
                "event": "error",
                "message": "这个对话已有一轮正在处理。请等待当前轮结束；其他对话仍可同时运行。",
                "type": "ConversationBusy",
                "detail": f"conversation_id={conversation_id}",
            }
            return
        ACTIVE_CHAT_CONVERSATIONS.add(active_key)
    try:
        yield from _run_agent_chat_events(payload)
    finally:
        with ACTIVE_CHAT_CONVERSATIONS_LOCK:
            ACTIVE_CHAT_CONVERSATIONS.discard(active_key)


def _run_agent_chat_events(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    registry = load_registry()
    profile = registry.get(str(payload.get("profile") or registry.default_profile))
    client = OpenAICompatibleClient()
    messages = sanitize_chat_messages(payload.get("messages"))
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("Chat messages must end with a user message.")
    context_file_paths = sanitize_context_file_paths(payload.get("context_file_paths"))
    context_file_paths, project_context, project_id = resolve_project_chat_context(
        payload.get("project_id"), context_file_paths
    )
    conversation_id = resolve_conversation_id(payload, messages)
    skill_hint = normalize_skill_hint(payload.get("skill_hint"))
    max_steps = int(payload.get("max_steps") or DEFAULT_MAX_STEPS)
    max_steps = max(1, min(max_steps, 60))
    reasoning_effort = normalize_reasoning_effort(payload.get("reasoning_effort"))
    debug_trace = DebugTrace(
        WORKSPACE_ROOT,
        conversation_id=conversation_id,
        route="/api/agent/chat-stream",
        profile=profile.name,
        model=profile.model,
    )
    debug_trace.emit(
        "http_chat_stream_start",
        incoming_message_count=len(messages),
        latest_user_chars=len(messages[-1]["content"]),
        skill_hint=skill_hint,
        context_file_path_count=len(context_file_paths),
        project_id=project_id,
        max_steps=max_steps,
    )

    rewind_user_ordinal = sanitize_rewind_user_message_ordinal(payload)
    if rewind_user_ordinal is not None:
        get_turn_store().discard_pending_for_conversation(conversation_id)
    resume_turn_id = (
        None
        if rewind_user_ordinal is not None
        else pending_approval_resume_turn_id(conversation_id, messages[-1]["content"])
    )
    if resume_turn_id:
        debug_trace.emit("manual_approval_resume_redirect", turn_id=resume_turn_id)
        redirected_payload = dict(payload)
        redirected_payload["conversation_id"] = conversation_id
        yield from approve_turn_events(resume_turn_id, redirected_payload)
        return

    turn_runtime = TurnRuntime.start(
        get_turn_store(),
        conversation_id=conversation_id,
        trace_id=debug_trace.trace_id,
        profile=profile.name,
        model=profile.model,
        route="/api/agent/chat-stream",
        metadata={
            "skill_hint": skill_hint,
            "context_file_paths": context_file_paths,
            "project_id": project_id,
            "max_steps": max_steps,
            "reasoning_effort": reasoning_effort,
        },
    )
    debug_trace.emit("turn_started", turn_id=turn_runtime.turn_id)

    started_at = turn_runtime.started_at
    yield turn_runtime.initial_event()
    yield turn_runtime.emit(
        {
            "event": "activity",
            "phase": "thinking",
            "title": "理解请求",
            "detail": "已载入后端会话 working memory，模型可直接回答或返回 tool_calls 调用本地工具/技能/MCP。",
            "elapsed_ms": 0,
            **debug_trace.context_payload(),
        }
    )

    store = get_session_store()
    session = store.load(conversation_id)
    bootstrapped = rewind_session_or_rebuild_from_display(
        store,
        session,
        messages,
        rewind_user_ordinal,
        debug_trace,
    )
    if not bootstrapped:
        bootstrapped = store.bootstrap_from_display_messages(session, messages, exclude_last_user=True)
    if not session.summary:
        session.summary = sanitize_conversation_summary(payload.get("conversation_summary"))
        session.summary_message_count = sanitize_summary_message_count(
            payload.get("conversation_summary_message_count")
        )
    session.metadata["project_id"] = project_id
    store.append_user_message(session, messages[-1]["content"])
    prepared_context = prepare_session_memory(client, profile, session)
    store.save(session)
    debug_trace.emit(
        "session_prepared",
        bootstrapped=bootstrapped,
        stored_message_count=len(session.messages),
        prepared_message_count=len(prepared_context.messages),
        summary_chars=len(session.summary),
        summary_message_count=session.summary_message_count,
        compacted=prepared_context.compacted,
        estimated_tokens=prepared_context.estimated_tokens,
    )
    if bootstrapped:
        yield turn_runtime.emit(
            {
                "event": "activity",
                "phase": "thinking",
                "title": "迁移会话上下文",
                "detail": "后端 session 为空，已从前端展示历史导入一次作为 working memory。",
                "elapsed_ms": int((time.monotonic() - started_at) * 1000),
            }
        )
    if prepared_context.compacted:
        yield turn_runtime.emit(
            {
                "event": "activity",
                "phase": "thinking",
                "title": "整理长上下文",
                "detail": (
                    f"估算上下文 {prepared_context.estimated_tokens} tokens，已超过 "
                    f"{CHAT_SUMMARY_TRIGGER_TOKENS} tokens，生成分点摘要并作为后续初始上下文。"
                ),
                "elapsed_ms": int((time.monotonic() - started_at) * 1000),
            }
        )

    tools = build_default_tools(
        WORKSPACE_ROOT,
        client,
        profile,
        data_workspace=account_workspace_root(),
        include_shared_tools=current_auth_user().role == "admin",
        session_store=store,
        conversation_id=conversation_id,
        project_id=project_id,
        enabled_skill_ids=enabled_skill_ids(),
    )
    agent = ReActAgent(
        client=client,
        profile=profile,
        tools=tools,
        max_steps=max_steps,
        debug_trace=debug_trace,
        cancel_check=turn_runtime.cancelled,
        reasoning_effort=reasoning_effort,
        extra_system_context=agent_system_context()
        + project_context
        + render_history_recall_system_context(session.summary_message_count, project_id=project_id)
        + build_chat_session_system_context(
            session.messages,
            skill_hint=skill_hint,
            context_file_paths=context_file_paths,
        ),
    )
    runtime_messages = list(prepared_context.messages)
    runtime_message_count_before_run = len(runtime_messages)
    session_saved_after_run = False
    try:
        for event in agent.iter_message_events(
            runtime_messages,
            system_context=prepared_context.system_context,
        ):
            turn_runtime.raise_if_cancelled()
            event["elapsed_ms"] = int((time.monotonic() - started_at) * 1000)
            event.setdefault("selected_skill", skill_hint)
            if event.get("event") == "final":
                pending_approval = event.pop("pending_approval", None)
                if event.get("waiting_approval") and isinstance(pending_approval, dict):
                    pending_approval["conversation_id"] = conversation_id
                    pending_approval["project_id"] = project_id
                    pending_approval["selected_skill"] = skill_hint
                    pending_approval["context_summary"] = prepared_context.summary
                    pending_approval["context_summary_message_count"] = prepared_context.summary_message_count
                    pending_approval["context_compacted"] = prepared_context.compacted
                    pending_approval["context_estimated_tokens"] = prepared_context.estimated_tokens
                    get_turn_store().set_pending_approval(turn_runtime.turn_id, pending_approval)
                content = str(event.get("content") or "")
                if contains_tool_call_markup(content):
                    event["content"] = (
                        "检测到模型把工具调用格式写入最终回复，后端已拦截，未展示原始工具JSON。"
                        "请重试刚才的请求；如果仍出现，请检查模型是否支持原生 tool calling。"
                    )
                    last_runtime_message = runtime_messages[-1] if runtime_messages else None
                    if isinstance(last_runtime_message, dict) and last_runtime_message.get("role") == "assistant":
                        last_runtime_message["content"] = event["content"]
                session.messages.extend(
                    message
                    for message in runtime_messages[runtime_message_count_before_run:]
                    if isinstance(message, dict)
                )
                session.summary = prepared_context.summary
                session.summary_message_count = prepared_context.summary_message_count
                store.save(session)
                session_saved_after_run = True
                debug_trace.emit(
                    "http_chat_stream_final",
                    turn_id=turn_runtime.turn_id,
                    steps_used=event.get("steps_used"),
                    used_tools=event.get("used_tools"),
                    final_chars=len(str(event.get("content") or "")),
                    stored_message_count=len(session.messages),
                )
                event["conversation_id"] = conversation_id
                event.update(debug_trace.context_payload())
                event["context_summary"] = session.summary
                event["context_summary_message_count"] = session.summary_message_count
                event["context_compacted"] = prepared_context.compacted
                event["context_estimated_tokens"] = prepared_context.estimated_tokens
            yield turn_runtime.emit(event)
    except (AgentCancelled, TurnCancelled):
        debug_trace.emit("http_chat_stream_cancelled", turn_id=turn_runtime.turn_id)
        yield turn_runtime.cancel_event()
        return
    if not session_saved_after_run and len(runtime_messages) > runtime_message_count_before_run:
        session.messages.extend(runtime_messages[runtime_message_count_before_run:])
        session.summary = prepared_context.summary
        session.summary_message_count = prepared_context.summary_message_count
        store.save(session)
        debug_trace.emit(
            "http_chat_stream_partial_saved",
            turn_id=turn_runtime.turn_id,
            stored_message_count=len(session.messages),
        )


def approve_turn_events(turn_id: str, payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    turn_store = get_turn_store()
    pending_approval = turn_store.pending_approval(turn_id)
    if not pending_approval:
        yield {
            "event": "error",
            "message": "没有找到这个 turn 的待审批工具批次，无法继续执行。",
            "type": "ValueError",
        }
        return

    # Keep the context attached to the approval being resumed.  A later final
    # event normally has no ``pending_approval`` field, so its absence must not
    # overwrite the state needed to finish the response.
    resumed_approval = dict(pending_approval)

    turn = turn_store.load(turn_id)
    conversation_id = sanitize_conversation_id(
        pending_approval.get("conversation_id") or turn.conversation_id or payload.get("conversation_id")
    )
    if not conversation_id:
        yield {
            "event": "error",
            "message": "待审批批次缺少 conversation_id，无法恢复会话。",
            "type": "ValueError",
        }
        return

    registry = load_registry()
    profile = registry.get(str(pending_approval.get("profile_name") or turn.profile or registry.default_profile))
    client = OpenAICompatibleClient()
    debug_trace = DebugTrace(
        WORKSPACE_ROOT,
        conversation_id=conversation_id,
        route="/api/agent/turns/:id/approve",
        profile=profile.name,
        model=profile.model,
    )
    debug_trace.emit("approval_resume_start", turn_id=turn_id)

    turn_runtime = TurnRuntime.resume(turn_store, turn_id)
    started_at = turn_runtime.started_at
    yield turn_runtime.initial_event()

    runtime_messages = pending_approval.get("runtime_messages_before_batch")
    if not isinstance(runtime_messages, list):
        yield turn_runtime.emit(
            {
                "event": "error",
                "message": "待审批批次缺少运行时消息，无法继续执行。",
                "type": "ValueError",
            }
        )
        return
    runtime_messages = repair_runtime_message_sequence(
        [item for item in runtime_messages if isinstance(item, dict)]
    )
    runtime_message_count_before_run = len(runtime_messages)
    system_context = str(pending_approval.get("system_context") or "")
    extra_system_context = str(pending_approval.get("extra_system_context") or agent_system_context())
    skill_hint = pending_approval.get("selected_skill")
    skill_hint = str(skill_hint) if skill_hint else None

    session_store = get_session_store()
    session = session_store.load(conversation_id)
    tools = build_default_tools(
        WORKSPACE_ROOT,
        client,
        profile,
        data_workspace=account_workspace_root(),
        include_shared_tools=current_auth_user().role == "admin",
        session_store=session_store,
        conversation_id=conversation_id,
        project_id=str(pending_approval.get("project_id") or ""),
        enabled_skill_ids=enabled_skill_ids(),
    )
    agent = ReActAgent(
        client=client,
        profile=profile,
        tools=tools,
        max_steps=max(1, min(int(pending_approval.get("max_steps") or DEFAULT_MAX_STEPS), 60)),
        debug_trace=debug_trace,
        cancel_check=turn_runtime.cancelled,
        extra_system_context=extra_system_context,
        reasoning_effort=normalize_reasoning_effort(pending_approval.get("reasoning_effort")),
    )

    session_saved_after_run = False
    try:
        for event in agent.iter_approved_tool_batch_events(
            runtime_messages,
            pending_approval,
            system_context=system_context,
        ):
            turn_runtime.raise_if_cancelled()
            event["elapsed_ms"] = int((time.monotonic() - started_at) * 1000)
            event.setdefault("selected_skill", skill_hint)
            if event.get("event") == "final":
                next_pending_approval = event.pop("pending_approval", None)
                if event.get("waiting_approval") and isinstance(next_pending_approval, dict):
                    next_pending_approval["conversation_id"] = conversation_id
                    next_pending_approval["project_id"] = str(
                        next_pending_approval.get("project_id")
                        or resumed_approval.get("project_id")
                        or ""
                    )
                    next_pending_approval["selected_skill"] = skill_hint
                    next_pending_approval["context_summary"] = str(
                        next_pending_approval.get("context_summary")
                        or resumed_approval.get("context_summary")
                        or session.summary
                        or ""
                    )
                    next_pending_approval["context_summary_message_count"] = int(
                        next_pending_approval.get("context_summary_message_count")
                        or resumed_approval.get("context_summary_message_count")
                        or session.summary_message_count
                        or 0
                    )
                    next_pending_approval["context_compacted"] = bool(
                        next_pending_approval.get(
                            "context_compacted",
                            resumed_approval.get("context_compacted", False),
                        )
                    )
                    next_pending_approval["context_estimated_tokens"] = int(
                        next_pending_approval.get("context_estimated_tokens")
                        or resumed_approval.get("context_estimated_tokens")
                        or 0
                    )
                    turn_store.set_pending_approval(turn_runtime.turn_id, next_pending_approval)
                context_approval = (
                    next_pending_approval
                    if isinstance(next_pending_approval, dict)
                    else resumed_approval
                )
                content = str(event.get("content") or "")
                if contains_tool_call_markup(content):
                    event["content"] = (
                        "检测到模型把工具调用格式写入最终回复，后端已拦截，未展示原始工具JSON。"
                        "请重试刚才的请求；如果仍出现，请检查模型是否支持原生 tool calling。"
                    )
                    last_runtime_message = runtime_messages[-1] if runtime_messages else None
                    if isinstance(last_runtime_message, dict) and last_runtime_message.get("role") == "assistant":
                        last_runtime_message["content"] = event["content"]
                session.messages.extend(
                    message
                    for message in runtime_messages[runtime_message_count_before_run:]
                    if isinstance(message, dict)
                )
                session.summary = str(context_approval.get("context_summary") or session.summary or "")
                session.summary_message_count = int(
                    context_approval.get("context_summary_message_count")
                    or session.summary_message_count
                    or 0
                )
                session_store.save(session)
                if not event.get("waiting_approval"):
                    turn_store.clear_pending_approval(turn_runtime.turn_id)
                session_saved_after_run = True
                debug_trace.emit(
                    "approval_resume_final",
                    turn_id=turn_runtime.turn_id,
                    steps_used=event.get("steps_used"),
                    waiting_approval=bool(event.get("waiting_approval")),
                    saved_pending_approval=bool(
                        event.get("waiting_approval") and isinstance(next_pending_approval, dict)
                    ),
                    final_chars=len(str(event.get("content") or "")),
                    stored_message_count=len(session.messages),
                )
                event["conversation_id"] = conversation_id
                event.update(debug_trace.context_payload())
                event["context_summary"] = session.summary
                event["context_summary_message_count"] = session.summary_message_count
                event["context_compacted"] = bool(context_approval.get("context_compacted"))
                event["context_estimated_tokens"] = int(
                    context_approval.get("context_estimated_tokens") or 0
                )
            yield turn_runtime.emit(event)
    except (AgentCancelled, TurnCancelled):
        debug_trace.emit("approval_resume_cancelled", turn_id=turn_runtime.turn_id)
        yield turn_runtime.cancel_event()
        return
    if not session_saved_after_run and len(runtime_messages) > runtime_message_count_before_run:
        session.messages.extend(runtime_messages[runtime_message_count_before_run:])
        session_store.save(session)
        debug_trace.emit(
            "approval_resume_partial_saved",
            turn_id=turn_runtime.turn_id,
            stored_message_count=len(session.messages),
        )



def generate_chat_title_payload(payload: dict[str, Any]) -> dict[str, str]:
    registry = load_registry()
    requested_profile = str(payload.get("profile") or registry.default_profile)
    try:
        profile = registry.get(requested_profile)
    except KeyError:
        profile = registry.get(registry.default_profile)
    client = OpenAICompatibleClient()
    messages = sanitize_chat_messages(payload.get("messages"))
    if not messages:
        return {"title": "新对话"}

    first_user_index = next(
        (index for index, message in enumerate(messages) if message["role"] == "user"),
        -1,
    )
    first_user = messages[first_user_index]["content"] if first_user_index >= 0 else ""
    first_assistant = next(
        (
            message["content"]
            for message in messages[first_user_index + 1 :]
            if message["role"] == "assistant"
        ),
        "",
    )
    title = request_conversation_title(client, profile, first_user, first_assistant)
    if not is_valid_generated_title(title, first_user):
        title = PENDING_CONVERSATION_TITLE
    return {"title": title or PENDING_CONVERSATION_TITLE, "model_profile": profile.name}


def request_conversation_title(
    client: OpenAICompatibleClient,
    profile: ModelProfile,
    first_user: str,
    first_assistant: str,
) -> str:
    response = client.chat(
        [
            {
                "role": "system",
                "content": (
                    "你是历史对话标题生成器。"
                    "根据首轮用户消息和首轮助手回复，生成一个4到16字的中文短标题。"
                    "只输出标题本身，不要解释，不要引号，不要编号，不要复述用户整句话。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "首轮用户消息：\n"
                    f"{first_user[:3000]}\n\n"
                    "首轮助手回复：\n"
                    f"{first_assistant[:3000]}\n\n"
                    "请输出一个短标题。"
                ),
            },
        ],
        profile=profile,
        max_tokens=64,
    )
    return normalize_conversation_title(response.content)


def is_valid_generated_title(title: str, first_user: str) -> bool:
    text = str(title or "").strip()
    if not text or text == PENDING_CONVERSATION_TITLE:
        return False
    if is_user_echo_title(text, first_user):
        return False
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 4:
        return False
    if re.search(r"[？?]$", compact):
        return False
    banned_fragments = ("新聊天", "对话", "问题", "用户", "助手", "标题", "待命名")
    return not any(fragment in compact for fragment in banned_fragments)


def sanitize_chat_messages(raw_messages: Any) -> list[dict[str, str]]:
    if not isinstance(raw_messages, list):
        raise ValueError("messages must be a list.")
    cleaned: list[dict[str, str]] = []
    for item in raw_messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        if role == "assistant" and contains_tool_call_markup(content):
            content = strip_tool_call_markup(content).strip()
            if not content:
                continue
        cleaned.append({"role": role, "content": content})
    return cleaned


def pending_approval_resume_turn_id(conversation_id: str, latest_user_content: str) -> str:
    if not looks_like_manual_approval_resume_message(latest_user_content):
        return ""
    pending = get_turn_store().pending_approval_for_conversation(conversation_id)
    return pending[0] if pending else ""


def looks_like_manual_approval_resume_message(content: str) -> bool:
    text = str(content or "").strip()
    if not text:
        return False
    compact = re.sub(r"\s+", "", text).lower()
    negative_fragments = (
        "不要执行",
        "不要继续",
        "不用执行",
        "先别执行",
        "别执行",
        "别继续",
        "不执行",
        "不继续",
        "取消执行",
        "取消",
        "拒绝",
        "不同意",
        "不批准",
        "deny",
        "reject",
        "cancel",
        "stop",
    )
    if any(fragment in compact for fragment in negative_fragments):
        return False

    approval_fragments = (
        "approved_by_user=true",
        "确认执行",
        "确认运行",
        "批准执行",
        "同意执行",
        "允许执行",
        "继续执行",
        "恢复执行",
        "按原命令",
        "上一条待审批",
        "待审批终端命令",
        "approve",
        "approved",
        "proceed",
        "runit",
    )
    if any(fragment in compact for fragment in approval_fragments):
        return True

    normalized = compact.strip("。.!！?？,，;；:：")
    return normalized in {
        "确认",
        "同意",
        "批准",
        "可以执行",
        "可以继续",
        "继续",
        "继续吧",
        "好",
        "好的",
        "ok",
        "okay",
        "yes",
        "y",
    }


def sanitize_conversation_summary(raw_value: Any) -> str:
    if not isinstance(raw_value, str):
        return ""
    value = raw_value.strip()
    if contains_tool_call_markup(value):
        value = strip_tool_call_markup(value).strip()
    return value


def sanitize_rewind_user_message_ordinal(payload: dict[str, Any]) -> int | None:
    raw_value = payload.get("rewind_user_message_ordinal")
    if raw_value is None or raw_value == "":
        return None
    try:
        ordinal = int(raw_value)
    except (TypeError, ValueError) as error:
        raise ValueError("rewind_user_message_ordinal must be an integer") from error
    if ordinal < 0:
        raise ValueError("rewind_user_message_ordinal must be non-negative")
    return ordinal


def sanitize_summary_message_count(raw_value: Any) -> int:
    try:
        return max(0, int(raw_value or 0))
    except (TypeError, ValueError):
        return 0


def sanitize_context_file_paths(raw_paths: Any) -> list[str]:
    if not isinstance(raw_paths, list):
        return []
    paths: list[str] = []
    seen: set[str] = set()
    for item in raw_paths[:120]:
        if not isinstance(item, str):
            continue
        for candidate in split_workspace_reference_candidates(item):
            normalized = normalize_workspace_reference_path(candidate)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            paths.append(normalized)
            if len(paths) >= 60:
                return paths
    return paths


def resolve_project_chat_context(
    raw_project_id: Any,
    context_file_paths: list[str],
) -> tuple[list[str], str, str]:
    if not raw_project_id:
        return context_file_paths, "", ""
    project_id = sanitize_project_id(raw_project_id)
    project = read_project(project_id)
    files = project_files(project_id)
    merged_paths: list[str] = []
    seen: set[str] = set()
    for path in [*(str(item.get("path") or "") for item in files), *context_file_paths]:
        normalized = normalize_workspace_reference_path(path)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        merged_paths.append(normalized)

    root = account_relative_path(project_dir(project_id))
    file_lines = [f"- {item['path']}" for item in files[:120]]
    file_block = "\n".join(file_lines) if file_lines else "- 当前项目还没有资料文件"
    instructions = str(project.get("instructions") or "").strip()
    instructions_block = instructions or "未设置额外项目指令。"
    context = (
        "\n\n当前项目上下文（仅限此项目，优先级高于账户工作背景）：\n"
        f"项目名称：{project.get('name') or '未命名项目'}\n"
        f"项目目录：{root}\n"
        "项目记忆范围：仅项目。不要主动引用其他项目的文件或对话。\n"
        "项目指令：\n"
        f"{instructions_block}\n\n"
        "项目资料清单：\n"
        f"{file_block}\n"
        "当用户的任务需要事实、数据、模板或既有材料时，先使用本地文件工具读取相关项目资料；"
        "不需要用户重复上传。资料很多时先列出或搜索项目目录，再读取最相关文件。"
    )
    return merged_paths, context, project_id


def serialize_chat_transcript(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(
        f"{'用户' if item['role'] == 'user' else '助手'}：\n{item['content']}"
        for item in messages
    )
def normalize_conversation_title(raw_value: str) -> str:
    text = str(raw_value or "").strip()
    text = text.strip("`\"'“”‘’")
    text = re.sub(r"^(对话)?标题\s*[:：]\s*", "", text).strip()
    lines = text.splitlines()
    if not lines:
        return ""
    text = lines[0].strip(" -#\t")
    if len(text) > 28:
        text = text[:28].rstrip() + "…"
    return text


def fallback_conversation_title(content: str) -> str:
    text = re.sub(r"\s+", " ", str(content or "")).strip()
    if not text:
        return PENDING_CONVERSATION_TITLE
    return text[:24].rstrip() + ("…" if len(text) > 24 else "")


def fallback_conversation_title_from_messages(messages: list[dict[str, str]]) -> str:
    first_user = next((message["content"] for message in messages if message["role"] == "user"), "")
    transcript = "\n".join(message["content"] for message in messages)
    meeting_name = infer_meeting_name_for_title(messages)
    if meeting_name and looks_like_meeting_minutes_request(transcript):
        return normalize_conversation_title(f"{meeting_name}会议纪要")

    attachment_name = infer_first_attachment_name(transcript)
    if attachment_name:
        stem = Path(attachment_name).stem.strip()
        if stem:
            if is_audio_filename(attachment_name):
                return normalize_conversation_title(f"{stem}录音处理")
            return normalize_conversation_title(f"{stem}文件处理")

    title = fallback_conversation_title(remove_attachment_block(first_user))
    return title if title != PENDING_CONVERSATION_TITLE else "工作智能体任务"


def infer_meeting_name_for_title(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        content = message["content"]
        match = re.search(r"会议名称\s*[:：]\s*([^\n，,。；;]{2,24})", content)
        if match:
            return cleanup_title_phrase(match.group(1))
        if message["role"] == "user":
            cleaned = cleanup_title_phrase(remove_attachment_block(content))
            if 2 <= len(cleaned) <= 16 and re.search(r"[\u4e00-\u9fff]", cleaned):
                return cleaned
    return ""


def infer_first_attachment_name(text: str) -> str:
    match = re.search(r"- \[[^\]]+\]\s*([^:\n]+):\s*meet_files/attachments/[^\s\n]+", text)
    return match.group(1).strip() if match else ""


def remove_attachment_block(text: str) -> str:
    return re.sub(r"\n*参考附件：[\s\S]*$", "", str(text or "")).strip()


def cleanup_title_phrase(text: str) -> str:
    cleaned = re.sub(r"\s+", "", str(text or ""))
    cleaned = re.sub(r"^[：:，,。；;\s]+|[：:，,。；;\s]+$", "", cleaned)
    return cleaned[:18]


def is_audio_filename(name: str) -> bool:
    return Path(name).suffix.lower() in {
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


def is_user_echo_title(title: str, first_user: str) -> bool:
    normalized_title = re.sub(r"\s+", "", str(title or "")).strip("。！？!?")
    normalized_user = re.sub(r"\s+", "", str(first_user or "")).strip("。！？!?")
    if not normalized_title or not normalized_user:
        return False
    return normalized_title == normalized_user


def normalize_skill_hint(raw_value: Any) -> str | None:
    value = str(raw_value or "").strip()
    aliases = {
        "meeting": "meeting-minutes",
        "meeting_minutes": "meeting-minutes",
        "meeting-minutes": "meeting-minutes",
        "会议纪要": "meeting-minutes",
        "@会议纪要": "meeting-minutes",
        "official_document": "official-document",
        "公文": "official-document",
        "@公文": "official-document",
    }
    for skill in skill_catalog_payload()["skills"]:
        skill_id = str(skill.get("id") or "")
        if not skill_id:
            continue
        aliases[skill_id] = skill_id
        aliases[str(skill.get("label") or "")] = skill_id
        aliases[str(skill.get("mention") or "")] = skill_id
    normalized = aliases.get(value, value)
    valid_ids = {str(skill.get("id")) for skill in skill_catalog_payload()["skills"]}
    return normalized if normalized in valid_ids else None


def resolve_conversation_id(payload: dict[str, Any], messages: list[dict[str, str]]) -> str:
    raw_id = payload.get("conversation_id") or payload.get("conversationId")
    conversation_id = sanitize_conversation_id(raw_id)
    if conversation_id:
        return conversation_id
    seed = "\n".join(message["content"] for message in messages[-4:])
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]
    return f"session-{digest}"


def build_chat_session_system_context(
    session_messages: list[dict[str, Any]],
    *,
    skill_hint: str | None = None,
    context_file_paths: list[str] | None = None,
) -> str:
    transcript = serialize_runtime_messages_for_context(session_messages)
    known_file_refs = extract_workspace_file_references(transcript, context_file_paths=context_file_paths)
    known_paths_block = render_known_file_references(known_file_refs)
    skills_block = render_chat_skill_catalog()
    skill_instruction = ""
    meeting_intent = skill_hint == "meeting-minutes" or looks_like_meeting_minutes_request(transcript)
    official_document_intent = (
        skill_hint == "official-document" or looks_like_official_document_request(transcript)
    )
    if skill_hint == "official-document":
        skill_instruction = (
            "\n\n本轮已选技能：official-document。先调用 "
            "sys_skill(op='open', skill_id='official-document') 判断文种、要素和格式，"
            "形成内容与文档规格后，再调用 sys_skill(op='open', skill_id='docx')，"
            "由完整 Word 技能生成、编辑并验收最终文件。"
        )
    elif meeting_intent:
        skill_instruction = (
            "\n\n本轮匹配技能：meeting-minutes。先调用 "
            "sys_skill(op='open', skill_id='meeting-minutes') 载入完整流程，再按说明执行。"
        )
    elif official_document_intent:
        skill_instruction = (
            "\n\n本轮匹配默认启用技能：official-document。先调用 "
            "sys_skill(op='open', skill_id='official-document') 判断文种、要素和格式，"
            "形成内容与文档规格后，再调用 sys_skill(op='open', skill_id='docx')，"
            "由完整 Word 技能生成、编辑并验收最终文件。"
        )
    elif skill_hint:
        skill_instruction = (
            f"\n\n已选技能：{skill_hint}。"
            f"先调用 sys_skill(op='open', skill_id='{skill_hint}') 载入说明，"
            "技能专用工具通过 sys_skill 的 show/call 分层使用。"
        )
    elif looks_like_office_request(transcript):
        skill_instruction = (
            "\n\n本轮包含办公文件。根据文件类型从技能索引选择 docx/pdf/pptx/xlsx，"
            "先用 sys_skill.open 载入说明，再通过 sys_skill.show/call 使用对应能力。"
        )
    return (
        "\n\n当前会话动态上下文：最近消息以标准 messages 形式提供；较新的消息和工具结果优先。\n"
        f"{skills_block}{known_paths_block}{skill_instruction}\n"
    )


def serialize_runtime_messages_for_context(messages: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for message in messages:
        role = message.get("role")
        if role in {"user", "assistant"}:
            content = str(message.get("content") or "")
            if content:
                blocks.append(content)
        elif role == "tool":
            name = str(message.get("name") or "")
            content = str(message.get("content") or "")
            if name or content:
                blocks.append(f"{name}\n{content[:2000]}")
    return "\n\n".join(blocks)



def format_chat_goal(
    messages: list[dict[str, str]],
    *,
    skill_hint: str | None = None,
    context_file_paths: list[str] | None = None,
    conversation_summary: str = "",
) -> str:
    transcript = serialize_chat_transcript(messages)
    summary_block = (
        "\n\n此前对话的分点摘要（作为本轮对话的初始上下文，原始历史仍保存在本地）：\n"
        f"{conversation_summary}"
        if conversation_summary
        else ""
    )
    known_file_refs = extract_workspace_file_references(transcript, context_file_paths=context_file_paths)
    known_paths_block = render_known_file_references(known_file_refs)
    skills_block = render_chat_skill_catalog()
    skill_instruction = ""
    meeting_intent = skill_hint == "meeting-minutes" or looks_like_meeting_minutes_request(transcript)
    official_document_intent = (
        skill_hint == "official-document" or looks_like_official_document_request(transcript)
    )
    if skill_hint == "official-document":
        skill_instruction = (
            "\n\n本轮已选技能 official-document：先调用 sys_skill.open 读取公文说明，"
            "形成内容与文档规格后，再打开 docx 技能生成、编辑和验收 Word。"
        )
    elif meeting_intent:
        skill_instruction = (
            "\n\n本轮匹配技能 meeting-minutes：先调用 sys_skill.open 读取完整说明，"
            "再通过 sys_skill.show/call 使用技能专用能力。"
        )
    elif official_document_intent:
        skill_instruction = (
            "\n\n本轮匹配默认启用技能 official-document：先调用 sys_skill.open 读取公文说明，"
            "形成内容与文档规格后，再打开 docx 技能生成、编辑和验收 Word。"
        )
    elif skill_hint:
        skill_instruction = (
            f"\n\n已选技能 {skill_hint}：先调用 sys_skill.open 读取说明，"
            "再通过 sys_skill.show/call 使用技能专用能力。"
        )
    elif looks_like_office_request(transcript):
        skill_instruction = (
            "\n\n本轮包含办公文件：从技能索引选择 docx/pdf/pptx/xlsx，"
            "先调用 sys_skill.open，再通过 sys_skill.show/call 执行。"
        )
    return (
        "你正在作为本地工作智能体与用户连续对话。"
        "core 文件与终端工具常驻；技能通过 sys_skill、外部 MCP 通过 mcporter 分层调用。"
        "请根据以下最近对话继续完成用户最新请求；不要重复寒暄，不要编造工具结果。"
        "最终答复请写成Markdown正文，方便前端渲染；不要把整段最终答复放进代码块。"
        "ReAct 过程（例如读取技能、准备搜索、执行脚本、检查环境、工具参数和中间观察）只应体现在活动/工具调用中，"
        "最终答复只写用户要的结论、摘要、文件路径或下一步建议；不要在最终答复中复述“先读取技能说明”“正在调用工具”“搜索词是……”等过程。"
        "领域任务先读取对应技能；不要猜测隐藏的技能工具名。"
        "浏览器快照中遇到无文字的图标按钮时，不要把空文本 button 的点击当作已成功；"
        "聊天发送优先对已确认的 textbox 调用 browser_type，并传 submit: true。"
        "如果只是普通问答，可以不调用任何工具，直接用 content 给出最终答复。"
        "修改现有文本文件时，小改优先使用 edit_text_file；跨多处或多文件改动优先使用 apply_unified_patch；"
        "只有创建完整新文件或确实需要重写成品时才用 write_text_file。\n\n"
        f"{skills_block}{summary_block}\n\n{transcript}{known_paths_block}{skill_instruction}\n\n"
        "请回答用户最后一条消息。"
    )


def render_chat_skill_catalog() -> str:
    skills = skill_catalog_payload()["skills"]
    if not skills:
        return ""
    compact_skills = [
        {
            "id": str(skill.get("id") or ""),
            "label": str(skill.get("label") or ""),
            "mention": str(skill.get("mention") or ""),
            "description": str(skill.get("description") or ""),
            "when_to_use": str(skill.get("when_to_use") or ""),
            "default_enabled": bool(skill.get("default_enabled", False)),
        }
        for skill in skills
    ]
    return (
        "当前已安装技能索引（只含路由摘要；匹配后用 sys_skill.open 读取完整说明）：\n"
        f"{json.dumps(compact_skills, ensure_ascii=False, indent=2)}\n\n"
    )


def extract_workspace_paths(text: str) -> list[str]:
    pattern = re.compile(
        r"(?:/Users/alian/workspace/work_agent/)?"
        r"(?:meet_files|meeting_audio_minutes|work_agent_skills|web_frontend|work_agent_core|config|schemas|tmp|产出材料|分析材料|学习笔记)"
        r"/[^\s`'\"<>|\\\x00-\x1f]+",
        re.IGNORECASE,
    )
    seen: set[str] = set()
    paths: list[str] = []
    workspace_prefix = "/Users/alian/workspace/work_agent/"
    for match in pattern.finditer(text):
        path = match.group(0).strip().rstrip("，。；;、,.!?！？:：)]}")
        if path.startswith(workspace_prefix):
            path = path[len(workspace_prefix) :]
        if path in seen:
            continue
        seen.add(path)
        paths.append(path)
    return paths


def extract_workspace_file_references(
    text: str,
    context_file_paths: list[str] | None = None,
) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    seen_paths: set[str] = set()

    for path in context_file_paths or []:
        for candidate in split_workspace_reference_candidates(path):
            normalized = normalize_workspace_reference_path(candidate)
            if not normalized or normalized in seen_paths:
                continue
            seen_paths.add(normalized)
            refs.append({"path": normalized, "source": "context"})

    for path in extract_workspace_paths(text):
        normalized = normalize_workspace_reference_path(path)
        if not normalized or normalized in seen_paths:
            continue
        seen_paths.add(normalized)
        refs.append({"path": normalized, "source": "path"})

    visible_files = visible_file_reference_index()
    for name in extract_file_names(text):
        for path in visible_files.get(name, []):
            if path in seen_paths:
                continue
            seen_paths.add(path)
            refs.append({"path": path, "source": "filename"})

    return refs


def render_known_file_references(refs: list[dict[str, str]]) -> str:
    if not refs:
        return ""

    meeting_outputs = [
        item for item in refs if any(marker in Path(item["path"]).name for marker in ("内部留档版", "工作提交版"))
    ]
    other_refs = [item for item in refs if item not in meeting_outputs]
    ordered = [*meeting_outputs, *other_refs]

    lines = []
    for item in ordered[:24]:
        path = item["path"]
        name = Path(path).name
        if item["source"] == "context":
            source_label = "对话活动上下文"
        elif item["source"] == "path":
            source_label = "文本路径"
        else:
            source_label = "文本文件名解析"
        role = ""
        if "内部留档版" in name:
            role = "，角色=内部留档版"
        elif "工作提交版" in name:
            role = "，角色=工作提交版"
        lines.append(f"- {path}（来源={source_label}{role}）")

    return (
        "\n\n当前对话中已识别的本地文件引用（这是上下文编译结果，不需要再调用 list_workspace_files 确认）：\n"
        + "\n".join(lines)
    )


def normalize_workspace_reference_path(path: str) -> str:
    workspace_prefix = "/Users/alian/workspace/work_agent/"
    candidate = path.strip().strip("`'\"").rstrip("，。；;、,.!?！？:：)]}")
    if not candidate or any(ord(char) < 32 for char in candidate):
        return ""
    try:
        candidate = decode_uri_path(candidate)
    except ValueError:
        return ""
    if candidate.startswith(workspace_prefix):
        candidate = candidate[len(workspace_prefix) :]
    candidate = candidate.replace("\\", "/")
    if candidate.startswith("./"):
        candidate = candidate[2:]
    encoded = candidate.encode("utf-8", errors="surrogatepass")
    if len(encoded) > 4096 or any(
        len(part.encode("utf-8", errors="surrogatepass")) > 255
        for part in Path(candidate).parts
    ):
        return ""
    try:
        resolved = (WORKSPACE_ROOT / candidate).resolve()
    except (OSError, RuntimeError, ValueError):
        return ""
    if WORKSPACE_ROOT not in (resolved, *resolved.parents):
        return ""
    try:
        exists = resolved.exists()
    except OSError:
        return ""
    if not exists:
        return candidate
    return str(resolved.relative_to(WORKSPACE_ROOT))


def split_workspace_reference_candidates(value: str) -> list[str]:
    """Split paths joined by NULs or their serialized ``\\u0000`` form."""
    return [
        item.strip()
        for item in re.split(r"(?:\x00|\\u0000|[\r\n]+)", str(value or ""))
        if item.strip()
    ]


def decode_uri_path(path: str) -> str:
    from urllib.parse import unquote

    decoded = unquote(path)
    if "\x00" in decoded:
        raise ValueError("NUL byte in path")
    return decoded


def extract_file_names(text: str) -> list[str]:
    extensions = (
        "md",
        "txt",
        "json",
        "ya?ml",
        "csv",
        "log",
        "srt",
        "vtt",
        "py",
        "tsx?",
        "jsx?",
        "css",
        "html",
        "pdf",
        "docx?",
        "pptx?",
        "xlsx?",
        "png",
        "jpe?g",
        "webp",
        "gif",
        "heic",
        "tiff?",
        "m4a",
        "mp3",
        "wav",
        "aac",
        "flac",
        "ogg",
        "opus",
    )
    pattern = re.compile(
        rf"(?<![/\w.-])([^\s`'\"<>|/\\]{{2,120}}?\.(?:{'|'.join(extensions)}))(?![\w.-])",
        re.IGNORECASE,
    )
    seen: set[str] = set()
    names: list[str] = []
    for match in pattern.finditer(text):
        name = match.group(1).strip().rstrip("，。；;、,.!?！？:：)]}")
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def visible_file_reference_index() -> dict[str, list[str]]:
    files: list[Path] = []
    for root_name in ("meet_files", "产出材料", "分析材料", "学习笔记"):
        root = WORKSPACE_ROOT / root_name
        if not root.exists():
            continue
        for item in root.rglob("*"):
            if item.is_file() and is_file_library_visible(item):
                files.append(item)

    files.sort(key=lambda item: int(item.stat().st_mtime), reverse=True)
    index: dict[str, list[str]] = {}
    for item in files:
        name = item.name
        path = str(item.relative_to(WORKSPACE_ROOT))
        index.setdefault(name, []).append(path)
    return index


def looks_like_meeting_minutes_request(text: str) -> bool:
    normalized = text.lower()
    meeting_words = ("会议", "纪要", "会议计较", "会议记录", "旁听")
    audio_words = ("录音", "音频", ".m4a", ".mp3", ".wav", ".webm", "asr", "转写", "降噪")
    output_words = ("整理", "生成", "怎么做", "流程", "产出", "留档", "提交")
    return (
        any(word in normalized for word in meeting_words)
        and (any(word in normalized for word in audio_words) or any(word in normalized for word in output_words))
    )


def looks_like_office_request(text: str) -> bool:
    normalized = text.lower()
    cues = (
        ".docx",
        ".xlsx",
        ".xlsm",
        ".csv",
        ".tsv",
        ".pptx",
        ".pdf",
        "word",
        "excel",
        "ppt",
        "pdf",
        "文档",
        "表格",
        "幻灯片",
        "演示",
        "研报",
        "白皮书",
    )
    return any(cue in normalized for cue in cues)


def looks_like_official_document_request(text: str) -> bool:
    normalized = text.lower()
    explicit_cues = (
        "公文格式",
        "正式发文",
        "红头文件",
        "发文字号",
        "主送机关",
        "抄送机关",
        "签发人",
        "版记",
        "gb/t 9704",
        "gbt 9704",
    )
    if any(cue in normalized for cue in explicit_cues):
        return True
    document_types = (
        "请示",
        "批复",
        "通报",
        "通告",
        "公告",
        "公报",
        "议案",
        "命令",
        "决定",
        "决议",
        "意见",
        "通知",
        "报告",
        "函",
        "纪要",
    )
    action_cues = (
        "写",
        "起草",
        "撰写",
        "制作",
        "生成",
        "形成",
        "修改",
        "润色",
        "排版",
        "转word",
        "转 word",
    )
    return any(kind in normalized for kind in document_types) and any(
        action in normalized for action in action_cues
    )


def run_meeting_minutes_payload(payload: dict[str, Any]) -> dict[str, Any]:
    registry = load_registry()
    profile = registry.get(str(payload.get("profile") or registry.default_profile))
    client = OpenAICompatibleClient()
    skill = MeetingMinutesSkill(workspace_root=WORKSPACE_ROOT, client=client, profile=profile)
    result = skill.run(
        {
            "input_path": str(payload.get("input_path") or payload.get("transcript_path") or payload.get("audio_path") or ""),
            "transcript_path": str(payload.get("transcript_path") or ""),
            "audio_path": str(payload.get("audio_path") or ""),
            "output_dir": str(payload.get("output_dir") or "meet_files"),
            "meeting_name": required_string(payload, "meeting_name"),
            "confirmed_info": str(payload.get("confirmed_info") or ""),
            "supplemental_paths": payload.get("supplemental_paths") or [],
            "spec_path": str(payload.get("spec_path") or "meeting_audio_minutes/ASR文本整理流程.md"),
        }
    )
    return {"result": json.loads(result)}


def sanitize_filename(name: str) -> str:
    cleaned = "".join(
        char if char not in '/\\:*?"<>|' and 32 <= ord(char) != 127 else "_"
        for char in name
    ).strip()
    return cleaned or "attachment"


def normalize_mime_type(value: Any) -> str:
    mime_type = str(value or "").split(";", 1)[0].strip().lower()
    if re.fullmatch(r"[a-z0-9.+-]+/[a-z0-9.+-]+", mime_type):
        return mime_type
    return "application/octet-stream"


def temporary_sync_root() -> Path:
    root = user_data_dir() / "temporary_sync"
    (root / "files").mkdir(parents=True, exist_ok=True)
    return root


def parse_temporary_sync_file_route(path: str) -> str | None:
    prefix = "/api/temporary-sync/files/"
    if not path.startswith(prefix):
        return None
    file_id = path[len(prefix) :]
    return file_id if TEMP_SYNC_FILE_ID_PATTERN.fullmatch(file_id) else None


def temporary_sync_payload(*, now: int | None = None) -> dict[str, Any]:
    timestamp = int(time.time()) if now is None else int(now)
    with TEMP_SYNC_LOCK:
        root = temporary_sync_root()
        cleanup_expired_temporary_sync_files(root, now=timestamp)
        text = {"content": "", "updated_at": None}
        text_path = root / "text.json"
        if text_path.is_file():
            try:
                stored = json.loads(text_path.read_text(encoding="utf-8"))
                if isinstance(stored, dict):
                    text = {
                        "content": str(stored.get("content") or ""),
                        "updated_at": optional_int(stored.get("updated_at")),
                    }
            except (OSError, ValueError, TypeError):
                pass

        files: list[dict[str, Any]] = []
        for metadata_path in (root / "files").glob("*.json"):
            metadata = read_temporary_sync_file_metadata(metadata_path)
            if metadata is None:
                continue
            data_path = metadata_path.with_suffix(".bin")
            if not data_path.is_file():
                metadata_path.unlink(missing_ok=True)
                continue
            files.append(temporary_sync_file_item(metadata))
        files.sort(key=lambda item: int(item["uploaded_at"]), reverse=True)
        return {
            "text": text,
            "files": files,
            "file_ttl_seconds": TEMP_SYNC_FILE_TTL_SECONDS,
            "server_time": timestamp,
        }


def save_temporary_sync_text_payload(payload: dict[str, Any]) -> dict[str, Any]:
    content = str(payload.get("content") or "")
    if len(content) > TEMP_SYNC_MAX_TEXT_CHARS:
        raise ValueError(f"文字内容不能超过 {TEMP_SYNC_MAX_TEXT_CHARS} 个字符")
    updated_at = int(time.time())
    with TEMP_SYNC_LOCK:
        path = temporary_sync_root() / "text.json"
        write_json_atomically(path, {"content": content, "updated_at": updated_at})
    return {
        "text": {"content": content, "updated_at": updated_at},
        "message": "文字已同步",
    }


def add_temporary_sync_file_payload(payload: dict[str, Any]) -> dict[str, Any]:
    name = sanitize_filename(required_string(payload, "name"))
    encoded = required_string(payload, "content_base64")
    try:
        data = base64.b64decode(encoded, validate=True)
    except Exception as error:
        raise ValueError("文件内容无效，请重新选择文件") from error
    if not data:
        raise ValueError("不能上传空文件")
    if len(data) > TEMP_SYNC_MAX_FILE_BYTES:
        raise ValueError("单个文件不能超过 100 MB")

    uploaded_at = int(time.time())
    metadata = {
        "id": uuid.uuid4().hex,
        "name": name,
        "size": len(data),
        "mime_type": normalize_mime_type(payload.get("mime_type")),
        "uploaded_at": uploaded_at,
        "expires_at": uploaded_at + TEMP_SYNC_FILE_TTL_SECONDS,
    }
    with TEMP_SYNC_LOCK:
        directory = temporary_sync_root() / "files"
        cleanup_expired_temporary_sync_files(directory.parent, now=uploaded_at)
        data_path = directory / f"{metadata['id']}.bin"
        metadata_path = directory / f"{metadata['id']}.json"
        data_path.write_bytes(data)
        try:
            write_json_atomically(metadata_path, metadata)
        except Exception:
            data_path.unlink(missing_ok=True)
            raise
    return {
        "file": temporary_sync_file_item(metadata),
        "message": "文件已放入临时同步区，1 小时后自动删除",
    }


def delete_temporary_sync_file_payload(payload: dict[str, Any]) -> dict[str, Any]:
    file_id = required_string(payload, "id")
    if not TEMP_SYNC_FILE_ID_PATTERN.fullmatch(file_id):
        raise ValueError("文件标识无效")
    with TEMP_SYNC_LOCK:
        directory = temporary_sync_root() / "files"
        data_path = directory / f"{file_id}.bin"
        metadata_path = directory / f"{file_id}.json"
        existed = data_path.is_file() or metadata_path.is_file()
        data_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
    return {"ok": True, "deleted": existed}


def resolve_temporary_sync_file(
    file_id: str,
    *,
    now: int | None = None,
) -> tuple[Path, dict[str, Any]]:
    if not TEMP_SYNC_FILE_ID_PATTERN.fullmatch(file_id):
        raise ValueError("文件标识无效")
    timestamp = int(time.time()) if now is None else int(now)
    with TEMP_SYNC_LOCK:
        directory = temporary_sync_root() / "files"
        metadata_path = directory / f"{file_id}.json"
        metadata = read_temporary_sync_file_metadata(metadata_path)
        data_path = directory / f"{file_id}.bin"
        if (
            metadata is None
            or int(metadata.get("expires_at") or 0) <= timestamp
            or not data_path.is_file()
        ):
            data_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            raise ValueError("文件不存在或已超过 1 小时有效期")
        return data_path, metadata


def cleanup_expired_temporary_sync_files(root: Path, *, now: int) -> None:
    directory = root / "files"
    directory.mkdir(parents=True, exist_ok=True)
    known_ids: set[str] = set()
    for metadata_path in directory.glob("*.json"):
        file_id = metadata_path.stem
        metadata = read_temporary_sync_file_metadata(metadata_path)
        data_path = directory / f"{file_id}.bin"
        if (
            metadata is None
            or int(metadata.get("expires_at") or 0) <= now
            or not data_path.is_file()
        ):
            metadata_path.unlink(missing_ok=True)
            data_path.unlink(missing_ok=True)
            continue
        known_ids.add(file_id)
    for data_path in directory.glob("*.bin"):
        if data_path.stem not in known_ids:
            data_path.unlink(missing_ok=True)


def read_temporary_sync_file_metadata(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    file_id = str(data.get("id") or "")
    if file_id != path.stem or not TEMP_SYNC_FILE_ID_PATTERN.fullmatch(file_id):
        return None
    return data


def temporary_sync_file_item(metadata: dict[str, Any]) -> dict[str, Any]:
    file_id = str(metadata["id"])
    return {
        "id": file_id,
        "name": str(metadata.get("name") or "download"),
        "size": int(metadata.get("size") or 0),
        "mime_type": str(metadata.get("mime_type") or "application/octet-stream"),
        "uploaded_at": int(metadata.get("uploaded_at") or 0),
        "expires_at": int(metadata.get("expires_at") or 0),
        "download_url": f"/api/temporary-sync/files/{file_id}",
    }


def write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(1, 1000):
        candidate = path.with_name(f"{stem}-{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate unique path for {path.name}")


def classify_attachment(extension: str, mime_type: str) -> str:
    audio_extensions = {
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
    }
    image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".heic", ".tif", ".tiff"}
    document_extensions = {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".md", ".txt"}
    if extension in audio_extensions or mime_type.startswith("audio/"):
        return "audio"
    if extension in image_extensions or mime_type.startswith("image/"):
        return "image"
    if extension in document_extensions:
        return "document"
    return "file"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local Work Agent web API/static server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--static-dir", default=str(STATIC_DIR))
    return parser.parse_args()


def main() -> int:
    global WORKSPACE_ROOT, CONFIG_PATH, STATIC_DIR, AUTH_STORE
    args = parse_args()
    WORKSPACE_ROOT = Path(args.workspace).resolve()
    CONFIG_PATH = Path(args.config)
    STATIC_DIR = Path(args.static_dir)
    AUTH_STORE = None
    get_auth_store()
    atexit.register(stop_asr_worker)
    atexit.register(lambda: VAD_WORKER.stop() if VAD_WORKER else None)
    server = ThreadingHTTPServer((args.host, args.port), WorkAgentHandler)
    print(f"Work Agent web server running at http://{args.host}:{args.port}")
    print(f"Workspace: {WORKSPACE_ROOT}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
