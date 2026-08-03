from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode
import base64
import hashlib
import json
import random
import secrets
import threading
import time
import urllib.error
import urllib.request
import uuid

from .message_channel import ChannelMessage, ChannelReply


ILINK_BASE_URL = "https://ilinkai.weixin.qq.com"
ILINK_BOT_TYPE = "3"
ILINK_APP_ID = "bot"
ILINK_CHANNEL_VERSION = "2.4.6"
ILINK_CLIENT_VERSION = (2 << 16) | (4 << 8) | 6
ILINK_BOT_AGENT = "WorkAgent/0.1.0"
MAX_TEXT_CHARS = 4000
LOGIN_TTL_SECONDS = 5 * 60
SYNC_TIMEOUT_SECONDS = 38
MESSAGE_PROCESSING_STALE_SECONDS = 10 * 60
MESSAGE_DEDUPE_RETENTION_SECONDS = 7 * 24 * 60 * 60


class WeixinApiError(RuntimeError):
    pass


@dataclass
class WeixinCredentials:
    account_id: str
    token: str
    base_url: str = ILINK_BASE_URL
    user_id: str = ""
    saved_at: int = 0

    def to_payload(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "token": self.token,
            "base_url": self.base_url,
            "user_id": self.user_id,
            "saved_at": self.saved_at or int(time.time()),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> WeixinCredentials:
        return cls(
            account_id=str(payload.get("account_id") or "").strip(),
            token=str(payload.get("token") or "").strip(),
            base_url=str(payload.get("base_url") or ILINK_BASE_URL).strip() or ILINK_BASE_URL,
            user_id=str(payload.get("user_id") or "").strip(),
            saved_at=int(payload.get("saved_at") or 0),
        )


@dataclass
class WeixinLoginSession:
    session_id: str
    qrcode: str
    qrcode_url: str
    started_at: float
    current_base_url: str = ILINK_BASE_URL
    status: str = "wait"

    @property
    def expired(self) -> bool:
        return time.monotonic() - self.started_at > LOGIN_TTL_SECONDS


def build_base_info() -> dict[str, Any]:
    return {
        "channel_version": ILINK_CHANNEL_VERSION,
        "bot_agent": ILINK_BOT_AGENT,
    }


def random_wechat_uin() -> str:
    value = random.SystemRandom().randrange(0, 2**32)
    return base64.b64encode(str(value).encode("ascii")).decode("ascii")


def chunk_weixin_text(text: str, limit: int = MAX_TEXT_CHARS) -> list[str]:
    value = str(text or "").strip()
    if not value:
        return []
    chunks: list[str] = []
    remaining = value
    while len(remaining) > limit:
        split_at = max(
            remaining.rfind("\n\n", 0, limit + 1),
            remaining.rfind("\n", 0, limit + 1),
            remaining.rfind("。", 0, limit + 1),
        )
        if split_at < max(1, limit // 2):
            split_at = limit
        elif remaining[split_at : split_at + 1] == "。":
            split_at += 1
        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


def extract_weixin_text(message: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in message.get("item_list") or []:
        if not isinstance(item, dict):
            continue
        if int(item.get("type") or 0) == 1:
            text_item = item.get("text_item")
            if isinstance(text_item, dict):
                text = str(text_item.get("text") or "").strip()
                if text:
                    parts.append(text)
        elif int(item.get("type") or 0) == 3:
            voice_item = item.get("voice_item")
            if isinstance(voice_item, dict):
                transcript = str(voice_item.get("text") or "").strip()
                if transcript:
                    parts.append(transcript)
    return "\n".join(parts).strip()


class WeixinIlinkClient:
    def __init__(self, *, opener: urllib.request.OpenerDirector | None = None) -> None:
        self.opener = opener or urllib.request.build_opener()

    def start_login(self, local_tokens: list[str] | None = None) -> WeixinLoginSession:
        payload = self._request_json(
            "POST",
            f"{ILINK_BASE_URL}/ilink/bot/get_bot_qrcode?bot_type={ILINK_BOT_TYPE}",
            {"local_token_list": list(local_tokens or [])[-10:]},
            authenticated=False,
            timeout=15,
        )
        qrcode = str(payload.get("qrcode") or "").strip()
        qrcode_url = str(payload.get("qrcode_img_content") or "").strip()
        if not qrcode or not qrcode_url:
            raise WeixinApiError("微信没有返回可用的二维码。")
        return WeixinLoginSession(
            session_id=secrets.token_urlsafe(18),
            qrcode=qrcode,
            qrcode_url=qrcode_url,
            started_at=time.monotonic(),
        )

    def poll_login(
        self,
        session: WeixinLoginSession,
        *,
        verify_code: str = "",
    ) -> dict[str, Any]:
        if session.expired:
            return {"status": "expired"}
        query = {"qrcode": session.qrcode}
        if verify_code.strip():
            query["verify_code"] = verify_code.strip()
        url = f"{session.current_base_url}/ilink/bot/get_qrcode_status?{urlencode(query)}"
        try:
            payload = self._request_json(
                "GET",
                url,
                None,
                authenticated=False,
                timeout=SYNC_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            return {"status": "wait"}
        status = str(payload.get("status") or "wait")
        session.status = status
        if status == "scaned_but_redirect":
            redirect_host = str(payload.get("redirect_host") or "").strip()
            if redirect_host:
                session.current_base_url = f"https://{redirect_host}"
        return payload

    def get_updates(
        self,
        credentials: WeixinCredentials,
        sync_buf: str,
        *,
        timeout: int = SYNC_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        return self._request_json(
            "POST",
            f"{credentials.base_url}/ilink/bot/getupdates",
            {
                "get_updates_buf": sync_buf,
                "base_info": build_base_info(),
            },
            token=credentials.token,
            timeout=timeout,
        )

    def send_text(
        self,
        credentials: WeixinCredentials,
        *,
        to_user_id: str,
        context_token: str,
        text: str,
    ) -> None:
        for index, chunk in enumerate(chunk_weixin_text(text)):
            payload = self._request_json(
                "POST",
                f"{credentials.base_url}/ilink/bot/sendmessage",
                {
                    "msg": {
                        "from_user_id": "",
                        "to_user_id": to_user_id,
                        "client_id": f"work-agent-{uuid.uuid4()}",
                        "message_type": 2,
                        "message_state": 2,
                        "item_list": [{"type": 1, "text_item": {"text": chunk}}],
                        "context_token": context_token or None,
                    },
                    "base_info": build_base_info(),
                },
                token=credentials.token,
                timeout=15,
            )
            if int(payload.get("ret") or 0) != 0:
                raise WeixinApiError(
                    f"微信发送失败：ret={payload.get('ret')} {payload.get('errmsg') or ''}".strip()
                )
            if index:
                time.sleep(0.3)

    def _request_json(
        self,
        method: str,
        url: str,
        body: dict[str, Any] | None,
        *,
        token: str = "",
        authenticated: bool = True,
        timeout: int,
    ) -> dict[str, Any]:
        headers = {
            "Accept": "application/json",
            "iLink-App-Id": ILINK_APP_ID,
            "iLink-App-ClientVersion": str(ILINK_CLIENT_VERSION),
        }
        data: bytes | None = None
        if method == "POST":
            headers["Content-Type"] = "application/json"
            data = json.dumps(body or {}, ensure_ascii=False).encode("utf-8")
        if authenticated:
            headers.update(
                {
                    "AuthorizationType": "ilink_bot_token",
                    "Authorization": f"Bearer {token}",
                    "X-WECHAT-UIN": random_wechat_uin(),
                }
            )
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with self.opener.open(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as error:
            raw = error.read().decode("utf-8", errors="replace")
            raise WeixinApiError(f"微信接口 HTTP {error.code}：{raw[:500]}") from error
        except (TimeoutError, urllib.error.URLError) as error:
            if isinstance(error, TimeoutError) or isinstance(getattr(error, "reason", None), TimeoutError):
                raise TimeoutError("微信接口等待超时") from error
            raise WeixinApiError(f"无法连接微信接口：{error}") from error
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise WeixinApiError(f"微信接口返回了无效数据：{raw[:300]}") from error
        if not isinstance(payload, dict):
            raise WeixinApiError("微信接口返回格式不正确。")
        return payload


def credentials_path(state_dir: Path) -> Path:
    return state_dir / "channel.json"


def load_credentials(state_dir: Path) -> WeixinCredentials | None:
    path = credentials_path(state_dir)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    credentials = WeixinCredentials.from_payload(payload)
    if not credentials.account_id or not credentials.token:
        return None
    return credentials


def save_credentials(state_dir: Path, credentials: WeixinCredentials) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = credentials_path(state_dir)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(
        json.dumps(credentials.to_payload(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp.replace(path)
    try:
        path.chmod(0o600)
    except OSError:
        pass


def clear_credentials(state_dir: Path) -> None:
    for name in ("channel.json", "sync.json", "context-tokens.json", "dedupe.json"):
        try:
            (state_dir / name).unlink()
        except FileNotFoundError:
            pass


class WeixinChannelWorker:
    channel_id = "weixin"

    def __init__(
        self,
        *,
        state_dir: Path,
        credentials: WeixinCredentials,
        on_message: Callable[[ChannelMessage], ChannelReply],
        client: WeixinIlinkClient | None = None,
    ) -> None:
        self.state_dir = state_dir
        self.credentials = credentials
        self.on_message = on_message
        self.client = client or WeixinIlinkClient()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._status = "stopped"
        self._last_error = ""
        self._last_message_at = 0

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._status = "starting"
            self._thread = threading.Thread(
                target=self._run,
                name=f"weixin-{self.credentials.account_id}",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2)
        with self._lock:
            self._status = "stopped"

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._status,
                "last_error": self._last_error,
                "last_message_at": self._last_message_at,
                "running": bool(self._thread and self._thread.is_alive()),
            }

    def _run(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        sync_buf = self._load_json("sync.json").get("get_updates_buf", "")
        context_tokens = self._load_json("context-tokens.json")
        deliveries = self._load_delivery_records()
        failures = 0
        with self._lock:
            self._status = "connected"
            self._last_error = ""
        while not self._stop.is_set():
            try:
                response = self.client.get_updates(self.credentials, str(sync_buf or ""))
                ret = int(response.get("ret") or response.get("errcode") or 0)
                if ret != 0:
                    if ret == -14:
                        raise WeixinApiError("微信连接已过期，请重新扫码。")
                    raise WeixinApiError(
                        f"微信轮询失败：ret={ret} {response.get('errmsg') or ''}".strip()
                    )
                failures = 0
                messages = response.get("msgs") or []
                for raw in messages:
                    if not isinstance(raw, dict) or int(raw.get("message_type") or 0) != 1:
                        continue
                    sender_id = str(raw.get("from_user_id") or "").strip()
                    if self.credentials.user_id and sender_id != self.credentials.user_id:
                        continue
                    text = extract_weixin_text(raw)
                    message_id = self._delivery_id(raw, sender_id, text)
                    record = deliveries.get(message_id)
                    record_state = str(record.get("state") or "") if isinstance(record, dict) else ""
                    record_updated_at = int(record.get("updated_at") or 0) if isinstance(record, dict) else 0
                    if record_state == "sent":
                        continue
                    if (
                        record_state == "processing"
                        and int(time.time()) - record_updated_at < MESSAGE_PROCESSING_STALE_SECONDS
                    ):
                        continue
                    if not text:
                        deliveries[message_id] = {
                            "state": "reply_ready",
                            "reply_text": "我目前先支持微信文字和已转写的语音消息；图片、文件和原始语音将在下一版接入。",
                            "updated_at": int(time.time()),
                        }
                        self._save_delivery_records(deliveries)
                        self.client.send_text(
                            self.credentials,
                            to_user_id=sender_id,
                            context_token=str(raw.get("context_token") or ""),
                            text=str(deliveries[message_id]["reply_text"]),
                        )
                        deliveries[message_id] = {
                            "state": "sent",
                            "updated_at": int(time.time()),
                        }
                        self._save_delivery_records(deliveries)
                        continue
                    context_token = str(raw.get("context_token") or "")
                    if context_token:
                        context_tokens[sender_id] = context_token
                        self._save_json("context-tokens.json", context_tokens)
                    if record_state == "reply_ready":
                        reply_text = str(record.get("reply_text") or "")
                    else:
                        deliveries[message_id] = {
                            "state": "processing",
                            "updated_at": int(time.time()),
                        }
                        self._save_delivery_records(deliveries)
                        incoming = ChannelMessage(
                            channel="weixin",
                            account_id=self.credentials.account_id,
                            conversation_id="friday-main",
                            sender_id=sender_id,
                            message_id=message_id,
                            timestamp_ms=int(raw.get("create_time_ms") or int(time.time() * 1000)),
                            text=text,
                            context_token=context_token,
                            raw=raw,
                        )
                        try:
                            reply_text = self.on_message(incoming).text.strip()
                        except Exception as error:
                            reply_text = "Friday 暂时无法处理这条消息，请稍后再试。"
                            with self._lock:
                                self._last_error = f"消息处理失败：{type(error).__name__}: {error}"
                        deliveries[message_id] = {
                            "state": "reply_ready",
                            "reply_text": reply_text,
                            "updated_at": int(time.time()),
                        }
                        self._save_delivery_records(deliveries)
                    if reply_text:
                        self.client.send_text(
                            self.credentials,
                            to_user_id=sender_id,
                            context_token=context_token or str(context_tokens.get(sender_id) or ""),
                            text=reply_text,
                        )
                    deliveries[message_id] = {
                        "state": "sent",
                        "updated_at": int(time.time()),
                    }
                    self._save_delivery_records(deliveries)
                    with self._lock:
                        self._last_message_at = int(time.time())
                next_sync = response.get("get_updates_buf")
                if isinstance(next_sync, str) and next_sync:
                    sync_buf = next_sync
                    self._save_json("sync.json", {"get_updates_buf": sync_buf})
            except TimeoutError:
                continue
            except Exception as error:
                failures += 1
                with self._lock:
                    self._status = "error"
                    self._last_error = str(error)
                wait_seconds = 30 if failures >= 3 else 2
                if self._stop.wait(wait_seconds):
                    break
                if failures < 3:
                    with self._lock:
                        self._status = "connected"
        with self._lock:
            self._status = "stopped"

    def _load_json(self, name: str) -> dict[str, Any]:
        try:
            payload = json.loads((self.state_dir / name).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _save_json(self, name: str, payload: dict[str, Any]) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        path = self.state_dir / name
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)

    def _delivery_id(self, raw: dict[str, Any], sender_id: str, text: str) -> str:
        message_id = str(raw.get("message_id") or raw.get("seq") or "").strip()
        if message_id:
            return message_id
        fingerprint = "\n".join(
            [
                sender_id,
                str(raw.get("create_time_ms") or raw.get("create_time") or ""),
                text,
            ]
        )
        return f"derived-{hashlib.sha256(fingerprint.encode('utf-8')).hexdigest()[:32]}"

    def _load_delivery_records(self) -> dict[str, dict[str, Any]]:
        raw = self._load_json("dedupe.json")
        records: dict[str, dict[str, Any]] = {}
        for key, value in raw.items():
            message_id = str(key).strip()
            if not message_id:
                continue
            if isinstance(value, dict):
                records[message_id] = dict(value)
            elif isinstance(value, (int, float)):
                # The old format only recorded completion timestamps.
                records[message_id] = {"state": "sent", "updated_at": int(value)}
        return records

    def _save_delivery_records(self, records: dict[str, dict[str, Any]]) -> None:
        cutoff = int(time.time()) - MESSAGE_DEDUPE_RETENTION_SECONDS
        retained = {
            key: value
            for key, value in records.items()
            if int(value.get("updated_at") or 0) >= cutoff
        }
        records.clear()
        records.update(retained)
        self._save_json("dedupe.json", retained)


class WeixinGatewayManager:
    """Own QR sessions and one long-poll worker per Work Agent account."""

    def __init__(
        self,
        *,
        on_message: Callable[[int, ChannelMessage], ChannelReply],
        client_factory: Callable[[], WeixinIlinkClient] = WeixinIlinkClient,
    ) -> None:
        self.on_message = on_message
        self.client_factory = client_factory
        self._lock = threading.RLock()
        self._logins: dict[int, WeixinLoginSession] = {}
        self._workers: dict[int, WeixinChannelWorker] = {}

    def start_login(
        self,
        user_id: int,
        state_dir: Path,
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            active = self._logins.get(int(user_id))
            if active is not None and not active.expired and not force:
                return self._login_payload(active) or {}
            previous = load_credentials(state_dir)
            local_tokens = [previous.token] if previous else []
            session = self.client_factory().start_login(local_tokens)
            self._logins[int(user_id)] = session
            return self._login_payload(session)

    def poll_login(
        self,
        user_id: int,
        state_dir: Path,
        *,
        session_id: str,
        verify_code: str = "",
    ) -> dict[str, Any]:
        with self._lock:
            session = self._logins.get(int(user_id))
        if session is None or session.session_id != session_id:
            raise ValueError("微信登录会话不存在或已经失效，请重新生成二维码。")
        result = self.client_factory().poll_login(session, verify_code=verify_code)
        status = str(result.get("status") or "wait")
        if status == "confirmed":
            account_id = str(result.get("ilink_bot_id") or "").strip()
            token = str(result.get("bot_token") or "").strip()
            if not account_id or not token:
                raise WeixinApiError("微信已确认扫码，但没有返回机器人凭证。")
            credentials = WeixinCredentials(
                account_id=account_id,
                token=token,
                base_url=str(result.get("baseurl") or session.current_base_url or ILINK_BASE_URL),
                user_id=str(result.get("ilink_user_id") or "").strip(),
                saved_at=int(time.time()),
            )
            save_credentials(state_dir, credentials)
            with self._lock:
                self._logins.pop(int(user_id), None)
            self.ensure_worker(user_id, state_dir)
            return {
                "connected": True,
                "status": "connected",
                "account_id": credentials.account_id,
                "user_id": credentials.user_id,
                "message": "微信已连接到 Friday。",
            }
        if status == "binded_redirect":
            credentials = load_credentials(state_dir)
            if credentials:
                self.ensure_worker(user_id, state_dir)
                return {
                    "connected": True,
                    "status": "connected",
                    "account_id": credentials.account_id,
                    "user_id": credentials.user_id,
                    "message": "这个微信机器人已经连接，无需重复扫码。",
                }
        if status in {"expired", "verify_code_blocked"}:
            with self._lock:
                self._logins.pop(int(user_id), None)
        return {
            "connected": False,
            "status": status,
            "session_id": session.session_id,
            "qr_url": session.qrcode_url,
            "needs_verify_code": status == "need_verifycode",
            "message": login_status_message(status),
        }

    def ensure_worker(self, user_id: int, state_dir: Path) -> WeixinChannelWorker | None:
        credentials = load_credentials(state_dir)
        if credentials is None:
            return None
        with self._lock:
            worker = self._workers.get(int(user_id))
            if worker and worker.credentials.token == credentials.token:
                worker.start()
                return worker
            if worker:
                worker.stop()
            worker = WeixinChannelWorker(
                state_dir=state_dir,
                credentials=credentials,
                on_message=lambda message: self.on_message(int(user_id), message),
                client=self.client_factory(),
            )
            self._workers[int(user_id)] = worker
            worker.start()
            return worker

    def status(self, user_id: int, state_dir: Path) -> dict[str, Any]:
        credentials = load_credentials(state_dir)
        with self._lock:
            login = self._logins.get(int(user_id))
        worker = self.ensure_worker(user_id, state_dir) if credentials else None
        worker_status = worker.status() if worker else {
            "state": "disconnected",
            "last_error": "",
            "last_message_at": 0,
            "running": False,
        }
        return {
            "channel": "weixin",
            "connected": credentials is not None,
            "account_id": credentials.account_id if credentials else "",
            "user_id": credentials.user_id if credentials else "",
            "state": worker_status["state"],
            "running": worker_status["running"],
            "last_error": worker_status["last_error"],
            "last_message_at": worker_status["last_message_at"],
            "login": self._login_payload(login) if login and not login.expired else None,
            "capabilities": {
                "direct_messages": True,
                "ordinary_groups": False,
                "text": True,
                "voice_transcript": True,
                "media": False,
            },
        }

    def disconnect(self, user_id: int, state_dir: Path) -> dict[str, Any]:
        with self._lock:
            worker = self._workers.pop(int(user_id), None)
            self._logins.pop(int(user_id), None)
        if worker:
            worker.stop()
        clear_credentials(state_dir)
        return {"ok": True, "connected": False, "message": "微信连接已从本机移除。"}

    def qr_url(self, user_id: int, session_id: str) -> str:
        with self._lock:
            session = self._logins.get(int(user_id))
        if session is None or session.session_id != session_id or session.expired:
            raise ValueError("微信二维码已经失效，请重新生成。")
        return session.qrcode_url

    def stop_all(self) -> None:
        with self._lock:
            workers = list(self._workers.values())
            self._workers.clear()
            self._logins.clear()
        for worker in workers:
            worker.stop()

    @staticmethod
    def _login_payload(session: WeixinLoginSession | None) -> dict[str, Any] | None:
        if session is None:
            return None
        return {
            "session_id": session.session_id,
            "qr_url": session.qrcode_url,
            "status": session.status,
            "expires_in": max(
                0,
                int(LOGIN_TTL_SECONDS - (time.monotonic() - session.started_at)),
            ),
        }


def login_status_message(status: str) -> str:
    return {
        "wait": "等待微信扫码。",
        "scaned": "已扫码，请在手机微信中确认。",
        "need_verifycode": "请输入手机微信显示的数字。",
        "scaned_but_redirect": "已扫码，正在切换微信接入节点。",
        "expired": "二维码已过期，请重新生成。",
        "verify_code_blocked": "验证码错误次数过多，请重新生成二维码。",
    }.get(status, f"微信登录状态：{status}")
