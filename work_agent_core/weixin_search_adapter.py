from __future__ import annotations

from datetime import datetime
import re
from urllib.parse import urlparse
from zoneinfo import ZoneInfo


_TIME_CONVERT_PATTERN = re.compile(r"timeConvert\(\s*['\"]?(\d{9,13})['\"]?\s*\)")
_PLAIN_TIMESTAMP_PATTERN = re.compile(r"^\s*(\d{9,13})\s*$")
_URL_APPEND_PATTERN = re.compile(r"url\s*\+=\s*(['\"])(.*?)\1", re.DOTALL)
_URL_ASSIGN_PATTERN = re.compile(r"(?:var\s+)?url\s*=\s*(['\"])(.*?)\1", re.DOTALL)


def extract_weixin_publish_timestamp(value: object) -> int | None:
    text = str(value or "").strip()
    match = _TIME_CONVERT_PATTERN.search(text) or _PLAIN_TIMESTAMP_PATTERN.fullmatch(text)
    if not match:
        return None
    timestamp = int(match.group(1))
    while timestamp > 9_999_999_999:
        timestamp //= 1000
    return timestamp


def normalize_weixin_publish_time(value: object, *, timezone: str = "Asia/Shanghai") -> str:
    text = str(value or "").strip()
    timestamp = extract_weixin_publish_timestamp(text)
    if timestamp is None:
        return text
    try:
        return datetime.fromtimestamp(timestamp, tz=ZoneInfo(timezone)).strftime("%Y-%m-%d %H:%M:%S")
    except (OSError, OverflowError, ValueError):
        return text


def extract_weixin_real_url(page_html: str) -> str:
    """Rebuild the WeChat URL emitted as JavaScript ``url +=`` fragments.

    Sogou deliberately splits the target URL across multiple statements. The
    upstream package skipped the first fragment, so every reconstructed URL
    was incomplete. This parser joins every fragment and accepts only the
    official WeChat article host.
    """

    pieces = [_decode_javascript_string(match.group(2)) for match in _URL_APPEND_PATTERN.finditer(page_html)]
    if pieces:
        candidate = "".join(pieces)
    else:
        assignment = _URL_ASSIGN_PATTERN.search(page_html)
        candidate = _decode_javascript_string(assignment.group(2)) if assignment else ""
    candidate = candidate.replace("&amp;", "&").replace("@", "").strip()
    if not candidate:
        return ""
    if candidate.startswith("//"):
        candidate = f"https:{candidate}"
    elif candidate.startswith("mp.weixin.qq.com"):
        candidate = f"https://{candidate}"
    elif candidate.startswith("weixin.qq.com"):
        candidate = f"https://mp.{candidate}"
    elif candidate.startswith("/s?"):
        candidate = f"https://mp.weixin.qq.com{candidate}"
    elif not candidate.startswith(("http://", "https://")):
        return ""

    parsed = urlparse(candidate)
    if parsed.hostname != "mp.weixin.qq.com":
        return ""
    return candidate.replace("http://mp.weixin.qq.com", "https://mp.weixin.qq.com", 1)


def _decode_javascript_string(value: str) -> str:
    text = value.replace(r"\/", "/")
    text = re.sub(r"\\x([0-9a-fA-F]{2})", lambda match: chr(int(match.group(1), 16)), text)
    text = re.sub(r"\\u([0-9a-fA-F]{4})", lambda match: chr(int(match.group(1), 16)), text)
    return text
