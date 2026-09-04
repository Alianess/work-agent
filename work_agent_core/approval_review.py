from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any
import json
import re

from .config import ModelProfile
from .llm import Message, OpenAICompatibleClient
from .shell_tools import approval_action_id


APPROVAL_REVIEW_SYSTEM_PROMPT = """你是无工具、无记忆、无规划能力的安全策略分类器，不是智能体。
你只能给系统提供的这一项精确动作分类，不能修改动作、扩展权限、调用工具、安排任务或批准后续动作。

判断原则（靠近 Claude Code / Codex 的分层权限模式）：
1. 你审查的是当前这一项精确动作，不是给主智能体授予一类永久权限。
2. 用户当前明确要求或完成其要求显然必需，且动作范围收敛、可由隔离环境约束时，批准常规开发动作：构建、测试、渲染、工作区内脚本，以及项目环境中的依赖安装。
3. 受限依赖安装只能使用项目规定的环境和官方包源；命令中出现任意 URL、git 源、备用 registry、凭证参数或绕过项目环境的写法，拒绝。
4. 删除属于不可逆边界，不由你批准；固定策略会拒绝宽范围删除，明确的单项删除也交给人工确认。
5. 涉及凭证、隐私外传、任意/未受限联网、持久化提权、系统设置、宿主敏感目录或越出工作区时拒绝。固定安全策略已经拒绝的动作不得重新解释为安全。
6. 审查失败、动作与当前任务无关或有实质疑问时拒绝；只有固定拒绝边界和审查失败才回到人工确认，不要把普通隔离开发动作一律升级给人。

只返回一个 JSON 对象，不要 Markdown：
{"action_id":"原样返回","decision":"approve或deny","reason":"简短且具体的中文理由"}
"""


@dataclass(frozen=True)
class ApprovalReview:
    decision: str
    reason: str
    action_id: str
    reviewer_profile: str
    failed: bool = False

    @property
    def approved(self) -> bool:
        return self.decision == "approve" and not self.failed


class ApprovalReviewer:
    """A bounded classifier call, never a second agent loop.

    It owns no tools, memory, inbox, plan or retry loop.  The single ReAct
    agent remains the only component that can act; this classifier can only
    return approve/deny for one action id and fails closed.
    """

    def __init__(
        self,
        *,
        client: OpenAICompatibleClient,
        profile: ModelProfile,
    ) -> None:
        self.client = client
        self.profile = replace(
            profile,
            name=f"{profile.name} · 审查",
            temperature=0,
            max_tokens=min(profile.max_tokens, 500),
            timeout_seconds=min(profile.timeout_seconds, 30),
        )

    def review(
        self,
        session_messages: list[Message],
        approval_payload: dict[str, Any],
    ) -> ApprovalReview:
        action_id = approval_action_id(
            command=str(approval_payload.get("command") or ""),
            cwd=str(approval_payload.get("cwd") or "."),
            timeout_seconds=int(approval_payload.get("timeout_seconds") or 120),
        )
        if approval_payload.get("reviewable_by_model") is not True:
            return ApprovalReview(
                decision="deny",
                reason="该动作不在安全策略分类器可批准的固定边界内。",
                action_id=action_id,
                reviewer_profile=self.profile.name,
            )

        request = {
            "action_id": action_id,
            "action": {
                "tool": "shell_exec",
                "command": str(approval_payload.get("command") or ""),
                "cwd": str(approval_payload.get("cwd") or "."),
                "timeout_seconds": int(approval_payload.get("timeout_seconds") or 120),
                "risk_category": str(approval_payload.get("risk_category") or "EXECUTE"),
                "policy_reason": str(approval_payload.get("reason") or ""),
            },
            "conversation": compact_review_transcript(session_messages),
        }
        try:
            response = self.client.chat(
                [
                    {"role": "system", "content": APPROVAL_REVIEW_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": "请审查以下 JSON 数据：\n" + json.dumps(request, ensure_ascii=False),
                    },
                ],
                profile=self.profile,
                temperature=0,
                max_tokens=500,
                reasoning_effort="light",
            )
            parsed = parse_review_response(response.content)
            returned_action_id = str(parsed.get("action_id") or "")
            decision = str(parsed.get("decision") or "").strip().lower()
            reason = str(parsed.get("reason") or "").strip()
            if returned_action_id != action_id:
                raise ValueError("审查结果未绑定当前精确动作")
            if decision not in {"approve", "deny"}:
                raise ValueError("审查结果 decision 无效")
            if not reason:
                raise ValueError("审查结果缺少理由")
            return ApprovalReview(
                decision=decision,
                reason=reason[:600],
                action_id=action_id,
                reviewer_profile=self.profile.name,
            )
        except Exception as error:
            return ApprovalReview(
                decision="deny",
                reason=f"独立审查失败，已按默认拒绝处理：{type(error).__name__}: {error}",
                action_id=action_id,
                reviewer_profile=self.profile.name,
                failed=True,
            )


def compact_review_transcript(messages: list[Message]) -> list[dict[str, str]]:
    transcript: list[dict[str, str]] = []
    remaining = 14_000
    for message in reversed(messages[-16:]):
        role = str(message.get("role") or "")
        if role not in {"user", "assistant"}:
            continue
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        content = content[: min(4_000, remaining)]
        transcript.append({"role": role, "content": content})
        remaining -= len(content)
        if remaining <= 0:
            break
    transcript.reverse()
    return transcript


def parse_review_response(content: str) -> dict[str, Any]:
    text = str(content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("审查模型没有返回 JSON")
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("审查模型返回值不是对象")
    return payload
