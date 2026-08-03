from __future__ import annotations

import json
import unittest

from work_agent_core.approval_review import ApprovalReviewer, compact_review_transcript
from work_agent_core.config import ModelProfile
from work_agent_core.llm import LLMResponse
from work_agent_core.shell_tools import approval_action_id


class _ReviewerClient:
    def __init__(self, mode: str = "approve") -> None:
        self.mode = mode
        self.calls: list[dict] = []

    def chat(self, messages, **kwargs) -> LLMResponse:
        self.calls.append({"messages": messages, **kwargs})
        request = json.loads(messages[-1]["content"].split("\n", 1)[1])
        if self.mode == "invalid":
            content = "not json"
        else:
            action_id = request["action_id"] if self.mode != "wrong_action" else "approval-wrong"
            content = json.dumps(
                {
                    "action_id": action_id,
                    "decision": "approve",
                    "reason": "动作与用户明确要求一致，且范围受限。",
                },
                ensure_ascii=False,
            )
        return LLMResponse(content=content, raw={})


class ApprovalReviewerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = ModelProfile(
            name="reviewer-test",
            provider="openai-compatible",
            base_url="https://example.invalid/v1",
            model="test-model",
            api_key_env="UNUSED",
        )
        self.payload = {
            "command": "npm run build",
            "cwd": "/workspace",
            "timeout_seconds": 120,
            "risk_category": "NETWORK",
            "reason": "需要执行构建",
            "reviewable_by_model": True,
        }

    def test_separate_reviewer_approves_exact_action_without_tools(self) -> None:
        client = _ReviewerClient()
        reviewer = ApprovalReviewer(client=client, profile=self.profile)  # type: ignore[arg-type]

        review = reviewer.review(
            [{"role": "user", "content": "请完成修改并构建验证"}],
            self.payload,
        )

        self.assertTrue(review.approved)
        self.assertEqual(
            review.action_id,
            approval_action_id(command="npm run build", cwd="/workspace", timeout_seconds=120),
        )
        self.assertEqual(len(client.calls), 1)
        self.assertNotIn("tools", client.calls[0])
        self.assertEqual(client.calls[0]["reasoning_effort"], "light")

    def test_wrong_action_id_fails_closed(self) -> None:
        reviewer = ApprovalReviewer(
            client=_ReviewerClient("wrong_action"),  # type: ignore[arg-type]
            profile=self.profile,
        )

        review = reviewer.review([{"role": "user", "content": "构建"}], self.payload)

        self.assertFalse(review.approved)
        self.assertTrue(review.failed)
        self.assertIn("默认拒绝", review.reason)

    def test_invalid_response_fails_closed(self) -> None:
        reviewer = ApprovalReviewer(
            client=_ReviewerClient("invalid"),  # type: ignore[arg-type]
            profile=self.profile,
        )

        review = reviewer.review([{"role": "user", "content": "构建"}], self.payload)

        self.assertFalse(review.approved)
        self.assertTrue(review.failed)

    def test_fixed_boundary_denial_skips_model_call(self) -> None:
        client = _ReviewerClient()
        reviewer = ApprovalReviewer(client=client, profile=self.profile)  # type: ignore[arg-type]
        payload = {**self.payload, "reviewable_by_model": False}

        review = reviewer.review([{"role": "user", "content": "执行"}], payload)

        self.assertFalse(review.approved)
        self.assertEqual(client.calls, [])

    def test_transcript_excludes_tool_messages_and_is_bounded(self) -> None:
        transcript = compact_review_transcript(
            [
                {"role": "user", "content": "开始"},
                {"role": "tool", "content": "secret tool output"},
                {"role": "assistant", "content": "处理中"},
            ]
        )

        self.assertEqual([item["role"] for item in transcript], ["user", "assistant"])
        self.assertNotIn("secret tool output", json.dumps(transcript, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
