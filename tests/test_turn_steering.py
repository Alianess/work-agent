"""用户中途说话时，循环应当接得住，而不是只能被打断。

这两条队列存在的意义是让"模型提前收尾"变成可恢复的：人补一句就继续，
所以系统提示词不必反复恐吓模型不许提前收尾。
"""

from __future__ import annotations

import tempfile
import unittest

from work_agent_core.config import ModelProfile
from work_agent_core.llm import LLMResponse
from work_agent_core.react import LoopHooks, ReActAgent
from work_agent_core.tool_bus import LocalToolProvider, ToolBus
from work_agent_core.tools import Tool
from work_agent_core.turn_store import TurnStore


def _profile() -> ModelProfile:
    return ModelProfile(
        name="steering-test",
        provider="openai-compatible",
        base_url="https://example.invalid/v1",
        model="test-model",
        api_key_env="UNUSED",
    )


def _text_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        raw={"choices": [{"message": {"role": "assistant", "content": content}}]},
    )


def _tool_response(name: str, call_id: str) -> LLMResponse:
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }
        ],
    }
    return LLMResponse(content="", raw={"choices": [{"message": message}]})


class _ScriptedClient:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = responses
        self.message_snapshots: list[list[dict]] = []

    def chat_tools_stream(self, messages, *_args, **_kwargs) -> LLMResponse:
        self.message_snapshots.append([dict(item) for item in messages])
        return self.responses[min(len(self.message_snapshots) - 1, len(self.responses) - 1)]


def _ping_tools() -> tuple[ToolBus, list[dict]]:
    calls: list[dict] = []
    provider = LocalToolProvider("core")
    provider.registry.register(
        Tool(
            name="ping",
            description="ping",
            parameters={"type": "object", "properties": {}},
            handler=lambda arguments: calls.append(arguments) or "pong",
        )
    )
    tools = ToolBus()
    tools.add_provider(provider)
    return tools, calls


class FollowUpQueueTests(unittest.TestCase):
    def test_queued_message_keeps_the_turn_going_instead_of_ending_it(self) -> None:
        client = _ScriptedClient([_text_response("先到这里"), _text_response("补完了")])
        tools, _ = _ping_tools()
        queued = [["再把结论写一句"]]

        agent = ReActAgent(
            client=client,  # type: ignore[arg-type]
            profile=_profile(),
            tools=tools,
            pending_messages=lambda: queued.pop(0) if queued else [],
        )
        events = list(agent.iter_message_events([{"role": "user", "content": "开始"}]))

        # Two model calls: the agent would have stopped after the first.
        self.assertEqual(len(client.message_snapshots), 2)
        second_request = client.message_snapshots[1]
        self.assertIn(
            "再把结论写一句",
            [str(item.get("content") or "") for item in second_request if item.get("role") == "user"],
        )
        final = next(event for event in events if event["event"] == "final")
        # What the model said before being nudged is not thrown away.
        self.assertIn("先到这里", final["content"])
        self.assertIn("补完了", final["content"])

    def test_empty_queue_ends_the_turn_as_before(self) -> None:
        client = _ScriptedClient([_text_response("完成")])
        tools, _ = _ping_tools()

        agent = ReActAgent(
            client=client,  # type: ignore[arg-type]
            profile=_profile(),
            tools=tools,
            pending_messages=lambda: [],
        )
        events = list(agent.iter_message_events([{"role": "user", "content": "开始"}]))

        self.assertEqual(len(client.message_snapshots), 1)
        self.assertEqual(
            next(event for event in events if event["event"] == "final")["content"], "完成"
        )

    def test_broken_inbox_never_takes_the_turn_down(self) -> None:
        client = _ScriptedClient([_text_response("完成")])
        tools, _ = _ping_tools()

        def _explode() -> list[str]:
            raise RuntimeError("inbox unavailable")

        agent = ReActAgent(
            client=client,  # type: ignore[arg-type]
            profile=_profile(),
            tools=tools,
            pending_messages=_explode,
        )
        events = list(agent.iter_message_events([{"role": "user", "content": "开始"}]))

        self.assertEqual(
            next(event for event in events if event["event"] == "final")["content"], "完成"
        )


class SteeringQueueTests(unittest.TestCase):
    def test_message_typed_during_tool_work_lands_before_the_next_model_call(self) -> None:
        client = _ScriptedClient([_tool_response("ping", "call-1"), _text_response("收到，照办")])
        tools, calls = _ping_tools()
        queued = [["改用另一个口径"]]

        agent = ReActAgent(
            client=client,  # type: ignore[arg-type]
            profile=_profile(),
            tools=tools,
            pending_messages=lambda: queued.pop(0) if queued else [],
        )
        events = list(agent.iter_message_events([{"role": "user", "content": "开始"}]))

        self.assertEqual(calls, [{}])
        second_request = client.message_snapshots[1]
        user_texts = [
            str(item.get("content") or "") for item in second_request if item.get("role") == "user"
        ]
        self.assertIn("改用另一个口径", user_texts)
        # It arrives after the tool result, i.e. it steers the next decision.
        roles = [str(item.get("role") or "") for item in second_request]
        self.assertLess(roles.index("tool"), len(roles) - 1 - roles[::-1].index("user"))
        self.assertTrue(
            any(event.get("activity_type") == "user_steering" for event in events)
        )


class TurnInboxTests(unittest.TestCase):
    def test_structured_inbox_is_not_removed_until_acknowledged(self) -> None:
        store = TurnStore(tempfile.mkdtemp())
        turn = store.create(conversation_id="c1")
        store.enqueue_message(turn.id, "补充口径")

        first = store.peek_message_events(turn.id)
        second = store.peek_message_events(turn.id)
        self.assertEqual(first, second)
        self.assertEqual(first[0]["kind"], "user_followup")
        self.assertEqual(first[0]["payload"]["content"], "补充口径")
        self.assertEqual(len(store.load(turn.id).queued_messages), 1)

        removed = store.ack_message_events(turn.id, [first[0]["event_id"]])
        self.assertEqual(removed, 1)
        self.assertEqual(store.peek_message_events(turn.id), [])

    def test_draining_is_exactly_once(self) -> None:
        store = TurnStore(tempfile.mkdtemp())
        turn = store.create(conversation_id="c1")

        store.enqueue_message(turn.id, "第一句")
        store.enqueue_message(turn.id, "   ")
        store.enqueue_message(turn.id, "第二句")

        self.assertEqual(store.drain_messages(turn.id), ["第一句", "第二句"])
        self.assertEqual(store.drain_messages(turn.id), [])
        self.assertEqual(store.load(turn.id).queued_messages, [])

    def test_finished_turn_does_not_accept_more(self) -> None:
        store = TurnStore(tempfile.mkdtemp())
        turn = store.create(conversation_id="c1")
        store.mark_cancelled(turn.id)

        store.enqueue_message(turn.id, "太晚了")

        self.assertEqual(store.load(turn.id).queued_messages, [])


class LoopHookTests(unittest.TestCase):
    def test_transform_context_can_inject_before_the_request(self) -> None:
        client = _ScriptedClient([_text_response("完成")])
        tools, _ = _ping_tools()

        def _inject(runtime, *, step):
            runtime.append_message({"role": "system", "content": f"记忆常驻块 step={step}"})
            return iter(())

        agent = ReActAgent(
            client=client,  # type: ignore[arg-type]
            profile=_profile(),
            tools=tools,
            hooks=LoopHooks(transform_context=_inject),
        )
        list(agent.iter_message_events([{"role": "user", "content": "开始"}]))

        contents = [str(item.get("content") or "") for item in client.message_snapshots[0]]
        self.assertIn("记忆常驻块 step=1", contents)

    def test_a_broken_hook_cannot_end_the_turn(self) -> None:
        client = _ScriptedClient([_text_response("完成")])
        tools, _ = _ping_tools()

        def _explode(_runtime, **_kwargs):
            raise RuntimeError("hook is broken")

        agent = ReActAgent(
            client=client,  # type: ignore[arg-type]
            profile=_profile(),
            tools=tools,
            hooks=LoopHooks(transform_context=_explode, should_stop_after_turn=_explode),
        )
        events = list(agent.iter_message_events([{"role": "user", "content": "开始"}]))

        self.assertEqual(
            next(event for event in events if event["event"] == "final")["content"], "完成"
        )

    def test_should_stop_after_turn_ends_the_turn_cleanly(self) -> None:
        client = _ScriptedClient([_tool_response("ping", "call-1"), _text_response("不该走到这里")])
        tools, calls = _ping_tools()

        agent = ReActAgent(
            client=client,  # type: ignore[arg-type]
            profile=_profile(),
            tools=tools,
            hooks=LoopHooks(should_stop_after_turn=lambda _runtime, **_kwargs: True),
        )
        events = list(agent.iter_message_events([{"role": "user", "content": "开始"}]))

        self.assertEqual(calls, [{}])
        self.assertEqual(len(client.message_snapshots), 1)
        final = next(event for event in events if event["event"] == "final")
        self.assertTrue(final.get("stopped_by_hook"))


class ToolHookTests(unittest.TestCase):
    def test_before_tool_call_can_block_and_answer_in_its_place(self) -> None:
        client = _ScriptedClient([_tool_response("ping", "call-1"), _text_response("好的")])
        tools, calls = _ping_tools()

        def _blocking_hook(*, tool_name, **_kwargs):
            yield {
                "event": "activity",
                "phase": "thinking",
                "title": f"拦下 {tool_name}",
                "step": 1,
            }
            return {"block": True, "observation": "策略不允许，已跳过。"}

        agent = ReActAgent(
            client=client,  # type: ignore[arg-type]
            profile=_profile(),
            tools=tools,
            hooks=LoopHooks(before_tool_call=_blocking_hook),
        )
        list(agent.iter_message_events([{"role": "user", "content": "开始"}]))

        self.assertEqual(calls, [])
        tool_texts = [
            str(item.get("content") or "")
            for item in client.message_snapshots[1]
            if item.get("role") == "tool"
        ]
        self.assertEqual(tool_texts, ["策略不允许，已跳过。"])

    def test_after_tool_call_can_rewrite_what_the_model_sees(self) -> None:
        client = _ScriptedClient([_tool_response("ping", "call-1"), _text_response("好的")])
        tools, calls = _ping_tools()

        def _rewriting_hook(*, observation, **_kwargs):
            if False:  # pragma: no cover
                yield {}
            return {"observation": f"[已加工] {observation}"}

        agent = ReActAgent(
            client=client,  # type: ignore[arg-type]
            profile=_profile(),
            tools=tools,
            hooks=LoopHooks(after_tool_call=_rewriting_hook),
        )
        list(agent.iter_message_events([{"role": "user", "content": "开始"}]))

        self.assertEqual(calls, [{}])
        tool_texts = [
            str(item.get("content") or "")
            for item in client.message_snapshots[1]
            if item.get("role") == "tool"
        ]
        self.assertEqual(tool_texts, ["[已加工] pong"])

    def test_a_broken_tool_hook_never_fails_the_tool(self) -> None:
        client = _ScriptedClient([_tool_response("ping", "call-1"), _text_response("好的")])
        tools, calls = _ping_tools()

        def _explode(**_kwargs):
            raise RuntimeError("hook is broken")
            yield  # pragma: no cover

        agent = ReActAgent(
            client=client,  # type: ignore[arg-type]
            profile=_profile(),
            tools=tools,
            hooks=LoopHooks(before_tool_call=_explode, after_tool_call=_explode),
        )
        list(agent.iter_message_events([{"role": "user", "content": "开始"}]))

        self.assertEqual(calls, [{}])
        tool_texts = [
            str(item.get("content") or "")
            for item in client.message_snapshots[1]
            if item.get("role") == "tool"
        ]
        self.assertEqual(tool_texts, ["pong"])


if __name__ == "__main__":
    unittest.main()
