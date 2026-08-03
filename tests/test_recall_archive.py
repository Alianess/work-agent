from __future__ import annotations

import unittest

from work_agent_core.recall_archive import (
    build_recall_episodes,
    compact_messages_for_archive,
    render_episode_text,
)


class RecallArchiveTests(unittest.TestCase):
    def test_small_turn_merges_into_neighboring_work_episode(self) -> None:
        messages = [
            {"role": "user", "content": "今天几点了？"},
            {"role": "assistant", "content": "现在是 20:05。"},
            {"role": "user", "content": "继续检查历史检索切片。"},
            {
                "role": "assistant",
                "content": "我先核对索引结构，再调整父子切片。",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "shell_exec",
                            "arguments": '{"command":"rg -n chunk history_recall.py"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "shell_exec",
                "content": "数百行终端回显\nwork_agent_core/history_recall.py",
            },
            {"role": "assistant", "content": "已完成父子切片调整。"},
        ]

        episodes = build_recall_episodes(messages)

        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0]["turn_count"], 2)
        rendered = render_episode_text(episodes[0])
        self.assertLess(rendered.index("今天几点了"), rendered.index("现在是 20:05"))
        self.assertLess(rendered.index("继续检查"), rendered.index("我先核对"))
        self.assertLess(
            rendered.index("我先核对"),
            rendered.index("运行命令：rg -n chunk history_recall.py"),
        )
        self.assertLess(
            rendered.index("运行命令：rg -n chunk history_recall.py"),
            rendered.index("shell_exec 已完成"),
        )
        self.assertLess(
            rendered.index("shell_exec 已完成"),
            rendered.index("已完成父子切片调整"),
        )
        self.assertIn("运行命令：rg -n chunk history_recall.py", rendered)
        self.assertNotIn("数百行终端回显", rendered)
        self.assertNotIn("执行轨迹：", rendered)
        self.assertNotIn("最终答复：", rendered)
        self.assertNotIn("关键产物与引用：", rendered)

    def test_public_path_is_verbatim_but_tool_bulk_is_folded(self) -> None:
        path_note = "先检查现有索引签名；如果切片格式变化，就整体重建，避免新旧父块混用。"
        messages = [
            {"role": "user", "content": "把切片做精细一点。"},
            {
                "role": "assistant",
                "content": path_note,
                "reasoning_content": "provider-private-token-stream",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "shell_exec",
                            "arguments": (
                                '{"command":"rg -n \\"session_signature\\" '
                                'work_agent_core/history_recall.py"}'
                            ),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "shell_exec",
                "content": "line 1\nline 2\nline 3\n/Users/example/work_agent/work_agent_core/history_recall.py",
            },
            {"role": "assistant", "content": "索引版本已更新。"},
        ]

        episode = build_recall_episodes(messages)[0]
        rendered = render_episode_text(episode)
        compacted = compact_messages_for_archive(messages)

        self.assertIn(path_note, rendered)
        self.assertNotIn("provider-private-token-stream", rendered)
        self.assertNotIn("line 1", rendered)
        self.assertEqual(compacted[1]["content"], path_note)
        self.assertNotIn("reasoning_content", compacted[1])
        self.assertIn(
            "history_recall.py",
            compacted[1]["tool_calls"][0]["function"]["arguments"],
        )
        self.assertNotIn("line 1", compacted[2]["content"])
        self.assertLess(rendered.index(path_note), rendered.index("运行命令："))
        self.assertLess(rendered.index("运行命令："), rendered.index("shell_exec 已完成"))
        self.assertLess(rendered.index("shell_exec 已完成"), rendered.index("索引版本已更新"))

    def test_artifact_reference_stays_at_its_original_tool_event(self) -> None:
        messages = [
            {"role": "user", "content": "生成会议纪要。"},
            {
                "role": "assistant",
                "content": "我先生成文件，再检查结果。",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "write_text_file",
                            "arguments": '{"path":"meet_files/会议纪要.md"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "write_text_file",
                "content": '{"ok":true,"output_path":"meet_files/会议纪要.md"}',
            },
            {"role": "assistant", "content": "文件已经生成。"},
        ]

        rendered = render_episode_text(build_recall_episodes(messages)[0])

        self.assertLess(rendered.index("我先生成文件"), rendered.index("工具调用"))
        self.assertLess(rendered.index("工具调用"), rendered.index("工具结果"))
        self.assertLess(rendered.index("工具结果"), rendered.index("文件已经生成"))
        self.assertIn("output_path=meet_files/会议纪要.md", rendered)
        self.assertNotIn("关键产物与引用：", rendered)

    def test_orphan_imported_user_message_does_not_hide_later_completed_turn(self) -> None:
        episodes = build_recall_episodes(
            [
                {"role": "user", "content": "带附件的旧请求"},
                {"role": "user", "content": "继续"},
                {"role": "assistant", "content": "我先读取附件。", "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "read_text_file", "arguments": '{"path":"a.md"}'},
                    }
                ]},
                {"role": "tool", "tool_call_id": "call-1", "name": "read_text_file", "content": "正文"},
                {"role": "assistant", "content": "附件已经处理完成。"},
            ]
        )

        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0]["user_texts"], ["带附件的旧请求", "继续"])
        self.assertIn("带附件的旧请求", render_episode_text(episodes[0]))
        self.assertIn("我先读取附件", render_episode_text(episodes[0]))


if __name__ == "__main__":
    unittest.main()
