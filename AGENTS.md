# 本工作区的约定

> 这是项目上下文文件，由智能体在启动时读取。写这里而不是写进系统提示词，
> 是因为它约束的是**这个工作区**，换一个工作区就不该带着走。

## 运行环境

本项目只有一套受支持的 Python 环境：工作区根目录的 `.venv`。
Python、pip、Office 脚本、会议 ASR 和 VAD 都必须用它。

不要创建或调用 `.venv_agent`、`meeting_audio_minutes/.venv_project`、`.venv_deepfilter`、
Conda、系统 Python 或临时 venv，也不要用 `--user` 安装包。

- 环境检查：`scripts/runtime_env.sh check`
- 缺依赖时说明原因，请求用户批准后再 `scripts/runtime_env.sh bootstrap`
- Node/npm 使用该脚本的 `node` / `npm` 子命令

FFmpeg、LibreOffice、Poppler 是声明过的原生工具，不代表额外的 Python 环境。
DeepFilterNet 因 Python 版本冲突不进主运行环境，音频降噪默认用 FFmpeg。

## 测试

全量回归：

```
.venv/bin/python -m unittest discover -s tests -q
```

## 工作区文件的去处

判据是**生命周期**，不是内容类型：这份东西能不能重建。

| 放哪 | 什么 | 能否删 |
|---|---|---|
| `meet_files/attachments/` `voice_inputs/` | 用户给的原件 | 不能，只进不出 |
| `meet_files/材料/` `文字稿/` `会议项目/` | 交付物与转写稿 | 不能 |
| `meet_files/asr_full/` | 会议转写正本 | 不能 |
| `meet_files/execution/` `office_workspace/` `debug_traces/` | 机器产物：沙箱快照、解包目录、调试轨迹 | **能，随时** |
| `_` 开头的目录 | 隔离区、临时解包 | **能，且不进检索索引** |

两条规矩：

- **不要往 `meet_files/` 根目录直接写文件。** 转写稿进 `文字稿/`，成品进 `材料/`。
- **机器产物不进检索索引。** 排除规则见 `work_agent_core/recall/sync.py` 的
  `DEFAULT_SKIP_DIRECTORIES`。
