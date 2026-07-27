# Work Agent Core

这是当前工作智能体的基础框架，目标是先搭好可扩展底座：

1. 通用 OpenAI-compatible 模型接入层。
2. 模型 profile 注册、查看和切换。
3. ReAct 循环，默认最多 30 轮工具调用。
4. ToolBus 工具总线：core 工具常驻 API `tools`；技能和 MCP 后端分层挂载，分别通过 `sys_skill`、`mcporter` 按需查看与调用。
5. 会议纪要生成 skill。

## 配置模型

模型 profile 写在：

```text
config/model_profiles.json
```

当前已注册：

1. `deepseek-v4-pro`
2. `deepseek-v4-flash`

密钥只从环境变量读取，不写入仓库文件：

```bash
export DEEPSEEK_API_KEY="你的key"
export WORK_AGENT_MODEL_PROFILE="deepseek-v4-pro"
```

也可以在项目根目录创建本地 `.env` 文件。该文件已被 `.gitignore` 排除，不会提交：

```bash
cp .env.example .env
# 然后把 .env 里的 DEEPSEEK_API_KEY 改成你的真实 key
```

查看模型：

```bash
python -m work_agent_core.cli models list
python -m work_agent_core.cli models current
```

持久切换默认模型：

```bash
python -m work_agent_core.cli models use deepseek-v4-flash
python -m work_agent_core.cli models use deepseek-v4-pro
```

临时切换模型：

```bash
python -m work_agent_core.cli --profile deepseek-v4-flash chat "回复ok"
```

## 运行 ReAct 智能体

```bash
python -m work_agent_core.cli run "读取会议纪要规范并总结工作提交版规则"
```

默认 ReAct 最大步数为 30，可改：

```bash
python -m work_agent_core.cli run "..." --max-steps 30
```

## MCP / 工具总线

运行时不再把所有技能和外部工具 schema 一次性塞进模型请求。`ReActAgent` 只依赖 `ToolBus`：

```text
LLM API tools schema -> core + sys_skill + mcporter
                         |          |
                         v          v
                    skill providers  MCP providers
```

MCP stdio server 配置写在：

```text
config/mcp_servers.json
```

格式兼容 Friday 风格的 `mcp_servers`、Claude 常见的 `mcpServers`，也支持本项目默认的 `servers`：

```json
{
  "servers": {
    "my-server": {
      "type": "stdio",
      "description": "任意 MCP stdio server",
      "command": "python",
      "args": ["path/to/server.py"],
      "cwd": ".",
      "env": {},
      "enabled": true
    }
  }
}
```

后端会在 `/api/tools` 返回当前模型常驻工具；provider 状态仍包含隐藏的技能/MCP 后端，便于诊断。具体技能工具通过 `sys_skill(op=list/open/show/call)` 分层使用。

## 轻量 Debug Trace

后端默认开启本地 JSONL trace（可用 `WORK_AGENT_DEBUG_TRACE=0` 关闭）。它只记录每轮运行的可观测摘要，不记录 API Key，并会对 `token`、`authorization`、`password`、`secret` 等字段脱敏。

落盘位置：

```text
meet_files/debug_traces/<conversation_id>.jsonl
meet_files/debug_traces/_recent.jsonl
```

记录内容包括：

- 会话 working memory 准备状态：消息数、summary 长度、是否压缩。
- LLM 规划请求/响应摘要：耗时、finish_reason、content 字符数、tool 名。
- 工具调用：工具名、参数摘要、耗时、结果字符数/预览、错误。
- 最终回复：步数、是否用工具、最终内容长度。

只读接口：

```text
GET /api/debug/traces?conversation_id=<id>&limit=200
GET /api/debug/traces?trace_id=<trace-id>&limit=200
GET /api/debug/traces?limit=200
```

## 单智能体 Turn Runtime

每次 `/api/agent/chat-stream` 会创建一个后端 turn。它不是多智能体任务图，只表示同一会话里的“一轮用户请求 + 单个 ReActAgent 运行”。

落盘位置：

```text
meet_files/conversation_history/turns/<turn_id>.json
```

保存内容包括：

- `conversation_id`、`turn_id`、`status`、`trace_id`、模型 profile。
- 该轮 SSE/activity 事件，含 `event_index`，可用于刷新后恢复活动流。
- `final_message`、`error`、`cancel_requested`。

只读/控制接口：

```text
GET  /api/agent/turns/<turn_id>
GET  /api/agent/turns/<turn_id>/events?after=<event_index>
POST /api/agent/turns/<turn_id>/cancel
```

前端收到首个 `event: "turn"` 后记录 `turn_id`；停止当前轮时会先请求 cancel，再中断本地 SSE。ReAct 循环在模型/工具轮询点检查取消标记并把 turn 标成 `cancelled`。

## 直接运行会议纪要 skill

```bash
python -m work_agent_core.cli skill meeting-minutes \
  --transcript meet_files/example/transcript.txt \
  --meeting-name 示例项目会议 \
  --output-dir meet_files
```

该 skill 会根据：

```text
meeting_audio_minutes/meeting_minutes_spec.md
```

输出两份 Markdown：

1. `*_会议沟通内容整理_内部留档版.md`
2. `*_会议纪要_工作提交版.md`

## 增加新模型端点

在 `config/model_profiles.json` 增加一项：

```json
{
  "name": "my-openai-compatible-model",
  "provider": "custom",
  "base_url": "https://example.com/v1",
  "model": "model-name",
  "api_key_env": "MY_MODEL_API_KEY",
  "temperature": 0.6,
  "max_tokens": 4096,
  "timeout_seconds": 120
}
```

然后：

```bash
export MY_MODEL_API_KEY="..."
python -m work_agent_core.cli --profile my-openai-compatible-model chat "测试"
```

也可以用命令添加：

```bash
python -m work_agent_core.cli models add my-openai-compatible-model \
  --base-url https://example.com/v1 \
  --model model-name \
  --api-key-env MY_MODEL_API_KEY \
  --set-default
```

## 运行 Web 工作台

前端源码在：

```text
web_frontend/
```

构建前端：

```bash
cd web_frontend
npm install
npm run build
cd ..
```

启动本地 Web 服务：

```bash
python3 -m work_agent_core.web_server --host 127.0.0.1 --port 8787 --workspace "$PWD"
```

然后打开：

```text
http://127.0.0.1:8787/
```

### 作为 macOS 常驻服务运行

先构建前端静态文件，然后安装 `launchd` 服务。常驻服务只运行 8787 后端；后端会直接托管 `web_frontend/dist`，不需要让 Vite 常驻。

```bash
cd web_frontend
npm run build
cd ..
chmod +x scripts/work_agent_service.sh scripts/work_agent_service_ctl.sh
scripts/work_agent_service_ctl.sh install
```

服务会在登录后启动，异常退出后自动重启。状态和日志：

```bash
scripts/work_agent_service_ctl.sh status
scripts/work_agent_service_ctl.sh logs
scripts/work_agent_service_ctl.sh errors
```

Qwen3-ASR 不会随 Web 服务启动而加载；首次转写才启动 worker，最后一次转写后默认空闲 90 秒自动卸载。可通过 `WORK_AGENT_ASR_IDLE_TIMEOUT_SECONDS` 调整。

## 唯一项目运行环境

项目根目录的 `.venv` 是唯一受支持的 Python 环境，统一承载主服务、Office
技能、会议 ASR 与实时 VAD。运行时不会回退到系统 Python、Conda、Codex 自带
Python 或技能私有 venv，避免“同一条命令在不同入口表现不同”。

统一入口：

```bash
scripts/runtime_env.sh bootstrap  # 创建或补齐环境
scripts/runtime_env.sh check      # 检查解释器、模块和原生工具
scripts/runtime_env.sh python ... # 使用项目 Python
scripts/runtime_env.sh pip ...    # 只向项目环境安装包
scripts/runtime_env.sh node ...   # 使用声明的 Node
scripts/runtime_env.sh npm ...    # 使用声明的 npm
```

Python 依赖统一记录在根目录 `requirements-runtime.txt`。FFmpeg、LibreOffice
和 Poppler 是运行契约中的原生工具，不是额外 Python 环境。DeepFilterNet 目前
只支持与主环境冲突的 Python/torch 组合，已退出主链路；音频降噪统一使用
FFmpeg。

## 实时转写的 WebRTC VAD

实时转写页面会同时做两件事：

1. 浏览器 `MediaRecorder` 保存当前音频片段，片段结束后交给本地 Qwen3-ASR 转写。
2. 浏览器把麦克风 PCM 重采样为 16kHz/30ms 小帧，调用 `/api/speech/vad`，由项目内 WebRTC VAD 判断停顿并切段。

WebRTC VAD 已包含在统一环境中：

```bash
cd /path/to/work_agent
scripts/runtime_env.sh bootstrap
```

后端会优先使用：

```text
.venv/bin/python
```

如果该环境没有 `webrtcvad`，前端会回退到音量阈值切段，但页面状态会提示 WebRTC VAD 不可用。
