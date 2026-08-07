# Apple 日历与提醒事项集成

## 目标与边界

Work Agent 通过 macOS `EventKit` 展示当前 Mac 系统账户已同步的 Apple 日历和提醒事项。网页「更多工具 → 日程与提醒」是**只读可视化页面**：用户自行管理日程和提醒事项，请使用 iPhone。

在对话中，用户明确要求“新增/添加/创建待办”或“提醒我……”时，AI 可以新增一条 Apple「提醒事项」。它不能创建 Apple 日历事件，也不能根据会议、项目、日报或聊天上下文推断并写入待办。

这不是对 iPhone 文件系统或 Apple ID 的直接访问。是否能在 iPhone 看到同一份数据，取决于该 iPhone 与当前 Mac 是否使用同一 Apple 账户并启用了对应的 iCloud 同步。

## 术语规则

- “待办”“待办事项”“我有什么待办”默认且只指 Apple「提醒事项」。回答时不混入工作台、项目、会议纪要或历史对话中推断的任务。
- “工作任务”“项目待办”才指工作上下文中的事项。
- Apple 日历安排与 Apple 提醒事项分开返回；没有截止日期的提醒仍是有效待办，不能伪造一个时间。

## 架构

```text
网页「更多工具 → 日程与提醒」
  │  管理员会话 Cookie（只读）
  ▼
Web API (/api/apple-pim/status, /access, /items)
  │  固定字段 JSON
  ▼
ApplePimService (Python)
  │  内容哈希后的受限 Swift helper
  ▼
EventKit / macOS TCC
  ├─ Calendar（只读）
  └─ Reminders（只读；AI 明确指令下可新增一项）
```

模型通过 `sys_skill` 打开 `apple-schedule` 技能：

- `get_apple_schedule_status`：读取权限状态。
- `list_apple_schedule`：读取日历与未完成提醒事项；查询待办时必须仅请求提醒事项。
- `create_apple_reminder`：仅在当前对话有用户直接新增指令时写入一条提醒。用户没提供时间或列表时，工具不会补全时间；列表未指定时使用 Apple 系统默认提醒事项列表。

Swift helper 只接受预定义 action，不使用 AppleScript，不接收任意 shell 命令，也不在项目目录保存 Apple 数据副本。

## 授权与数据隔离

- 首次点击「授权并连接」后，由 macOS 显示日历和提醒事项的完整读取授权弹窗。
- 若此前拒绝，用户需要到「系统设置 → 隐私与安全性 → 日历 / 提醒事项」重新允许运行 Work Agent 的进程。
- Apple 日历是当前 Mac 系统账户范围的数据，不能按 Work Agent 普通成员账号虚假隔离。因此所有 `/api/apple-pim/*` 路由仅限管理员。
- 网页 API 没有创建日历事件或提醒事项的路由；浏览器页面不能手工写入 Apple 数据。

## Web API

所有接口要求登录，且仅限管理员。

### `GET /api/apple-pim/status`

返回当前主机 EventKit 可用性与两类授权状态：`full_access`、`write_only`、`not_determined`、`denied`、`restricted` 或 `unavailable`。

### `POST /api/apple-pim/access`

请求 macOS 授权。至少指定 `events` 或 `reminders` 之一。

```json
{ "events": true, "reminders": true }
```

### `GET /api/apple-pim/items`

参数：

- `start_at`、`end_at`：ISO 8601 时间；最大范围 366 天，只限制日历安排。
- `include_events`、`include_reminders`：设为 `false` 可只读取一类数据。

返回范围内日历事项，以及**全部未完成**的提醒事项（包括无截止日期和逾期项）。每类最多 200 条；长文本字段受限，响应会用 `events_truncated` / `reminders_truncated` 表示已截断。

## 页面交互

1. 页面只读取状态，不会主动触发 macOS 授权弹窗。
2. 用户明确点击授权后，页面展示未来 7、30 或 90 天的日历内容，以及全部未完成提醒事项。
3. 页面不提供输入框、目标列表选择器或任何写入按钮。
4. 如需 AI 新增提醒，用户在对话中直接说明标题；可选说明时间和 Apple 提醒事项列表。AI 仅会依据该明确指令写入，并在结果中确认。

若未看到 iPhone 同步结果，应先检查两台设备的 Apple 账户及 iCloud 中「日历」「提醒事项」开关，而不是重复创建记录。
