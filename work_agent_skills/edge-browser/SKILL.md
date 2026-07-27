---
name: edge-browser
description: Text-first, isolated Edge browser operation through Playwright MCP.
---

# Edge 浏览器操作

这是浏览器操作技能，不是搜索技能。先快照，再按可及性元素引用点击、输入或导航；禁止截图、坐标点击、任意页面脚本和上传。浏览器 profile 按账号和对话隔离。

快照显示 `...[ref=e36]` 时，交互工具的 `target` 使用裸引用 `e36`（不要传入方括号和 `ref=`）。优先使用刚获得快照中的唯一 ref，不要为了绕过 ref 错误改用宽泛 CSS 选择器；页面可能存在同名但隐藏的输入框。若工具已报告点击完成但等待跳转超时，先快照确认当前 URL/标签页，而不是重复点击。

需要真实渲染、登录态或 JavaScript 页面时启用。提交、发布、支付、上传等有后果操作必须先取得用户确认。
