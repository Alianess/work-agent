# 参与贡献

感谢你愿意改进 Work Agent。这个项目强调本地数据边界、可验证的运行状态和清晰的能力分层；提交代码前，请先确认变更没有把真实工作数据、密钥或本机路径带入仓库。

## 开始之前

1. 先通过 Issue 描述较大的功能或架构改动；小型修复可以直接提交 Pull Request。
2. 不要提交 `.env`、真实 `config/*.json`、`config/auth.sqlite3`、`meet_files/`、下载模型、浏览器状态或生成产物。
3. 新能力应放在正确层级：稳定基础能力进入 Core Tool，领域工作流进入 Skill，独立外部服务优先接 MCP。
4. 涉及账户、文件路径或项目数据时，必须保留现有账户工作区隔离。

## 本地开发

```bash
scripts/runtime_env.sh bootstrap
npm --prefix web_frontend install
npm --prefix web_frontend run build
```

启动本地服务：

```bash
.venv/bin/python -m work_agent_core.web_server \
  --host 127.0.0.1 \
  --port 8787 \
  --workspace "$PWD" \
  --static-dir web_frontend/dist
```

## 提交前检查

```bash
scripts/runtime_env.sh check
.venv/bin/python -m pip check
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
npm --prefix web_frontend run build
git diff --check
```

如果修改了服务启动、模型调用、外部工具或设备相关能力，还应完成对应的真实运行验证；单元测试通过不能代替外部服务或设备验收。

## Pull Request 建议

- 一个 PR 聚焦一个明确问题；
- 说明用户可见的变化、实现边界和验证结果；
- UI 变更附上无敏感信息的截图；
- 新增配置必须同时提供安全的 `*.example.*` 模板；
- 新增 Skill 同时提交 `SKILL.md` 与 `work_agent.json`；
- 不要在日志、截图、测试夹具或 PR 描述中粘贴真实 API Key、Cookie、用户材料和绝对私人路径。

## 安全问题

如果问题可能导致密钥、本地文件、账户数据或浏览器会话泄露，请不要创建公开 Issue。请按 [SECURITY.md](SECURITY.md) 的说明私下报告。
