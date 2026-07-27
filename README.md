# Work Agent

Work Agent is a local-first AI workbench for file-based office workflows. It combines an OpenAI-compatible model layer, a bounded ReAct agent, optional MCP tools, and reusable skills for tasks such as meeting transcription, document generation, and structured research.

The project is designed to keep source code, credentials, user data, browser state, and generated work artifacts separate. Nothing in `meet_files/`, local configuration, or model caches is part of the public source distribution.

## Included capabilities

- OpenAI-compatible model profiles, selected manually by the user.
- A browser-based local workbench with account isolation.
- ReAct tool execution with activity and trace visibility.
- Local meeting transcription and conservative meeting-minutes generation.
- Word, PDF, PowerPoint, spreadsheet, and extensible Skill workflows.
- Optional MCP stdio integrations, disabled unless configured locally.

## Supported environment

The maintained setup is macOS with Python 3.12, Node.js, and optional FFmpeg/LibreOffice. The service scripts are macOS-oriented; the Python backend and web frontend can also be started manually on other platforms after their dependencies are installed.

## Quick start

1. Create local configuration files. They are intentionally ignored by Git.

   ```bash
   cp .env.example .env
   cp config/model_profiles.example.json config/model_profiles.json
   cp config/agent_settings.example.json config/agent_settings.json
   cp config/asr_settings.example.json config/asr_settings.json
   cp config/mcp_servers.example.json config/mcp_servers.json
   ```

2. Edit `.env` and set a real model API key plus a strong first-admin password. A new installation refuses to create the former insecure default account.

3. Edit `config/model_profiles.json` to match the selected provider and model. Its `api_key_env` must name the environment variable containing the key.

4. Create the managed Python environment and install frontend dependencies.

   ```bash
   scripts/runtime_env.sh bootstrap
   npm --prefix web_frontend install
   npm --prefix web_frontend run build
   ```

5. Start the workbench.

   ```bash
   .venv/bin/python -m work_agent_core.web_server \
     --host 127.0.0.1 \
     --port 8787 \
     --workspace "$PWD" \
     --static-dir web_frontend/dist
   ```

Open `http://127.0.0.1:8787` in a browser. For launchd-based local operation on macOS, use the scripts in `scripts/` only after creating a machine-local launchd plist.

To register the supplied generic template on macOS, run:

```bash
scripts/work_agent_service_ctl.sh install
```

It materializes an ignored `launchd/com.work-agent.plist` with your local
absolute path; do not commit that generated file.

## Configuration and data boundary

| Category | Local path | Commit it? |
| --- | --- | --- |
| API keys and first-admin credentials | `.env` | No |
| Model endpoints and selected model | `config/model_profiles.json` | No |
| Personal work background and document preferences | `meet_files/users/u<id>/agent_settings.json` | No |
| ASR model and hotwords | `config/asr_settings.json` | No |
| Browser/MCP integration | `config/mcp_servers.json` | No |
| Accounts and sessions | `config/auth.sqlite3` | No |
| Meetings, uploads, generated files, memories, and traces | `meet_files/` | No |
| Downloaded models | `meeting_audio_minutes/model_cache/` | No |

The corresponding `*.example.*` files are safe starting points for a fresh installation. Do not copy a real local configuration file into a public issue, commit, or release archive.

## Development checks

```bash
scripts/runtime_env.sh check
.venv/bin/python -m unittest discover -s tests
npm --prefix web_frontend run build
curl -fsS http://127.0.0.1:8787/api/health
```

## Project structure

- `work_agent_core/` — backend, auth, ReAct loop, model routing, MCP and skill gateways.
- `web_frontend/` — React/Vite workbench UI.
- `work_agent_skills/` — repository-local reusable skills.
- `meeting_audio_minutes/` — audio transcription and meeting workflow implementation.
- `config/*.example.*` — safe configuration templates.

## Security and privacy

Read [SECURITY.md](SECURITY.md) before deploying the service beyond a trusted local machine. In particular, do not expose the default HTTP server directly to the public internet, do not commit local data, and treat browser sessions and uploaded meeting materials as sensitive.

## License and third-party components

Original Work Agent code is released under the [MIT License](LICENSE).
Read [NOTICE](NOTICE) before redistributing nested skills or dependencies:
some local skill packs are deliberately excluded because their upstream terms
do not permit redistribution.
