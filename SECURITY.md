# Security policy

## Before publishing

This repository is intended to contain source code and safe examples only. Never publish:

- `.env` files, API keys, access tokens, or cookies;
- `config/*.json` local configuration or `config/auth.sqlite3`;
- `meet_files/` user workspaces, audio, uploads, transcripts, generated documents, memories, traces, or browser artifacts;
- model caches, virtual environments, `node_modules/`, temporary files, or local launchd plists;
- customer names, meeting participants, recordings, or internal research material embedded in examples.
- local skill packs whose upstream license prohibits redistribution (see [NOTICE](NOTICE)).

Run `git status --ignored` and review the staged diff before every push. A secret scanner should be enabled in the eventual remote repository.

## Deployment guidance

- Set `WORK_AGENT_ADMIN_USERNAME` and a unique `WORK_AGENT_ADMIN_PASSWORD` before the first startup.
- Keep the service bound to `127.0.0.1` unless it is placed behind an authenticated, TLS-terminating reverse proxy.
- Use a dedicated local model key with minimal scope where the provider supports it.
- Browser integrations can contain authenticated website sessions. Keep their profile/state per user and outside the repository.
- Back up private application data separately from source control.

## Reporting a vulnerability

Do not open a public issue with secrets or exploit details. Until a public reporting channel is established, contact the maintainer privately through the repository owner's preferred channel.
