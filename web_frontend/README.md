# Work Agent Web

React + Vite frontend for the local Work Agent workbench.

## Install

```bash
cd /path/to/work_agent/web_frontend
npm install
```

If your shell proxy makes npm slow, run without proxy for this command only:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy -u WSS_PROXY -u wss_proxy npm install
```

## Build

```bash
npm run build
```

The Python server serves `web_frontend/dist` directly.

## Develop

Start the API/static server:

```bash
cd /path/to/work_agent
python3 -m work_agent_core.web_server --host 127.0.0.1 --port 8787 --workspace "$PWD"
```

For Vite hot reload during UI work:

```bash
cd /path/to/work_agent/web_frontend
npm run dev
```

Vite proxies `/api` to `http://127.0.0.1:8787`.
