#!/usr/bin/env python3
"""Small Tavily search/extract client; credentials are never printed."""
from __future__ import annotations
import argparse, json, os, sys, urllib.error, urllib.request
BASE_URL = "https://api.tavily.com"
def api_key():
    value = os.environ.get("TAVILY_API_KEY", "").strip()
    if not value: raise RuntimeError("未配置 TAVILY_API_KEY。请写入 Work Agent 根目录 .env。")
    return value
def request(path, *, method="POST", payload=None):
    req = urllib.request.Request(f"{BASE_URL}{path}", data=json.dumps(payload).encode() if payload is not None else None, method=method, headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=45) as response: return json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Tavily 请求失败（HTTP {error.code}）：{error.read().decode(errors='replace')[:500]}") from error
def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    search = sub.add_parser("search"); search.add_argument("query"); search.add_argument("--max-results", type=int, default=5); search.add_argument("--depth", choices=["basic", "advanced"], default="basic"); search.add_argument("--topic", choices=["general", "news", "finance"], default="general")
    extract = sub.add_parser("extract"); extract.add_argument("--urls", required=True); extract.add_argument("--depth", choices=["basic", "advanced"], default="basic")
    sub.add_parser("usage"); args = parser.parse_args()
    try:
        if args.command == "search":
            maximum = max(1, min(args.max_results, 10)); result = request("/search", payload={"query": args.query, "max_results": maximum, "search_depth": args.depth, "topic": args.topic, "include_answer": False, "include_images": False, "include_raw_content": False})
            output = {"query": result.get("query"), "provider": "tavily", "results": [{"title": x.get("title", ""), "url": x.get("url", ""), "snippet": x.get("content", "")} for x in (result.get("results") or [])[:maximum]], "usage": result.get("usage") or {}, "request_id": result.get("request_id", "")}
        elif args.command == "extract":
            urls = json.loads(args.urls)
            if not isinstance(urls, list) or not 1 <= len(urls) <= 5 or not all(isinstance(x, str) and x for x in urls): raise ValueError("urls 必须是包含 1-5 个公开 URL 的 JSON 数组。")
            result = request("/extract", payload={"urls": urls, "extract_depth": args.depth, "format": "markdown", "include_usage": True}); output = {"provider": "tavily", "results": [{"url": x.get("url", ""), "content": x.get("raw_content", "")} for x in result.get("results") or []], "failed_results": result.get("failed_results") or [], "usage": result.get("usage") or {}}
        else: output = request("/usage", method="GET")
        print(json.dumps(output, ensure_ascii=False))
    except Exception as error: print(json.dumps({"error": str(error)}, ensure_ascii=False)); sys.exit(2)
if __name__ == "__main__": main()
