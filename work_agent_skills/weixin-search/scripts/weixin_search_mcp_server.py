from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any
import sys
import time
from urllib.parse import quote, urljoin, urlparse


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import requests
from fastmcp import FastMCP
from lxml import html as lxml_html

from work_agent_core.weixin_search_adapter import (
    extract_weixin_publish_timestamp,
    extract_weixin_real_url,
    normalize_weixin_publish_time,
)


REQUEST_TIMEOUT_SECONDS = 15
SEARCH_URL = "https://weixin.sogou.com/weixin"
DEFAULT_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36"
    ),
}
HTTP_SESSION = requests.Session()
HTTP_SESSION.headers.update(DEFAULT_HEADERS)

mcp = FastMCP("微信公众号文章搜索")


def _is_antispider(response: requests.Response) -> bool:
    final_url = str(response.url or "").lower()
    body = response.text.lower()
    return "antispider" in final_url or "seccoderight" in body or "anti.min.css" in body


def _resolve_article_url(sogou_url: str) -> dict[str, str]:
    parsed = urlparse(str(sogou_url or ""))
    if parsed.hostname != "weixin.sogou.com" or not parsed.path.startswith("/link"):
        return {"sogou_url": sogou_url, "real_url": "", "status": "invalid_sogou_url"}
    try:
        response = HTTP_SESSION.get(
            sogou_url,
            headers={"Referer": "https://weixin.sogou.com/"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as error:
        return {"sogou_url": sogou_url, "real_url": "", "status": f"request_error: {error}"}
    if _is_antispider(response):
        return {"sogou_url": sogou_url, "real_url": "", "status": "antispider"}
    if urlparse(str(response.url)).hostname == "mp.weixin.qq.com":
        real_url = str(response.url).replace("http://mp.weixin.qq.com", "https://mp.weixin.qq.com", 1)
    else:
        real_url = extract_weixin_real_url(response.text)
    return {
        "sogou_url": sogou_url,
        "real_url": real_url,
        "status": "resolved" if real_url else "not_found",
    }


def _search_page(query: str, page: int, resolve_limit: int) -> list[dict[str, Any]]:
    if not query.strip():
        raise ValueError("query 不能为空")
    if page < 1:
        raise ValueError("page 必须大于等于 1")
    params = {
        "type": "2",
        "s_from": "input",
        "query": query,
        "ie": "utf8",
        "page": page,
        "_sug_": "n",
        "_sug_type_": "",
    }
    headers = {**DEFAULT_HEADERS, "Referer": f"{SEARCH_URL}?query={quote(query)}"}
    response = HTTP_SESSION.get(SEARCH_URL, params=params, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    if _is_antispider(response):
        raise RuntimeError("搜狗微信触发反爬验证，请降低频率后重试或改用公开网页搜索")

    tree = lxml_html.fromstring(response.text)
    elements = tree.xpath("//a[contains(@id, 'sogou_vr_11002601_title_')]")
    results: list[dict[str, Any]] = []
    resolve_limit = max(0, min(int(resolve_limit), 3))
    for index, element in enumerate(elements):
        link = str(element.get("href") or "")
        link = urljoin("https://weixin.sogou.com", link)
        box = element.xpath("ancestor::li[contains(@id, 'sogou_vr_11002601_box_')][1]")
        time_nodes = box[0].xpath(".//div[contains(@class, 's-p')]/span[contains(@class, 's2')]") if box else []
        account_nodes = box[0].xpath(".//div[contains(@class, 's-p')]/a") if box else []
        raw_time = time_nodes[0].text_content().strip() if time_nodes else ""
        timestamp = extract_weixin_publish_timestamp(raw_time)
        resolution = (
            _resolve_article_url(link)
            if index < resolve_limit
            else {"sogou_url": link, "real_url": "", "status": "not_requested"}
        )
        results.append(
            {
                "title": element.text_content().strip(),
                "account": account_nodes[0].text_content().strip() if account_nodes else "",
                "link": link,
                "real_url": resolution["real_url"],
                "real_url_status": resolution["status"],
                "publish_time": normalize_weixin_publish_time(raw_time),
                "publish_timestamp": timestamp,
                "page": page,
            }
        )
        if index + 1 < resolve_limit:
            time.sleep(0.8)
    return results


@mcp.tool
def weixin_search(
    query: Annotated[str, "搜索关键词"],
    page: Annotated[int, "页码，默认1"] = 1,
    resolve_limit: Annotated[int, "最多解析前几条真实链接，范围0-3，默认1"] = 1,
) -> list[dict[str, Any]]:
    """搜索单页微信公众号文章，并规范化发布时间。"""

    return _search_page(query, page, resolve_limit)


@mcp.tool
def weixin_search_all(
    query: Annotated[str, "搜索关键词"],
    max_pages: Annotated[int, "最大页数，默认3，最多5"] = 3,
    resolve_limit: Annotated[int, "每页最多解析前几条真实链接，范围0-3，默认0"] = 0,
) -> list[dict[str, Any]]:
    """低频分页搜索微信公众号文章。"""

    all_results: list[dict[str, Any]] = []
    for page in range(1, max(1, min(int(max_pages), 5)) + 1):
        results = _search_page(query, page, resolve_limit)
        if not results:
            break
        all_results.extend(results)
        if page < max_pages:
            time.sleep(1.5)
    return all_results


@mcp.tool
def resolve_weixin_article_url(
    sogou_url: Annotated[str, "weixin_search 返回的搜狗微信跳转链接"],
) -> dict[str, str]:
    """按需解析一篇文章的 mp.weixin.qq.com 真实链接。"""

    return _resolve_article_url(sogou_url)


@mcp.tool
def get_weixin_article_content(
    real_url: Annotated[str, "微信公众号真实链接；也可传入搜狗微信跳转链接"],
    referer: Annotated[str | None, "请求来源，可传 weixin_search 返回的 link"] = None,
) -> str:
    """读取微信公众号文章正文；传入搜狗链接时先执行一次按需解析。"""

    target_url = real_url
    if urlparse(target_url).hostname == "weixin.sogou.com":
        resolution = _resolve_article_url(target_url)
        target_url = resolution["real_url"]
        referer = referer or real_url
        if not target_url:
            return f"获取文章内容失败: 真实链接解析状态为 {resolution['status']}"
    if urlparse(target_url).hostname != "mp.weixin.qq.com":
        return "获取文章内容失败: 仅支持 mp.weixin.qq.com 文章链接或搜狗微信跳转链接"
    headers = dict(DEFAULT_HEADERS)
    if referer:
        headers["Referer"] = referer
    try:
        response = HTTP_SESSION.get(target_url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as error:
        return f"获取文章内容失败: {error}"
    tree = lxml_html.fromstring(response.text)
    content = [text.strip() for text in tree.xpath("//div[@id='js_content']//text()") if text.strip()]
    return "\n".join(content) if content else "获取文章内容失败: 页面未包含可读正文"


if __name__ == "__main__":
    mcp.run(transport="stdio")
