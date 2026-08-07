from __future__ import annotations

import unittest

from work_agent_core.weixin_search_adapter import (
    extract_weixin_publish_timestamp,
    extract_weixin_real_url,
    normalize_weixin_publish_time,
)


class WeixinSearchAdapterTests(unittest.TestCase):
    def test_normalizes_sogou_time_convert_markup(self) -> None:
        raw = "document.write(timeConvert('1774337016'))"
        self.assertEqual(extract_weixin_publish_timestamp(raw), 1774337016)
        self.assertEqual(normalize_weixin_publish_time(raw), "2026-03-24 15:23:36")

    def test_rebuilds_all_javascript_url_fragments(self) -> None:
        page = """
        <script>
        var url = '';
        url += 'https://mp.weix';
        url += 'in.qq.com/s?';
        url += '__biz=abc\\x26mid=123';
        </script>
        """
        self.assertEqual(
            extract_weixin_real_url(page),
            "https://mp.weixin.qq.com/s?__biz=abc&mid=123",
        )

    def test_rejects_non_wechat_redirect_targets(self) -> None:
        self.assertEqual(extract_weixin_real_url("url += 'https://example.com/a';"), "")

    def test_preserves_timestamp_query_parameter(self) -> None:
        page = "url += 'https://mp.weixin.qq.com/s?src=11&timestamp=1774337016&amp;new=1';"
        self.assertEqual(
            extract_weixin_real_url(page),
            "https://mp.weixin.qq.com/s?src=11&timestamp=1774337016&new=1",
        )


if __name__ == "__main__":
    unittest.main()
