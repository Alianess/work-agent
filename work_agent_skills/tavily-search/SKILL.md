---
name: tavily-search
description: Cost-controlled global web search and extraction through Tavily.
credentials:
  - name: TAVILY_API_KEY
    required: true
---

# 全球网页检索（Tavily）

优先覆盖全球公开网页、海外来源、国际原始资料和英文技术文档，也可以使用中文查询词；选择它的依据是**来源覆盖范围**，不是查询词的语言。

路由规则：国际原始资料、海外媒体、全球公司/产品和英文技术文档优先用本技能；中国大陆政策、中文媒体、国内企业或百度索引结果优先用“百度搜索”。需要同题交叉核验时，可分别调用两个独立技能后再比对来源。

先用 basic 和 3–5 条结果；只有明确需要更高召回或正文提取时才使用 advanced 或 extract。
