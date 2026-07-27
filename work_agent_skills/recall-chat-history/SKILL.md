---
name: recall-chat-history
description: Retrieve manageable cross-chat memories and exact raw passages across locally stored chats using multi-keyword lexical search, without embeddings or another model call. Use when the user refers to something discussed in an earlier chat, a project needs context from its other chats, a compressed long-context summary lacks an exact detail, or the agent needs to verify an earlier name, number, decision, wording, correction, or file path.
---

# Recall Chat History

Use the core `recall_chat_history` tool. Keep retrieval read-only and account-isolated.

## Workflow

1. Extract several distinctive terms from the user's request: names, numbers, product names, file names, or quoted wording.
2. Call `recall_chat_history` with `scope="auto"`: inside a project, search only that project's memories and chats; outside a project, search non-project memories and chats in the current account.
3. Use `scope="current"` to restrict recall to this chat, or `scope="compressed"` to inspect only original messages already represented by its long-context summary.
4. If there is no result, retry once with aliases, synonyms, or fewer but more distinctive keywords.
5. Read `memory_results` first. Prefer entries with `state="corrected"` because the user explicitly corrected them.
6. Use raw `results` to verify exact wording, names, numbers, decisions, or paths. Keep the source conversation visible in the answer when it matters.
7. If both searches miss, say that the local search did not find the passage. Do not claim the user never said it.

## Query guidance

- Good: `query="示例制造 可行性报告 投资金额"`, `keywords=["设备预算", "一期"]`
- Good: `query="用户纠正过的公司名称"`, `keywords=["不是", "应为"]`
- Weak: `query="之前那个事情"`

Synchronize automatic memories from saved conversation summaries. Preserve their source conversation. Do not recreate a user-deleted memory during synchronization, and keep a user correction when the source summary changes. The retrieval path uses deterministic lexical indexing and BM25 ranking. It does not create embeddings or call an LLM. Keep project memories project-only and all memory files isolated by account.
