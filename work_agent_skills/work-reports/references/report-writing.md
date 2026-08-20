# Periodic report writing

## Daily brief

Keep a daily brief compact and factual:

```markdown
# YYYY-MM-DD 工作简报

## 今日完成
1. **项目或工作线：** 已完成的动作、形成的结果及关键产物。

## 推进中
1. **项目或工作线：** 当前进展、待闭合事项和必要约束。

## 下一步
1. 明确的下一动作、责任主体或时间节点。

## 待补充
- 仅列本地记录无法覆盖的外部工作；没有缺口时删除本节。
```

Do not turn every chat into a bullet. Merge several chats that supported the
same deliverable.

All report types are workplace-facing by default. The evidence may describe how
an agent processed audio, generated files, corrected formatting, or validated a
document; the report must translate that evidence into the employee's business
action and result. Do not make AI assistance itself the subject of a work item.

## Weekly report

Use the user's reporting week when specified. Otherwise use the explicit date
range returned by the evidence tool. Prefer:

1. 本周重点工作完成情况：grouped by business line/project.
2. 重点成果与交付物：only material results.
3. 推进中事项及风险：state the actual gap, not generic difficulty.
4. 下周工作计划：continue directly from unfinished work and known deadlines.

Do not repeat daily reports chronologically unless the user explicitly asks for
a day-by-day log.

## Biweekly report

Match the common departmental structure:

1. 上两周工作总结
   - organize by the department's stable business headings;
   - use numbered outcome statements;
   - retain important dates, amounts, counterparties, approvals, payments,
     signed agreements, completed materials, and delivered artifacts when
     confirmed.
2. 下两周工作计划
   - convert open loops into concrete next actions;
   - do not promise completion when the evidence supports only推进、协调、编制,
     or内部论证;
   - avoid empty numbering and placeholder items.

When an approved sample is supplied, extract its heading hierarchy, sentence
length, verb strength, detail density, owner/date placement, and treatment of
unfinished work. Apply that style without copying private names or facts into
the reusable skill.

The approved departmental sample provided in July 2026 establishes this local
pattern: a short department/date header; `第一部分：上两周工作总结`; stable
Chinese hierarchical headings (`一、` → `（一）` → `1.`); numbered outcome
sentences with dates, amounts and counterparties retained when verified; then
`第二部分：下两周工作计划` using the same business headings. Keep this pattern
for future biweekly drafts, remove empty numbering, and do not transplant any
organization-specific facts from the sample into another report.

## Manager-facing content filter

Before saving a weekly or biweekly report, apply this filter to every item:

1. Identify the underlying business event: meeting, visit, negotiation,
   coordination, research, application, contract, project delivery, operation,
   or decision support.
2. State what was actually advanced or clarified, the confirmed stage result,
   and the next owner or milestone when useful.
3. Remove the production trail used to prepare the evidence: agent/model use,
   ASR/OCR, audio duration or chunk completion, Markdown/Word/PDF, internal and
   submission editions, file generation or conversion, archive/manifest work,
   paragraph counts, OOXML/schema checks, rendering, and local paths.
4. Retain a named report, plan, agreement, application, or meeting record only
   when its preparation or submission is itself a material business outcome;
   describe its purpose and approval/submission status, not its file format.
5. Read the sentence as if it were submitted directly to a department leader.
   If it mainly proves that the assistant worked hard, rewrite or remove it.

Example transformation:

- Do not write: `完成杭州弘翌团队来访会议纪要两版 Markdown、Word 及归档清单，录音转写共 51/51 个分块完成。`
- Write: `8月6日，接待杭州弘翌团队赴训练场开展技术对接与现场勘测，围绕场地功能分区、设备安置、水电改造及实训动线进行逐项沟通；明确后续由对方提供课程体系、装修物料和教辅办公设备清单，并持续推进产教融合运营合作方案。`

The same rule applies to plan items. Do not plan to repair document XML,
re-render pages, maintain reports, or verify file packaging unless the user
explicitly requests an internal technical activity log. State the business
closure instead, such as `完善调研报告并提交审议`.

## Compression and assertion strength

- `讨论、研究、对接` alone are weak. State what scope, option, material, or
  next step was clarified.
- `起草、编制、修订` are deliverables only when a file or explicit confirmation
  exists.
- `完成、办结、签订、支付、交付` require direct evidence.
- Plans must begin with actionable verbs such as `完成、推动、完善、组织、
  协调、形成、提交`, calibrated to the actual certainty.
- Omit raw prompts, internal chain-of-thought, tool names, model names, AI/ASR/OCR
  process, file formats, retries, validation mechanics, and local implementation
  details from workplace-facing reports.
