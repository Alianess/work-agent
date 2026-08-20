---
name: work-reports
description: Create and maintain Chinese daily work briefs, weekly reports, and biweekly reports from timestamped local Work Agent activity, prior daily reports, generated artifacts, project records, and user-supplied offline work. Use when the user asks for 日报、每日简报、周报、双周报、工作总结、近期工作盘点, wants to fill a missing workday, or needs a periodic report without manually reopening every chat.
---

# Work Reports

Build periodic reports from the account-local work ledger. Treat chat summaries
as supporting context, not as the time source. Use turn/activity timestamps and
successful artifact records to determine what happened during a period.

## Workflow

Period rule: this account files **biweekly reports only**. Daily reports are
internal raw material and are never submitted; there is no weekly report — do
not generate one or use the word 周报. Before writing a biweekly report, read
`references/biweekly-style.md` and follow it exactly: verb-first completed
statements, concrete dates and amounts, 一是/二是 for conclusions, and the
fixed 一、/（一）/1. heading levels. Reject adjectives such as 顺利/扎实/深入 and
summary filler such as 本期重点/工作亮点.

Hard rule for biweekly reports: if `daily_reports` contains a date,
use that saved daily report as the source for that date. The backend deliberately
omits that date from raw `evidence`; do not reopen or reconstruct its chats unless
the user asks to verify one specific assertion. Raw projected turns are only for
dates still missing a daily report.

1. Use `sys_skill(op='show')` and `sys_skill(op='call')` to call
   `collect_work_report_evidence` for the requested `daily` or `biweekly` period. Use explicit start/end dates when the user provides them.
   This remains mandatory when editing, correcting, or supplementing an existing
   report: collect the same period again so the saved report and date-scoped
   account evidence are both available before rewriting it.
2. Read existing daily reports first. Use raw conversation evidence only to fill
   gaps or verify a concrete result; do not rescan every chat manually.
   The evidence tool already projects each turn through the durable recall
   archive: user requests, public implementation-path notes, and final answers
   remain complete, while detailed tool arguments/results are folded to paths
   and compact outcomes. Do not reopen detailed tool logs unless a specific
   assertion cannot be verified otherwise. A saved daily report is evidence,
   not wording that must be copied: rewrite any process-heavy daily wording into
   the audience-facing business result before using it in a weekly or biweekly
   report.
3. Group work by business/project topic, not by chat title or tool name. Merge
   repeated discussion, drafting, revision, validation, and delivery into one
   outcome-oriented item.
4. Distinguish:
   - completed results and delivered artifacts;
   - work in progress and the current blocking point;
   - next actions with known dates or owners;
   - missing offline evidence that requires user input.
5. Never convert an attempt, model plan, failed tool call, or unverified draft
   into a completed result. A successful file edit is evidence that an artifact
   changed, not proof that the underlying business decision was approved.
6. If a workday has no local evidence, ask one concise question naming the date.
   Do not invent an empty day's work. For weekly or biweekly reports, still
   produce a best-effort draft from available days and clearly list the gaps.
7. Draft using the appropriate structure in
   `references/report-writing.md`. A user-provided approved report is the
   highest-priority style reference. Apply any account-local `style_references`
   returned by the evidence tool before the reusable default structure. Unless
   the user asks for a technical activity log, write from the employee or
   department's submission perspective for a manager: report the business work,
   coordination, decision, result, risk, and next step rather than how an agent
   produced the supporting material.
8. Call `save_work_report` through `sys_skill` with the complete Markdown. Its
   successful result includes `verified=true`, a byte count, and a content hash;
   only treat that as a confirmed saved report. Work reports intentionally live
   in the account-local report store, so do **not** pass `content_path` to
   `read_file`. If a second read is genuinely needed, call
   `read_saved_work_report` through this skill instead. Use
   `source_coverage=external_gap` and `needs_user_input=true` when offline work
   is missing; otherwise use `partial` or `full` according to the evidence.
9. When the user explicitly identifies an erroneous saved report for removal, call
   `delete_work_report` with the exact report type and date/range. It removes the
   Markdown and metadata files and returns `verified=true` only after both are
   confirmed absent. Never delete a report based only on a missing-date status or
   an inferred date.
10. Finalize only after `save_work_report` or `delete_work_report` returns `verified=true`.
   Cite the saved report path or deleted paths and mention any dates still requiring
   user input.

## Correction and verification routing

- For any daily-report supplement or correction, use
  `collect_work_report_evidence(report_type='daily', target_date='YYYY-MM-DD')`
  as the primary lookup. It is the authoritative date-indexed, account-level
  route and returns the existing saved report plus projected work evidence.
- Do not use `recall_chat_history(scope='compressed')` to find work from a
  date or another chat. `compressed` searches only summarized-away messages in
  the current conversation, so a miss says nothing about other conversations.
- Use `recall_chat_history` only after the evidence collector when one concrete
  name, number, quotation, correction, or file path still needs verification.
  Use account/project scope appropriate to the cited source; never describe a
  current-chat compressed miss as an account-wide miss.
- Prefer Markdown/text artifacts returned by the evidence collector. Never pass
  `.docx`, `.xlsx`, `.pptx`, `.pdf`, audio, or other binary files to
  `read_file`; open those through their document/media skill only when the
  projected evidence and text companion are insufficient.

## Evidence rules

- Prefer confirmed user corrections over older assistant wording.
- Prefer delivered files, successful edits, signed/paid/completed actions, and
  explicit user confirmations over exploratory discussion.
- Keep exact amounts, dates, organization names, owners, and completion states
  only when supported by reliable evidence.
- Exclude mechanical activity such as opening skills, listing files, retries,
  environment checks, and transport errors unless they materially blocked work.
- Treat AI-assisted production traces as evidence only. Do not report use of an
  agent, model, ASR/OCR, audio chunk counts, Markdown/Word/PDF formats, file
  conversion, directory or manifest operations, OOXML/schema checks, rendering,
  or similar implementation and validation details. Keep a quantity only when
  it measures the business itself (for example meetings held, enterprises
  visited, agreements signed, applications submitted, or equipment delivered),
  not the mechanics of producing the report.
- A document is reportable only by its business purpose and status. Prefer
  `形成会议纪要并明确后续事项` or `完成论证报告并提交审议` when those facts are
  material; never enumerate companion formats, intermediate versions, paragraph
  counts, file paths, or technical validation. If the underlying meeting,
  survey, negotiation, or project推进 is the real work, lead with that work and
  omit the document-production step altogether.
- Do not impose per-message character clipping on the user request, public path
  notes, or final answer. If the complete projected period would threaten the
  context window, let the evidence tool balance whole turns across dates and
  disclose `evidence_truncated_for_context=true`.
- Do not count the same artifact twice when it appears in both a chat activity
  and a turn-runtime record.

## Privacy and portability

Keep all evidence and reports in the current account's local `work_reports`
folder. Do not upload the work ledger. Before reusing this skill in an open
source project, keep organization names, people, internal prices, contracts,
credentials, absolute home paths, and proprietary examples out of the skill
instructions. Put organization-specific phrasing in a user-provided local
reference or settings field instead.

## Workday handling

`check_work_report_status` uses account-local China workday overrides from
`work_reports/calendar_overrides.json`; dates absent from that file fall back
to Monday-Friday. If a requested year lacks official override data, disclose
the fallback instead of claiming statutory holiday accuracy. When exact Chinese
statutory coverage matters, retrieve that year's official State Council holiday
notice, then call `update_workday_calendar` with the source and the holiday /
adjusted-workday dates. Do not infer adjustments from an unofficial calendar.
An override file uses this shape:

```json
{
  "source": "official annual holiday notice URL",
  "days": {
    "2026-01-01": false,
    "2026-01-04": true
  }
}
```
