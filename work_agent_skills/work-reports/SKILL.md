---
name: work-reports
description: Create and maintain Chinese daily work briefs, weekly reports, and biweekly reports from timestamped local Work Agent activity, prior daily reports, generated artifacts, project records, and user-supplied offline work. Use when the user asks for 日报、每日简报、周报、双周报、工作总结、近期工作盘点, wants to fill a missing workday, or needs a periodic report without manually reopening every chat.
---

# Work Reports

Build periodic reports from the account-local work ledger. Treat chat summaries
as supporting context, not as the time source. Use turn/activity timestamps and
successful artifact records to determine what happened during a period.

## Workflow

Hard rule for weekly and biweekly reports: if `daily_reports` contains a date,
use that saved daily report as the source for that date. The backend deliberately
omits that date from raw `evidence`; do not reopen or reconstruct its chats unless
the user asks to verify one specific assertion. Raw projected turns are only for
dates still missing a daily report.

1. Use `sys_skill(op='show')` and `sys_skill(op='call')` to call
   `collect_work_report_evidence` for the requested `daily`, `weekly`, or
   `biweekly` period. Use explicit start/end dates when the user provides them.
   This remains mandatory when editing, correcting, or supplementing an existing
   report: collect the same period again so the saved report and date-scoped
   account evidence are both available before rewriting it.
2. Read existing daily reports first. Use raw conversation evidence only to fill
   gaps or verify a concrete result; do not rescan every chat manually.
   The evidence tool already projects each turn through the durable recall
   archive: user requests, public implementation-path notes, and final answers
   remain complete, while detailed tool arguments/results are folded to paths
   and compact outcomes. Do not reopen detailed tool logs unless a specific
   assertion cannot be verified otherwise.
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
   returned by the evidence tool before the reusable default structure.
8. Call `save_work_report` through `sys_skill` with the complete Markdown. Its
   successful result includes `verified=true`, a byte count, and a content hash;
   only treat that as a confirmed saved report. Work reports intentionally live
   in the account-local report store, so do **not** pass `content_path` to
   `read_text_file`. If a second read is genuinely needed, call
   `read_saved_work_report` through this skill instead. Use
   `source_coverage=external_gap` and `needs_user_input=true` when offline work
   is missing; otherwise use `partial` or `full` according to the evidence.
9. Finalize only after `save_work_report` returns `verified=true`. Cite the saved report path
   and mention any dates still requiring user input.

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
  `read_text_file`; open those through their document/media skill only when the
  projected evidence and text companion are insufficient.

## Evidence rules

- Prefer confirmed user corrections over older assistant wording.
- Prefer delivered files, successful edits, signed/paid/completed actions, and
  explicit user confirmations over exploratory discussion.
- Keep exact amounts, dates, organization names, owners, and completion states
  only when supported by reliable evidence.
- Exclude mechanical activity such as opening skills, listing files, retries,
  environment checks, and transport errors unless they materially blocked work.
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
