---
name: meeting-minutes
description: "Use whenever the user wants to turn Chinese meeting audio, noisy recordings, ASR transcripts, dragged local files, supplemental materials, or follow-up corrections into meeting records. This skill owns transcription, fact selection, internal-archive Markdown, and conservative work-submission content; it must hand formal-document classification to official-document when applicable and all Word creation or editing to the full docx skill."
---

# Meeting Minutes

This skill handles Chinese meeting-recording workflows inside the local work agent. It is a workflow skill, not a single black-box tool: read this file, then combine the available lower-level tools.

## Completion gate

For a requested meeting-minutes deliverable, classification, source reading,
drafting decisions, opening downstream skills, viewing tool schemas, and
environment prechecks are preparation only. They are never a completed result.
Do not emit a content-only final response while any required output below has
not been written and verified.

A meeting-minutes task is complete only after all of these conditions hold:

1. A complete ASR Markdown exists and `canonical_outputs.asr` points to it.
   Reuse an existing completed transcript in place; do not duplicate it merely
   to place another copy in the stable meeting archive folder. The internal
   archive Markdown and work-submission Markdown exist in that archive folder.
2. The work-submission Word file exists and has passed the applicable DOCX
   validation and rendered-page visual QA.
3. `manifest.json` exists, has a non-empty `meeting_time.display`, and its four
   `canonical_outputs` paths were read back and confirmed to exist.
4. The final response cites the verified artifact paths and describes only work
   that actually completed.

If any condition is still false, continue with the next native write,
generation, or verification tool call in the same assistant turn. Phrases such
as `会保留`, `将记录`, `按某文种处理`, or `准备生成` do not satisfy this gate.
Never satisfy this gate by feeding an already complete ASR transcript back into
`write_text_file` or `edit_text_file` under a second filename.

## Trigger

Use this skill when the user mentions any of:

- meeting audio, recording, noisy recording, ASR, transcription, denoise, Qwen3-ASR
- 会议纪要, 会议记录, 会议计较, 会议沟通内容整理
- dragged audio or transcript files that should become meeting notes
- refining existing meeting outputs into internal and work-submission versions

Do not say the agent cannot listen to recordings. If the user has not provided a file path, ask them to drag in the audio or provide an ASR transcript path.

## Runtime contract

This skill uses the workspace's single managed `.venv` for ASR, VAD, and any
Python helper. Never create or call a skill-local venv, Conda environment,
system Python, `.venv_agent`, `.venv_project`, or `.venv_deepfilter`. Use
`scripts/runtime_env.sh check` for diagnostics and the approved
`scripts/runtime_env.sh bootstrap` path for dependency repair. The supported
denoise backends are `ffmpeg` and `none`; DeepFilterNet is excluded because its
Python/torch requirements conflict with the unified runtime.

## Inputs

Accept these inputs when available:

- `input_path`: local path to a dragged audio file or existing `.md` / `.txt` transcript
- `meeting_name`: meeting name; if absent, infer from filename only for drafts
- `confirmed_info`: facts the user has explicitly confirmed
- `work_background`: optional user-provided work/project background from web settings or the current request; use it to disambiguate ASR names, organizations, and context, but do not invent facts from it
- `supplemental_paths`: PPT/BP/PDF/image/notes paths used only as reference
- `output_dir`: default `meet_files`

User-confirmed facts override ASR, work background, and materials. Work background helps correct ASR ambiguity and frame the meeting, but it is not proof that a fact was stated in the meeting. Supplemental materials are reference only, not proof that the content was stated in the meeting.

## Tool Workflow

Prefer this composable workflow:

1. If the input is an audio file, first check whether a previous ASR run already exists unless the user explicitly asks to restart from zero. Call `check_meeting_asr_progress` with the complete attachment path as `input_path`. This is a native read-only tool: do not use `shell_exec`, do not ask for terminal approval, and do not split or manually quote paths containing spaces.

   Read the result before deciding the next step. If it shows partial progress such as `22/23`, do **not** discard that work and do **not** say it must start over. Resume by calling `transcribe_meeting_audio` again; the underlying Qwen3 command uses `--skip-existing`, so existing chunk results are reused and only missing chunks are transcribed. If the helper finds a complete transcript, use that transcript directly instead of rerunning ASR.
2. If the input is an audio file and no complete transcript already exists, call `transcribe_meeting_audio`.
   Treat attachment/audio filenames only as file identifiers and possible weak location labels. Do **not** infer the meeting counterpart, attendees, organizer, or cooperation relationship from names such as `客户现场 4.m4a`; wait for the transcript or explicit user confirmation.
   Read the `recording_metadata` returned by the tool. Use `recording_started_at` only when `recording_time_validation` is `plausible_file_timeline`. Some imported M4A files use `creation_time` for export or container creation; when validation reports a conflict, preserve `media_created_at` for audit but do not treat it as a recording or meeting start. Never substitute file upload time, filename prefix, filesystem mtime, or browser `lastModified` for the recording time.
3. If the input is an existing `.md` / `.txt` transcript, read it directly with `read_text_file`.
4. For supplemental `.docx` files, open the `docx` skill and use its extraction capability. For `.pdf` / `.pptx` / `.xlsx` / `.csv`, open the corresponding format skill before extracting a preview.
5. Draft the internal archive as Markdown and write it with `write_text_file`.
6. Draft the work-submission version as conservative Markdown content and write it with `write_text_file`. This is an intermediate artifact, not completion.
7. Perform the modular Word handoff for the canonical deliverable:
   - Open `official-document` when the requested output is a formal `纪要`, carries formal issuing elements, or otherwise clearly involves official-document content. Let it decide between a full statutory document and a company public-document-style text material.
   - Then open `docx` and use the complete Word workflow for creation, editing, comments, tracked changes, validation, and rendered-page QA. Pass the company document-format setting and any template without rewriting them in this skill.
8. Write `manifest.json`, then read it back and verify the four canonical paths, DOCX validation, and rendered-page QA before finalizing.
9. Use `shell_exec` only when a skill instruction or deterministic script is needed; safe read-only commands may run directly, while script/write/install/long commands need user approval.

### Recording time metadata

Treat embedded media `creation_time` as a recording-start candidate. Accept it only when the duration-derived end does not occur after the file was saved; otherwise classify it as export/container time. Even a validated recording start is not automatically the formal meeting start. Apply this priority:

1. User-confirmed meeting time.
2. A clear time stated in the recording/transcript.
3. Embedded recording start time, explicitly labeled as `录音开始时间`.
4. Unknown. Do not fall back to upload time, filesystem timestamps, or filename guesses.

The internal archive should preserve the exact timezone-aware recording start, duration, and derived recording end when available. The work-submission version may safely use the recording date. Use the exact clock time as the meeting time only when it agrees with user-confirmed information or clear transcript evidence; otherwise omit the exact clock time or label it as recording time. For multiple recordings, preserve metadata per segment and order segments by user confirmation or embedded start time only when the timestamps are mutually consistent.

### Multi-recording meetings

When the user provides multiple audio files that appear to be consecutive parts
of one meeting, treat them as one meeting archive after either transcript
evidence or user confirmation. Process each audio file with the checkpoint rule,
then concatenate the transcripts in meeting order into one canonical ASR
Markdown and one canonical ASR `.txt` under:

`meet_files/会议项目/<会议名称>/`

Store the per-recording ASR outputs under a supporting folder such as
`meet_files/会议项目/<会议名称>/asr/segments/`. Do not let segment filenames like
`客户现场 4.m4a`, `5.m4a`, or `6.m4a` become separate meeting titles in the
archive. The manifest `canonical_outputs.asr` must point to the combined ASR
file for the whole meeting, while `supporting_outputs` may point to segment
folders or legacy drafts.

This concatenation rule applies only when multiple recordings need one combined
view. For a single recording with an existing complete `transcript.md`, point
`canonical_outputs.asr` directly to that file and do not create another ASR
copy in the archive folder.

### Public-source lookup and background notes

Before drafting the internal archive, do a small targeted public-source lookup
when the transcript, user confirmation, work background, or supplemental
materials contain confirmed or high-confidence entities such as company names,
schools, parks, government platforms, products, named partners, or stable
industry terms. Prefer AnySearch (`anysearch` skill / available search and
extract tools) when it is available.

Use public lookup to:

- Confirm exact organization names, official websites, locations, public
  business scope, product lines, and public news/background for established
  counterparties or repeatedly mentioned partners.
- Disambiguate ASR homophones only when the public result strongly matches the
  meeting context and/or the user has confirmed the entity.
- Collect adjacent background that helps the user understand the meeting later,
  especially for old-line companies, known local institutions, industrial
  parks, public platforms, and ecosystem partners.

Do not use public lookup to invent meeting facts. Internal numbers,采购规模,
pricing, financing terms, commitments, unpublished cooperation, attendee
comments, and project-stage claims are often not searchable; if they cannot be
verified publicly, mark them as `公开渠道未核验` in the internal archive rather
than forcing a source.

When lookups were run, the internal Markdown should include a section such as
`公开资料核验与背景补充`, listing the query/entity, source title or URL when
available, what the source supports, and what remains unverified. The
work-submission `.docx` should not include search-only speculation or long
background notes; only use externally verified facts when they are conservative
and consistent with the meeting record.

If AnySearch or network lookup is unavailable, continue the meeting-minutes
workflow and note `本轮未完成公开检索` in the internal archive when the missing
lookup matters for names or background.

The legacy one-click HTTP endpoint remains only for backward compatibility and
is not part of this skill's modular tool surface. Do not route new chat work to
it because it bundles content and Word generation.

### ASR resume / checkpoint rule

Before rerunning a long recording after an interruption, always assume useful intermediate files may already exist. Inspect them first with `check_meeting_asr_progress`, passing the complete audio or output path in `input_path`.

The checker reads `chunk_plan.json`, `summary.json`, `progress.jsonl`, and `items/chunk_*/transcript.txt`, then reports completed and missing chunk indexes. Treat that output as the current checkpoint. Typical decisions:

- `complete` or all chunks done: use `transcript.md` / `transcript.txt` directly.
- Partial progress, e.g. `22/23`: rerun `transcribe_meeting_audio`; it will reuse completed chunks via `--skip-existing` and continue the missing chunk(s).
- No run found: start a normal `transcribe_meeting_audio` run.

Never restart a 23-chunk recording from chunk 1 merely because the user message only shows the original audio file. If prior chunks exist, preserving them is mandatory unless the user explicitly requests a clean rerun.

## Required Output Files

For every completed meeting-minutes task, create or update one meeting archive folder:

`meet_files/会议项目/<会议名称>/`

Do not delete intermediate ASR, audio, or previous scattered files. The archive folder is the stable frontend contract for final display.

Always produce these meeting deliverables in that archive folder:

- Internal archive Markdown: named like `会议名称_会议沟通内容整理_内部留档版.md`.
- Work-submission Markdown draft: named like `会议名称_会议纪要_工作提交版.md`.
- Work-submission Word `.docx`, produced by the `docx` skill after the content handoff: named like `MMDD会议主题会议纪要.docx` when the date is known, otherwise `会议主题会议纪要.docx`.

For ASR, preserve and reuse the completed source instead of producing another
copy:

- If a single recording already has a complete `transcript.md`, use that
  existing path as `canonical_outputs.asr`.
- If only `transcript.txt` exists, create one Markdown representation only when
  the frontend needs Markdown; do not create both a public copy and an archive
  copy.
- If multiple recordings must be combined, create one combined canonical ASR
  Markdown in the archive folder as described above.
- Never manually replay a large ASR through repeated `write_text_file` or
  `edit_text_file` calls. Existing complete ASR content is an immutable source,
  not a draft to rewrite.

After writing these files, create or update `manifest.json` in the same folder. Its `canonical_outputs` must point to the four final display files:

`meeting_time` is also mandatory because the meeting archive page uses it for both display and sorting. Never omit it and never substitute upload time, filesystem mtime, or the day the file was generated. Set `display` from user-confirmed meeting time or a clear time stated in the transcript. If only a validated embedded recording time is available, use its date or appropriately labeled recording period and set `source` accordingly. If no defensible meeting or recording time exists, set `display` to `会议时间未确认` and omit `start`/`end`; do not leave the field absent.

```json
{
  "schema_version": 1,
  "meeting_id": "<会议名称>",
  "title": "<会议名称>",
  "archive_dir": "meet_files/会议项目/<会议名称>",
  "meeting_time": {
    "display": "2026年7月16日上午",
    "start": "2026-07-16T10:06:00+08:00",
    "end": "2026-07-16T10:34:09+08:00",
    "source": "user_confirmed | transcript | embedded_recording_time"
  },
  "canonical_outputs": {
    "asr": "meet_files/asr_full/.../transcript.md",
    "internal": "meet_files/会议项目/<会议名称>/<内部留档版>.md",
    "work_md": "meet_files/会议项目/<会议名称>/<工作提交版>.md",
    "work_docx": "meet_files/会议项目/<会议名称>/<工作提交版>.docx"
  },
  "transcript_path": "meet_files/asr_full/.../transcript.txt",
  "supporting_outputs": {
    "asr_text": "meet_files/会议项目/<会议名称>/<ASR转写稿>.txt"
  },
  "recording_metadata": {
    "recording_started_at": "timezone-aware ISO 8601 timestamp when available",
    "recording_ended_at": "derived timestamp when duration is available",
    "duration_seconds": 0,
    "recording_time_source": "embedded_media_creation_time"
  }
}
```

Before reporting completion, read back `manifest.json` and verify that `meeting_time.display` is non-empty, all four `canonical_outputs` paths exist, and the archive API can classify them. The ASR path may point to the existing completed transcript outside the archive folder. A manifest without `meeting_time.display` is incomplete even when all four files were generated successfully.

The meeting archive page reads this manifest. Do not rely on filename guessing for final display.

## Work-submission Content Strategy

Classify the meeting before choosing an outline. The approved human-edited
working-meeting sample is the default structure, including ordinary visits and
counterpart exchanges. A title or folder containing `来访`, `座谈`, or `沟通`
is not evidence that the document should become a four-part company assessment.
Do not force meetings into
`会议基本情况 / 对方单位基本情况 / 双方交流情况 / 初步研判意见`.

- **项目推进、方案讨论、工作协调、一般来访座谈类**：use three to six conclusion-led
  paragraphs. Each paragraph begins with a short Chinese-numbered conclusion
  such as `一、形成资源互补。`, `二、构建标准化实训内容。`,
  `三、探索平台运营模式。`, `四、明确近期工作。` Put the supporting facts in
  the same paragraph. Do not add a generic company-background section or a
  separate `初步研判意见` unless the user explicitly requests a cooperation
  assessment and the meeting content primarily evaluates the counterparty.
- **明确要求的考察或合作对象研判类**：use the four-part assessment structure
  only when the user explicitly asks for an enterprise/cooperation assessment,
  or when the reliable meeting record is predominantly due diligence and contains
  no substantive project decisions, resource commitments, or next actions.
- **政策宣贯、情况汇报、专题学习类**：organize by the actual agenda, policy
  points, requirements, and implementation implications.

For detailed selection rules, content priority, anti-patterns, and two approved
reference patterns, read
`meeting_audio_minutes/skills/meeting-minutes/references/work-submission-writing.md`
before drafting the work-submission version.

## Writing Rules

For the internal archive:

- Preserve useful details for later review.
- Record uncertain names, amounts, model sizes, financing terms, and ASR instability only as uncertain.
- Keep material-vs-meeting-source differences visible.
- Include `公开资料核验与背景补充` when public lookup was performed, and keep
  public-source notes separate from meeting-stated facts.

For the work-submission version:

- Do not expose ASR traces.
- Do not write uncertain names, organizations, places, dates, amounts, valuation, model parameters, TS counts, investment shares, cooperation terms, or technical claims as facts. This is a confidence rule, not a ban on specifics: preserve exact areas, quantities, resources, names, completed actions, owners, deadlines, and next steps when they are user-confirmed, clearly stated in reliable source material, or supplied by an approved human-edited reference.
- **Never put uncertain information into the submitted `.docx`.** If a detail is uncertain, either omit it, generalize it, or write a conservative unit-level phrase. Do not write `待确认`, `疑似`, `可能是`, `约`, `音`, `转写为`, `ASR识别`, `需核实`, parenthesized uncertainty notes, or alternative names in the work-submission version.
- If ASR, background, and supplemental materials conflict, use only user-confirmed wording. Without confirmed wording, remove the conflicting detail from the `.docx`; keep the uncertainty only in the internal archive Markdown.
- Use `work_background` and `confirmed_info` before drafting to lock recurring terms such as company names, project names, industry terms, and participant-side names. If `work_background` is empty, do not guess missing context; write more generally.
- Treat the user's own-side organization from `work_background` / `confirmed_info` as the default perspective for phrases such as `我方` and `平台方`. Do not replace it with ASR-similar names.
- In the submitted `.docx`, describe the user's own side with concise formal perspective wording. Prefer `我方介绍了...`, `我方围绕...进行了说明`, `我方表示...`, or `平台方介绍了...` when referring to actions by the user's side. Avoid stiff third-person side labels unless the sentence specifically needs the full organization name.
- Attribute opinions and investment logic to the actual speaker side. If a strategy such as `投早、投小、长期陪跑` belongs to the user's side, write `我方介绍/提出...投资思路` rather than attributing it to the counterpart. Never move a sentence from one side to another just to make the prose smoother.
- Avoid unexplained industry abbreviations in the submitted `.docx` unless the user has confirmed them. Prefer plain wording such as `家电制造业务` or `冰箱、洗衣机、空调等业务`; do not rely on shorthand such as `白电` when the intended reader may not use that term.
- Treat supplemental-material summaries as secondary. Distinguish meeting facts from a note author's `整理理解`. Do not copy interpretive phrases such as `形成数据记录、报警、报表和后台管理闭环` into the submitted `.docx` unless they are clearly confirmed by transcript or the user; otherwise generalize to `结合具体业务流程进行适配` or omit.
- For the meeting action/location phrase in the submitted `.docx`, follow the reference template's object-based wording. If the counterpart is confirmed, **must** use the confirmed meeting object, not the city: prefer `赴X开展座谈沟通` (reference style) or `与X开展座谈沟通` when `赴` is not appropriate. Examples: `赴示例科技开展座谈沟通`, `与示例合作单位开展座谈沟通`. Do not write `在某地开展座谈沟通` when the counterpart is known. Use city-level wording only when no counterpart or project name is confirmed. Never put ASR-suspected venue names, office names, homophones, or inferred addresses into the submitted `.docx`.
- Match assertion strength to the source. For an observational or noisy-ASR record, use listener-safe wording such as `会议中提到`, `与会人员围绕...进行了交流`, `后续可关注`. For a working meeting with confirmed decisions, preserve action ownership and completion state directly, such as `资料已发送X`, `下一步由X会同Y形成方案`, or `近期赴X调研`; do not dilute agreed actions into `后续可关注`.
- List confirmed units; list people only when confirmed. Do not write `人员待补充`.
- Keep it formal, concise, and workplace-submission safe. It should read like a normal employee's organized meeting note, not an AI transcript analysis.
- Prefer section-level summaries and key points over over-detailed numbered subitems. Remove obvious ASR noise, repeated oral fillers, and uncertain fragments. Prefer conclusion-led headings that state what the meeting established, rather than generic container headings.
- Use `会议基本情况` / `对方单位基本情况` / `双方交流情况` / `初步研判意见` only for assessment-style meetings. For working meetings, prefer action/content headings derived from the actual conclusions.
- Use a final bold `总结建议` only when the document is an assessment and the meeting record supports a recommendation. Do not append it to a working meeting whose final paragraph already records agreed next steps. When needed, acceptable styles include:
  - `**总结建议，可将X纳入后续X方向合作储备范围。**`
  - `**总结建议，可围绕X场景继续保持沟通，结合具体项目需求评估合作可能。**`
  - `**总结建议，可将X作为后续X场景的潜在合作对象，待产品成熟度和落地条件进一步明确后再深化对接。**`
  - `**总结建议，可将X纳入后续场景合作储备范围，待产品成熟度和落地条件进一步明确后再深化对接。**`
  Do not write a long final recommendation paragraph.
- Avoid saying the user will verify, is responsible for, or should request things from other parties.

If the work-submission version lacks enough confirmed facts, ask concise follow-up questions instead of filling gaps.

## Work-submission Word Handoff

Keep document appearance out of this skill. Use the approved samples only to
select content structure, assertion strength, and paragraph roles:

- Use a user-provided, approved meeting-minutes sample as the content and visual reference when one is required.

Hand the final Markdown and the following semantic roles to `official-document`
when applicable and then to `docx`: document title, opening overview,
conclusion-led paragraphs, ordinary body paragraphs, optional final
recommendation, and confirmed attachment/issuer/date fields. The Word skill
must apply the current company document-format setting or the selected official
document specification, preserve all its read/create/edit/comment/redline
capabilities, validate the DOCX, and inspect rendered pages.

Before handoff, verify: every paragraph carries a distinct conclusion; exact
confirmed resources and actions are retained; no generic section duplicates
the opening; no unsupported risk, recommendation, company background, or
formal issuing element was added; and the final paragraph records agreed next
steps when they exist.

## References

Before producing final meeting documents, follow:

- `meeting_audio_minutes/meeting_minutes_spec.md`

Do not treat local user meeting outputs as reusable public examples.

If the user provides a standard `.docx` meeting-minutes sample, first extract its title pattern, metadata layout, heading ladder, paragraph tone, information selection, assertion strength, action ownership, and table usage. Treat that extracted style as higher priority than the generic patterns above.
