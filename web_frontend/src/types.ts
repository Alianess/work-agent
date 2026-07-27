export type ModelProfile = {
  name: string;
  provider: string;
  base_url: string;
  model: string;
  api_key_env: string;
  temperature: number;
  max_tokens: number;
  timeout_seconds: number;
  default: boolean;
  api_key_configured: boolean;
};

export type ReasoningEffort = "light" | "medium" | "high" | "very_high";

export type ModelsPayload = {
  default_profile: string;
  env_override: string | null;
  message?: string;
  profiles: ModelProfile[];
};

export type ToolInfo = {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
  provider_id?: string;
  provider_kind?: string;
};

export type ToolsPayload = {
  tools: ToolInfo[];
  providers?: {
    id: string;
    kind: string;
    status?: string;
    config_path?: string;
    servers?: {
      name: string;
      transport: string;
      enabled: boolean;
      status: string;
      tool_count: number;
      description?: string;
      error?: string;
    }[];
    tools: { name: string; owned: boolean }[];
  }[];
  default_max_steps: number;
};

export type AsrProfileOption = {
  name: string;
  label: string;
  default_model_id: string;
};

export type AsrSettingsPayload = {
  profile: "qwen3-asr-mlx-8bit";
  model_id: string;
  backend?: "mlx";
  hotwords: string;
  available_profiles: AsrProfileOption[];
  message?: string;
};

export type AgentSettingsPayload = {
  work_background: string;
  company_document_format: string;
  message?: string;
};

export type CrossChatMemory = {
  id: string;
  conversation_id: string;
  conversation_title: string;
  project_id: string;
  content: string;
  source_summary: string;
  summary_message_count: number;
  conversation_updated_at: number;
  state: "automatic" | "corrected";
  created_at: number;
  updated_at: number;
};

export type CrossChatMemoriesPayload = {
  memories: CrossChatMemory[];
  count: number;
  automatic: boolean;
  source: "conversation_summaries";
  project_id: string | null;
};

export type SkillInfo = {
  id: string;
  label: string;
  mention: string;
  description: string;
  when_to_use: string;
  tool_name?: string;
  outputs?: string[];
  source_url?: string;
  path?: string;
  default_enabled?: boolean;
  enabled: boolean;
};

export type SkillsPayload = {
  skills: SkillInfo[];
};

export type SkillInstructionsPayload = {
  skill_id: string;
  path: string;
  content: string;
  editable: boolean;
  message?: string;
};

export type MeetingMinutesSettingsPayload = {
  default_output_dir: string;
  custom_instructions: string;
  message?: string;
};

export type FileItem = {
  path: string;
  name: string;
  size: number;
  modified: number;
  extension: string;
  mime_type: string;
  kind: "audio" | "image" | "document" | "file";
  previewable: boolean;
};

export type FilesPayload = {
  root: string;
  files: FileItem[];
};

export type TemporarySyncFile = {
  id: string;
  name: string;
  size: number;
  mime_type: string;
  uploaded_at: number;
  expires_at: number;
  download_url: string;
};

export type TemporarySyncPayload = {
  text: {
    content: string;
    updated_at: number | null;
  };
  files: TemporarySyncFile[];
  file_ttl_seconds: number;
  server_time: number;
};

export type ProjectSummary = {
  id: string;
  name: string;
  instructions: string;
  memory_scope: "project_only";
  created_at: number;
  updated_at: number;
  root: string;
  file_count: number;
};

export type Project = ProjectSummary & {
  files: FileItem[];
};

export type ProjectsPayload = {
  projects: ProjectSummary[];
};

export type MeetingArchiveOutput = FileItem & {
  exists: boolean;
};

export type MeetingTime = {
  display: string;
  start?: string;
  end?: string;
  source?: string;
  precision?: string;
  scope?: string;
  recording_start?: string;
  recording_end?: string;
};

export type MeetingArchive = {
  schema_version: number;
  meeting_id: string;
  title: string;
  archive_dir: string;
  manifest_path: string;
  meeting_time?: MeetingTime | null;
  created_at: number;
  updated_at: number;
  source_path: string;
  transcript_path: string;
  outputs: {
    asr?: MeetingArchiveOutput | null;
    internal?: MeetingArchiveOutput | null;
    work_md?: MeetingArchiveOutput | null;
    work_docx?: MeetingArchiveOutput | null;
  };
};

export type MeetingArchivesPayload = {
  root: string;
  meetings: MeetingArchive[];
};

export type FilePayload = {
  path: string;
  name: string;
  content: string;
  truncated: boolean;
  chars: number;
  size: number;
  modified: number;
  extension: string;
  mime_type: string;
  kind: "audio" | "image" | "document" | "file";
  previewable: boolean;
  preview_mode: "markdown" | "text" | "pdf" | "image" | "audio" | "video" | "none";
  preview_url: string;
  source_url: string;
  rendered_path: string;
  editable: boolean;
};

export type AttachmentItem = {
  name: string;
  path: string;
  size: number;
  mime_type: string;
  extension: string;
  kind: "audio" | "image" | "document" | "file";
  deduplicated?: boolean;
  recording_metadata?: {
    recording_started_at?: string;
    recording_started_at_utc?: string;
    recording_started_at_epoch?: number;
    recording_ended_at?: string;
    duration_seconds?: number;
    recording_time_source?: string;
    recording_time_timezone_known?: boolean;
    raw_creation_time?: string;
    media_created_at?: string;
    media_created_at_utc?: string;
    media_creation_time_epoch?: number;
    media_time_source?: string;
    media_time_timezone_known?: boolean;
    file_saved_at?: string;
    recording_time_validation?: string;
  };
};

export type AttachmentPayload = {
  attachment: AttachmentItem;
};

export type SpeechTranscriptionPayload = {
  text: string;
  audio_path: string;
  wav_path: string;
  transcript_path: string;
  engine?: string;
  asr_elapsed_ms?: number;
  filter_chain?: string;
  skipped?: boolean;
  signal?: {
    duration_ms?: number;
    rms?: number;
    max_rms?: number;
    active_ratio?: number;
    has_voice_like_signal?: boolean;
  };
};

export type SpeechVadPayload = {
  available: boolean;
  provider?: string;
  sample_rate: number;
  frame_ms: number;
  speech_frames: boolean[];
  speech_count: number;
  error?: string;
};

export type RealtimeTranscriptSavePayload = {
  ok: boolean;
  title: string;
  path: string;
  segments: number;
  chars: number;
};

export type AgentResult = {
  final: string;
  steps_used: number;
  model_profile: string;
  used_tools?: boolean;
};

export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

export type AgentChatResult = {
  message: ChatMessage;
  steps_used: number;
  model_profile: string;
  used_tools: boolean;
  conversation_id?: string;
  trace_id?: string;
  debug_trace_path?: string;
  selected_skill?: string | null;
  context_summary?: string;
  context_summary_message_count?: number;
  context_compacted?: boolean;
  context_estimated_tokens?: number;
};

export type ChatTitlePayload = {
  title: string;
  model_profile?: string;
};

export type AgentActivityPhase = "thinking" | "action" | "observation" | "complete" | "error";

export type AgentActivityEvent = {
  event: "activity";
  id?: string;
  turn_id?: string;
  event_index?: number;
  phase: AgentActivityPhase;
  title: string;
  detail?: string;
  content?: string;
  activity_type?: "command" | "file_edit";
  command?: string;
  command_status?: "running" | "success" | "error" | "approval_required";
  risk_category?: "READ" | "MODIFY" | "EXECUTE" | "NETWORK" | "DELETE" | "SYSTEM" | string;
  approval_required?: boolean;
  approval_preview?: string;
  approval_resolved?: boolean;
  approval_batch_count?: number;
  approval_batch_remaining?: number;
  approval_batch_commands?: Array<{
    index?: number;
    command?: string;
    cwd?: string;
    timeout_seconds?: number;
  }>;
  file_path?: string;
  additions?: number;
  deletions?: number;
  step?: number;
  tool_name?: string;
  selected_skill?: string | null;
  elapsed_ms?: number;
  trace_id?: string;
  debug_trace_path?: string;
};

export type AgentStreamEvent =
  | {
      event: "turn";
      turn_id: string;
      conversation_id: string;
      turn_status: "queued" | "running" | "waiting_approval" | "succeeded" | "failed" | "cancelled" | string;
      trace_id?: string;
      profile?: string;
      model?: string;
      event_index?: number;
      elapsed_ms?: number;
    }
  | AgentActivityEvent
  | {
      event: "activity_delta";
      id: string;
      turn_id?: string;
      event_index?: number;
      phase: AgentActivityPhase;
      title: string;
      content: string;
      append_mode?: "append" | "replace";
      detail?: string;
      activity_type?: "command" | "file_edit";
      command?: string;
      command_status?: "running" | "success" | "error" | "approval_required";
      risk_category?: "READ" | "MODIFY" | "EXECUTE" | "NETWORK" | "DELETE" | "SYSTEM" | string;
      approval_required?: boolean;
      approval_preview?: string;
      approval_resolved?: boolean;
      approval_batch_count?: number;
      approval_batch_remaining?: number;
      approval_batch_commands?: Array<{
        index?: number;
        command?: string;
        cwd?: string;
        timeout_seconds?: number;
      }>;
      file_path?: string;
      additions?: number;
      deletions?: number;
      step?: number;
      tool_name?: string;
      selected_skill?: string | null;
      elapsed_ms?: number;
      trace_id?: string;
      debug_trace_path?: string;
    }
  | {
      event: "delta";
      content: string;
      turn_id?: string;
      event_index?: number;
      elapsed_ms?: number;
    }
  | {
      event: "draft_delta";
      content: string;
      step?: number;
      turn_id?: string;
      event_index?: number;
      elapsed_ms?: number;
    }
  | {
      event: "draft_reset";
      step?: number;
      turn_id?: string;
      elapsed_ms?: number;
    }
  | {
      event: "final";
      content: string;
      steps_used: number;
      model_profile: string;
      used_tools: boolean;
      conversation_id?: string;
      trace_id?: string;
      debug_trace_path?: string;
      selected_skill?: string | null;
      context_summary?: string;
      context_summary_message_count?: number;
      context_compacted?: boolean;
      context_estimated_tokens?: number;
      elapsed_ms?: number;
      turn_id?: string;
      event_index?: number;
      waiting_approval?: boolean;
    }
  | {
      event: "error";
      message: string;
      type?: string;
      detail?: string;
      trace?: string[];
      elapsed_ms?: number;
      turn_id?: string;
      event_index?: number;
    }
  | {
      event: "cancelled";
      message: string;
      turn_status?: "cancelled" | string;
      elapsed_ms?: number;
      turn_id?: string;
      event_index?: number;
    }
  | {
      event: "done";
    };

export type MeetingResult = {
  archive_dir?: string;
  manifest_path?: string;
  source_path?: string;
  transcript_path?: string;
  asr_transcript_path?: string;
  asr_markdown_path?: string;
  asr_text_path?: string;
  internal_path: string;
  work_path: string;
  work_markdown_path?: string;
  work_docx_path?: string;
  processing_note?: string;
  recording_metadata?: AttachmentItem["recording_metadata"];
};
