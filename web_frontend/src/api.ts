import type {
  AgentResult,
  AgentChatResult,
  AgentSettingsPayload,
  ApplePimItemsPayload,
  ApplePimStatus,
  AgentStreamEvent,
  AsrSettingsPayload,
  AttachmentPayload,
  ChatMessage,
  ChatTitlePayload,
  CrossChatMemoriesPayload,
  CrossChatMemory,
  FilePayload,
  FileItem,
  FilesPayload,
  FridayNotificationsPayload,
  MeetingArchivesPayload,
  MeetingResult,
  MeetingMinutesSettingsPayload,
  ModelsPayload,
  OfficePdfInput,
  OfficePdfMergePayload,
  Project,
  ProjectSummary,
  ProjectsPayload,
  ReasoningEffort,
  RealtimeTranscriptSavePayload,
  RealtimeTranscriptSessionPayload,
  SkillsPayload,
  SkillInstructionsPayload,
  SpeechVadPayload,
  SpeechTranscriptionPayload,
  TemporarySyncFile,
  TemporarySyncPayload,
  ToolsPayload,
  WeixinChannelStatus,
  WeixinLoginState,
  WorkCalendarPayload,
  WorkDayDetailPayload
} from "./types";

export type AuthUser = {
  id: number;
  username: string;
  role: "admin" | "member";
  created_at: number;
};

export type AuthPayload = {
  authenticated: boolean;
  user: AuthUser | null;
};

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {})
    }
  });
  const raw = await response.text();
  if (!raw.trim()) {
    throw new Error(
      response.ok
        ? "后端没有返回内容，请确认本地后端服务仍在运行。"
        : `请求失败且后端返回为空，HTTP 状态码 ${response.status}`
    );
  }
  let data: T & { error?: string };
  try {
    data = JSON.parse(raw) as T & { error?: string };
  } catch (error) {
    throw new Error(
      `后端返回的不是有效 JSON，HTTP 状态码 ${response.status}：${raw.slice(0, 160)}`
    );
  }
  if (!response.ok) {
    throw new Error(data.error || `请求失败，HTTP 状态码 ${response.status}`);
  }
  return data;
}

async function uploadFile<T>(path: string, file: File): Promise<T> {
  const query = new URLSearchParams({
    name: file.name,
    mime_type: file.type || "application/octet-stream",
    last_modified: String(file.lastModified || "")
  });
  return requestJson<T>(`${path}?${query.toString()}`, {
    method: "POST",
    // Keep the file out of JSON/base64 so public proxies see the real size.
    body: file,
    headers: { "Content-Type": file.type || "application/octet-stream" }
  });
}

export const api = {
  authMe: () => requestJson<AuthPayload>("/api/auth/me"),
  login: (username: string, password: string) =>
    requestJson<AuthPayload>("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ username, password })
    }),
  register: (username: string, password: string, inviteCode: string) =>
    requestJson<AuthPayload>("/api/auth/register", {
      method: "POST",
      body: JSON.stringify({ username, password, invite_code: inviteCode })
    }),
  logout: () =>
    requestJson<{ ok: boolean }>("/api/auth/logout", {
      method: "POST",
      body: JSON.stringify({})
    }),
  changePassword: (currentPassword: string, newPassword: string) =>
    requestJson<{ ok: boolean; message: string }>("/api/auth/password", {
      method: "POST",
      body: JSON.stringify({ current_password: currentPassword, new_password: newPassword })
    }),
  health: () => requestJson<{ ok: boolean; auth_enabled?: boolean; workspace: string; config: string }>("/api/health"),
  models: () => requestJson<ModelsPayload>("/api/models"),
  asrSettings: () => requestJson<AsrSettingsPayload>("/api/settings/asr"),
  agentSettings: () => requestJson<AgentSettingsPayload>("/api/settings/agent"),
  weixinStatus: () => requestJson<WeixinChannelStatus>("/api/channels/weixin"),
  startWeixinLogin: (force = false) =>
    requestJson<WeixinLoginState>("/api/channels/weixin/login/start", {
      method: "POST",
      body: JSON.stringify({ force })
    }),
  pollWeixinLogin: (sessionId: string, verifyCode = "") =>
    requestJson<
      | (WeixinLoginState & { connected: false; needs_verify_code?: boolean })
      | {
          connected: true;
          status: "connected";
          account_id: string;
          user_id: string;
          message: string;
        }
    >("/api/channels/weixin/login/poll", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, verify_code: verifyCode })
    }),
  disconnectWeixin: () =>
    requestJson<{ ok: boolean; connected: false; message: string }>("/api/channels/weixin/disconnect", {
      method: "POST",
      body: JSON.stringify({})
    }),
  notifications: () => requestJson<FridayNotificationsPayload>("/api/notifications"),
  applePimStatus: () => requestJson<ApplePimStatus>("/api/apple-pim/status"),
  applePimAccess: (events: boolean, reminders: boolean) =>
    requestJson<{ ok: boolean; events_granted?: boolean | null; reminders_granted?: boolean | null; status: ApplePimStatus }>(
      "/api/apple-pim/access",
      { method: "POST", body: JSON.stringify({ events, reminders }) }
    ),
  applePimItems: (params?: { start_at?: string; end_at?: string; include_events?: boolean; include_reminders?: boolean }) => {
    const query = new URLSearchParams();
    if (params?.start_at) query.set("start_at", params.start_at);
    if (params?.end_at) query.set("end_at", params.end_at);
    if (params?.include_events === false) query.set("include_events", "false");
    if (params?.include_reminders === false) query.set("include_reminders", "false");
    return requestJson<ApplePimItemsPayload>(`/api/apple-pim/items${query.size ? `?${query}` : ""}`);
  },
  workReportCalendar: (year: number, month: number) =>
    requestJson<WorkCalendarPayload>(`/api/work-reports/calendar?year=${year}&month=${month}`),
  workReportDay: (date: string) =>
    requestJson<WorkDayDetailPayload>(`/api/work-reports/day?date=${encodeURIComponent(date)}`),
  markNotificationsRead: (id?: string) =>
    requestJson<FridayNotificationsPayload>("/api/notifications/read", {
      method: "POST",
      body: JSON.stringify(id ? { id } : { all: true })
    }),
  deleteNotification: (id: string) =>
    requestJson<FridayNotificationsPayload>("/api/notifications/delete", {
      method: "POST",
      body: JSON.stringify({ id })
    }),
  meetingMinutesSettings: () => requestJson<MeetingMinutesSettingsPayload>("/api/settings/meeting-minutes"),
  tools: () => requestJson<ToolsPayload>("/api/tools"),
  debugTraces: (params?: { conversation_id?: string; trace_id?: string; limit?: number }) => {
    const query = new URLSearchParams();
    if (params?.conversation_id) query.set("conversation_id", params.conversation_id);
    if (params?.trace_id) query.set("trace_id", params.trace_id);
    if (params?.limit) query.set("limit", String(params.limit));
    return requestJson<{ ok: boolean; enabled: boolean; path: string; events: unknown[]; traces: unknown[] }>(
      `/api/debug/traces${query.toString() ? `?${query.toString()}` : ""}`
    );
  },
  agentTurn: (turnId: string) =>
    requestJson<{ ok: boolean; turn: unknown }>(`/api/agent/turns/${encodeURIComponent(turnId)}`),
  agentTurnEvents: (turnId: string, after = -1) =>
    requestJson<{
      ok: boolean;
      turn_id: string;
      conversation_id: string;
      status: string;
      cancel_requested: boolean;
      latest_event_index: number;
      events: AgentStreamEvent[];
    }>(`/api/agent/turns/${encodeURIComponent(turnId)}/events?after=${encodeURIComponent(String(after))}`),
  cancelAgentTurn: (turnId: string) =>
    requestJson<{ ok: boolean; turn_id: string; status: string; cancel_requested: boolean }>(
      `/api/agent/turns/${encodeURIComponent(turnId)}/cancel`,
      { method: "POST", body: JSON.stringify({}) }
    ),
  approveAgentTurn: (
    turnId: string,
    payload: { conversation_id?: string },
    onEvent: (event: AgentStreamEvent) => void,
    options?: { signal?: AbortSignal }
  ) =>
    streamJsonEvents(
      `/api/agent/turns/${encodeURIComponent(turnId)}/approve`,
      payload,
      onEvent,
      options
    ),
  skills: () => requestJson<SkillsPayload>("/api/skills"),
  skillInstructions: (skillId: string) =>
    requestJson<SkillInstructionsPayload>(`/api/skills/${encodeURIComponent(skillId)}/instructions`),
  saveSkillInstructions: (skillId: string, content: string) =>
    requestJson<SkillInstructionsPayload>(`/api/skills/${encodeURIComponent(skillId)}/instructions`, {
      method: "POST",
      body: JSON.stringify({ content })
    }),
  setSkillEnabled: (skillId: string, enabled: boolean) =>
    requestJson<SkillsPayload>("/api/skills/settings", {
      method: "POST",
      body: JSON.stringify({ skill_id: skillId, enabled })
    }),
  saveMeetingMinutesSettings: (payload: {
    default_output_dir: string;
    custom_instructions: string;
  }) =>
    requestJson<MeetingMinutesSettingsPayload>("/api/settings/meeting-minutes", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  useModel: (name: string) =>
    requestJson<ModelsPayload>("/api/models/use", {
      method: "POST",
      body: JSON.stringify({ name })
    }),
  saveModelKey: (name: string, apiKey: string) =>
    requestJson<ModelsPayload>("/api/models/key", {
      method: "POST",
      body: JSON.stringify({ name, api_key: apiKey })
    }),
  addModel: (payload: Record<string, unknown>) =>
    requestJson<ModelsPayload>("/api/models/add", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  updateModel: (payload: Record<string, unknown>) =>
    requestJson<ModelsPayload>("/api/models/update", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  deleteModel: (name: string) =>
    requestJson<ModelsPayload>("/api/models/delete", {
      method: "POST",
      body: JSON.stringify({ name })
    }),
  testModel: (payload: Record<string, unknown>) =>
    requestJson<{
      ok: boolean;
      status: number;
      latency_ms: number;
      endpoint: string;
      model: string;
      message: string;
    }>("/api/models/test", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  discoverModels: (payload: Record<string, unknown>) =>
    requestJson<{
      ok: boolean;
      models: string[];
      count: number;
      endpoint: string;
      latency_ms: number;
      message: string;
    }>("/api/models/discover", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  saveAsrSettings: (payload: {
    profile: string;
    model_id: string;
    hotwords: string;
  }) =>
    requestJson<AsrSettingsPayload>("/api/settings/asr", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  saveAgentSettings: (payload: AgentSettingsPayload) =>
    requestJson<AgentSettingsPayload>("/api/settings/agent", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  memories: (params?: { project_id?: string; query?: string }) => {
    const query = new URLSearchParams();
    if (params && Object.prototype.hasOwnProperty.call(params, "project_id")) {
      query.set("project_id", params.project_id ?? "");
    }
    if (params?.query) query.set("query", params.query);
    return requestJson<CrossChatMemoriesPayload>(
      `/api/memories${query.toString() ? `?${query.toString()}` : ""}`
    );
  },
  updateMemory: (id: string, content: string) =>
    requestJson<{ ok: boolean; memory: CrossChatMemory; message: string }>("/api/memories/update", {
      method: "POST",
      body: JSON.stringify({ id, content })
    }),
  deleteMemory: (id: string) =>
    requestJson<{ ok: boolean; message: string }>("/api/memories/delete", {
      method: "POST",
      body: JSON.stringify({ id })
    }),
  runAgent: (payload: { goal: string; profile: string; max_steps: number }) =>
    requestJson<{ result: AgentResult }>("/api/agent/run", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  chatAgent: (payload: {
    conversation_id?: string;
    project_id?: string;
    messages: ChatMessage[];
    profile: string;
    reasoning_effort: ReasoningEffort;
    auto_approve?: boolean;
    skill_hint?: string | null;
    conversation_summary?: string | null;
    conversation_summary_message_count?: number;
    context_file_paths?: string[];
    rewind_user_message_ordinal?: number;
  }) =>
    requestJson<AgentChatResult>("/api/agent/chat", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  streamChatAgent: (
    payload: {
      conversation_id?: string;
      project_id?: string;
      messages: ChatMessage[];
      profile: string;
      reasoning_effort: ReasoningEffort;
      auto_approve?: boolean;
      skill_hint?: string | null;
      conversation_summary?: string | null;
      conversation_summary_message_count?: number;
      context_file_paths?: string[];
      rewind_user_message_ordinal?: number;
    },
    onEvent: (event: AgentStreamEvent) => void,
    options?: { signal?: AbortSignal }
  ) => streamJsonEvents("/api/agent/chat-stream", payload, onEvent, options),
  generateChatTitle: (payload: { messages: ChatMessage[]; profile: string }) =>
    requestJson<ChatTitlePayload>("/api/agent/title", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  conversations: () => requestJson<{ items: unknown[]; revision: number }>("/api/conversations"),
  conversationFiles: (conversationId: string) =>
    requestJson<{ conversation_id: string; title: string; files: FileItem[] }>(
      `/api/conversations/${encodeURIComponent(conversationId)}/files`
    ),
  saveConversations: (payload: {
    base_revision?: number;
    upserts?: unknown[];
    deleted_ids?: string[];
    items?: unknown[];
  }) =>
    requestJson<{
      ok: boolean;
      conflict?: boolean;
      path?: string;
      count: number;
      revision: number;
      items?: unknown[];
    }>("/api/conversations/save", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  moveConversationToProject: (conversationId: string, projectId: string | null) =>
    requestJson<{
      ok: boolean;
      conversation_id: string;
      project_id: string | null;
      project: ProjectSummary | null;
      files: FileItem[];
      copied_count: number;
      unchanged_count: number;
    }>("/api/conversations/move-project", {
      method: "POST",
      body: JSON.stringify({ conversation_id: conversationId, project_id: projectId })
    }),
  projects: () => requestJson<ProjectsPayload>("/api/projects"),
  project: (projectId: string) =>
    requestJson<{ project: Project }>(`/api/projects/${encodeURIComponent(projectId)}`),
  createProject: (payload: { name: string; instructions?: string }) =>
    requestJson<{ project: Project }>("/api/projects/create", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  updateProject: (projectId: string, payload: { name: string; instructions: string }) =>
    requestJson<{ project: Project }>(
      `/api/projects/${encodeURIComponent(projectId)}/settings`,
      { method: "POST", body: JSON.stringify(payload) }
    ),
  addProjectFile: (
    projectId: string,
    payload: { name: string; mime_type: string; content_base64: string }
  ) =>
    requestJson<{ project: Project; file: unknown; attachment: AttachmentPayload["attachment"] }>(
      `/api/projects/${encodeURIComponent(projectId)}/files/add`,
      { method: "POST", body: JSON.stringify(payload) }
    ),
  uploadProjectFile: (projectId: string, file: File) =>
    uploadFile<{ project: Project; file: unknown; attachment: AttachmentPayload["attachment"] }>(
      `/api/projects/${encodeURIComponent(projectId)}/files/upload`,
      file
    ),
  deleteProjectFile: (projectId: string, path: string) =>
    requestJson<{ ok: boolean; project: Project }>(
      `/api/projects/${encodeURIComponent(projectId)}/files/delete`,
      { method: "POST", body: JSON.stringify({ path }) }
    ),
  syncMeetingToProject: (projectId: string, manifestPath: string) =>
    requestJson<{
      ok: boolean;
      meeting_title: string;
      copied_count: number;
      unchanged_count: number;
      project: Project;
    }>(`/api/projects/${encodeURIComponent(projectId)}/sync-meeting`, {
      method: "POST",
      body: JSON.stringify({ manifest_path: manifestPath })
    }),
  createProjectTimeline: (projectId: string) =>
    requestJson<{ ok: boolean; project: Project }>(
      `/api/projects/${encodeURIComponent(projectId)}/timeline/create`,
      { method: "POST", body: JSON.stringify({}) }
    ),
  selectProjectTimeline: (projectId: string, path: string) =>
    requestJson<{ ok: boolean; project: Project }>(
      `/api/projects/${encodeURIComponent(projectId)}/timeline/select`,
      { method: "POST", body: JSON.stringify({ path }) }
    ),
  updateProjectTimeline: (
    projectId: string,
    changes: Array<{
      action: "add" | "update" | "delete";
      match?: Record<string, unknown>;
      values?: Record<string, unknown>;
      delete_mode?: "soft" | "row";
    }>
  ) =>
    requestJson<{ ok: boolean; project: Project; result: unknown }>(
      `/api/projects/${encodeURIComponent(projectId)}/timeline/changes`,
      {
        method: "POST",
        body: JSON.stringify({ changes, change_source: "项目页面" })
      }
    ),
  generateMinutes: (payload: {
    transcript_path: string;
    output_dir: string;
    meeting_name: string;
    confirmed_info: string;
    supplemental_paths: string[];
    profile: string;
  }) =>
    requestJson<{ result: MeetingResult }>("/api/skills/meeting-minutes", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  addAttachment: (payload: {
    name: string;
    mime_type: string;
    content_base64: string;
    size?: number;
    last_modified?: number;
    relative_path?: string;
    source_path?: string;
  }) =>
    requestJson<AttachmentPayload>("/api/attachments/add", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  uploadAttachment: (file: File) => uploadFile<AttachmentPayload>("/api/attachments/upload", file),
  addOfficePdf: (payload: { name: string; mime_type: string; content_base64: string }) =>
    requestJson<{ ok: boolean; input: OfficePdfInput; message: string }>("/api/office-workspace/pdf/add", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  mergeOfficePdfs: (payload: { source_paths: string[]; output_name: string }) =>
    requestJson<OfficePdfMergePayload>("/api/office-workspace/pdf/merge", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  temporarySync: () => requestJson<TemporarySyncPayload>("/api/temporary-sync"),
  saveTemporarySyncText: (content: string) =>
    requestJson<{
      text: TemporarySyncPayload["text"];
      message: string;
    }>("/api/temporary-sync/text", {
      method: "POST",
      body: JSON.stringify({ content })
    }),
  addTemporarySyncFile: (payload: {
    name: string;
    mime_type: string;
    content_base64: string;
  }) =>
    requestJson<{ file: TemporarySyncFile; message: string }>("/api/temporary-sync/files/add", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  deleteTemporarySyncFile: (id: string) =>
    requestJson<{ ok: boolean; deleted: boolean }>("/api/temporary-sync/files/delete", {
      method: "POST",
      body: JSON.stringify({ id })
    }),
  transcribeSpeech: (payload: {
    name: string;
    mime_type: string;
    content_base64: string;
    use_denoise?: boolean;
    skip_if_silent?: boolean;
    realtime_session_id?: string;
    segment_index?: number;
    started_at?: number;
    finished_at?: number;
    title?: string;
  }) =>
    requestJson<SpeechTranscriptionPayload>("/api/speech/transcribe", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  detectSpeechVad: (payload: {
    sample_rate: number;
    frame_ms: number;
    aggressiveness?: number;
    frames_base64: string[];
  }) =>
    requestJson<SpeechVadPayload>("/api/speech/vad", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  saveRealtimeTranscript: (payload: {
    title: string;
    session_id: string;
    segments: Array<{
      index: number;
      text: string;
      started_at: number;
      finished_at: number;
      audio_path?: string;
      transcript_path?: string;
      engine?: string;
      asr_elapsed_ms?: number;
    }>;
  }) =>
    requestJson<RealtimeTranscriptSavePayload>("/api/realtime-transcript/save", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  realtimeTranscriptSession: (sessionId: string) =>
    requestJson<RealtimeTranscriptSessionPayload>(
      `/api/realtime-transcript/session?session_id=${encodeURIComponent(sessionId)}`
    ),
  retryRealtimeTranscriptSegment: (sessionId: string, segmentIndex: number) =>
    requestJson<SpeechTranscriptionPayload>("/api/realtime-transcript/retry", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, segment_index: segmentIndex })
    }),
  files: (path: string, limit = 500) =>
    requestJson<FilesPayload>(
      `/api/files?path=${encodeURIComponent(path)}&limit=${encodeURIComponent(String(limit))}`
    ),
  meetingArchives: () => requestJson<MeetingArchivesPayload>("/api/meeting-archives"),
  file: (path: string) =>
    requestJson<FilePayload>(`/api/file?path=${encodeURIComponent(path)}&max_chars=50000`),
  openLocalFile: (path: string) =>
    requestJson<{ ok: boolean; path: string; action: string }>("/api/file/open", {
      method: "POST",
      body: JSON.stringify({ path })
    }),
  revealLocalFile: (path: string) =>
    requestJson<{ ok: boolean; path: string; action: string }>("/api/file/reveal", {
      method: "POST",
      body: JSON.stringify({ path })
    })
};

async function streamJsonEvents(
  path: string,
  payload: Record<string, unknown>,
  onEvent: (event: AgentStreamEvent) => void,
  options?: { signal?: AbortSignal }
) {
  const response = await fetch(path, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    signal: options?.signal,
    body: JSON.stringify(payload)
  });
  if (!response.ok || !response.body) {
    let message = `请求失败，HTTP 状态码 ${response.status}`;
    try {
      const data = (await response.json()) as { error?: string };
      message = data.error || message;
    } catch {
      // keep the HTTP fallback message
    }
    throw new Error(message);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let terminalEventReceived = false;
  const handleEvent = (event: AgentStreamEvent | null) => {
    if (!event) return;
    if (event.event === "final" || event.event === "error" || event.event === "cancelled") {
      terminalEventReceived = true;
    }
    onEvent(event);
  };
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";
    for (const part of parts) {
      handleEvent(parseSseBlock(part));
    }
  }
  buffer += decoder.decode();
  handleEvent(parseSseBlock(buffer));
  if (!terminalEventReceived && !options?.signal?.aborted) {
    throw new Error(
      "连接意外中断：后端没有返回完成、失败或停止状态。已执行的工具结果可能已经保留，请点击“继续”接着处理，不要重复启动整项任务。"
    );
  }
}

function parseSseBlock(block: string): AgentStreamEvent | null {
  const data = block
    .split("\n")
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trimStart())
    .join("\n")
    .trim();
  if (!data) return null;
  return JSON.parse(data) as AgentStreamEvent;
}
