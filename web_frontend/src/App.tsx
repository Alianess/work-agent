import {
  AlertCircle,
  AtSign,
  Brain,
  Check,
  ChevronDown,
  ChevronRight,
  CheckCircle2,
  Clock3,
  Cloud,
  Copy,
  Cpu,
  Download,
  Eye,
  EyeOff,
  ExternalLink,
  Folder,
  FileText,
  FolderOpen,
  ImageIcon,
  Library,
  LogOut,
  Loader2,
  MessageCircle,
  MessageSquarePlus,
  Mic,
  MoreHorizontal,
  Music2,
  Paperclip,
  PanelLeft,
  Pencil,
  Pin,
  PinOff,
  Plus,
  RefreshCw,
  Search,
  SendHorizontal,
  Settings2,
  Sparkles,
  Trash2,
  Upload,
  UserRound,
  Wrench,
  X
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties, Dispatch, DragEvent, FormEvent, MouseEvent, MutableRefObject, ReactNode, SetStateAction } from "react";
import { api } from "./api";
import type { AuthUser } from "./api";
import type {
  AgentActivityEvent,
  AgentSettingsPayload,
  AgentStreamEvent,
  AsrSettingsPayload,
  AttachmentItem,
  ChatMessage,
  CrossChatMemory,
  FileItem,
  FilePayload,
  MeetingArchive,
  MeetingTime,
  MeetingResult,
  MeetingMinutesSettingsPayload,
  ModelsPayload,
  ModelProfile,
  Project,
  ProjectSummary,
  ReasoningEffort,
  SkillInfo,
  SkillInstructionsPayload,
  TemporarySyncPayload,
  ToolsPayload
} from "./types";

type View = "agent" | "projects" | "skills" | "artifacts" | "models" | "transcribe" | "more" | "sync";
type ArtifactTab = "meeting" | "files";
type FileFilter = "all" | "audio" | "image" | "document" | "output";
type StatusTone = "idle" | "success" | "error" | "loading";
type RealtimeTranscriptionStatus = "idle" | "recording" | "processing" | "error";
type ModelProviderPresetId = "openai-compatible" | "openai" | "deepseek" | "openrouter";
type ComposerSubmenu = "model" | "reasoning" | "advanced" | null;
type ModelEditorMode = "add" | "edit" | null;

const REASONING_OPTIONS: Array<{ value: ReasoningEffort; label: string; shortLabel: string }> = [
  { value: "light", label: "轻度", shortLabel: "轻" },
  { value: "medium", label: "中", shortLabel: "中" },
  { value: "high", label: "高", shortLabel: "高" },
  { value: "very_high", label: "极高", shortLabel: "极高" }
];
const REASONING_STORAGE_KEY = "work-agent-reasoning-effort";

const MODEL_PROVIDER_PRESETS: Record<
  ModelProviderPresetId,
  {
    label: string;
    description: string;
    provider: string;
    base_url: string;
    model: string;
    temperature: number;
    max_tokens: number;
    timeout_seconds: number;
  }
> = {
  "openai-compatible": {
    label: "自定义兼容接口",
    description: "适用于实现 OpenAI Chat Completions 的代理或本地服务",
    provider: "openai-compatible",
    base_url: "",
    model: "",
    temperature: 0.6,
    max_tokens: 16384,
    timeout_seconds: 180
  },
  openai: {
    label: "OpenAI",
    description: "OpenAI 官方 Chat Completions 接口",
    provider: "openai-compatible",
    base_url: "https://api.openai.com/v1",
    model: "",
    temperature: 0.6,
    max_tokens: 16384,
    timeout_seconds: 180
  },
  deepseek: {
    label: "DeepSeek",
    description: "DeepSeek 官方 OpenAI-compatible 接口",
    provider: "deepseek",
    base_url: "https://api.deepseek.com",
    model: "deepseek-v4-flash",
    temperature: 0.6,
    max_tokens: 8192,
    timeout_seconds: 180
  },
  openrouter: {
    label: "OpenRouter",
    description: "通过统一接口选择 OpenRouter 上的模型",
    provider: "openai-compatible",
    base_url: "https://openrouter.ai/api/v1",
    model: "",
    temperature: 0.6,
    max_tokens: 16384,
    timeout_seconds: 180
  }
};

function createDefaultModelForm() {
  const preset = MODEL_PROVIDER_PRESETS["openai-compatible"];
  return {
    name: "",
    preset: "openai-compatible" as ModelProviderPresetId,
    provider: preset.provider,
    base_url: preset.base_url,
    model: preset.model,
    api_key: "",
    temperature: preset.temperature,
    max_tokens: preset.max_tokens,
    timeout_seconds: preset.timeout_seconds,
    set_default: false,
    source_name: ""
  };
}

type ConversationHistoryItem = {
  id: string;
  title: string;
  group: string;
  messages: ChatMessage[];
  contextSummary?: string;
  contextSummaryMessageCount?: number;
  activities?: ActivityRecordMap;
  activeActivityIndex?: number | null;
  pinned?: boolean;
  projectId?: string;
};

type ActivityRecord = {
  events: AgentActivityEvent[];
  elapsedMs: number;
  completed: boolean;
  turnId?: string;
};

type ActivityRecordMap = Record<number, ActivityRecord>;
type QueuedChatItem = {
  content: string;
  attachments: AttachmentItem[];
  skill: SkillInfo | null;
};
type ActiveChatRun = {
  abortController: AbortController;
  turnId: string | null;
  assistantIndex: number;
  startedAt: number;
};
type RunChatMessageOptions = {
  conversationId?: string;
  baseMessages?: ChatMessage[];
  activityRecords?: ActivityRecordMap;
  conversationSummary?: string;
  conversationSummaryMessageCount?: number;
  rewindUserMessageOrdinal?: number;
};
type ActivityDisplayGroupKind = "commands" | "files" | "tools";
type ActivityDisplayItem =
  | { kind: "event"; key: string; event: AgentActivityEvent }
  | {
      kind: "group";
      key: string;
      groupKind: ActivityDisplayGroupKind;
      status: NonNullable<AgentActivityEvent["command_status"]>;
      title: string;
      detail: string;
      icon: string;
      open: boolean;
      events: AgentActivityEvent[];
    };
type RealtimeTranscriptSegment = {
  id: string;
  index: number;
  text: string;
  startedAt: number;
  finishedAt: number;
  audioPath?: string;
  transcriptPath?: string;
  asrElapsedMs?: number;
  pending?: boolean;
  error?: string;
};

type MeetingGroup = {
  key: string;
  manifestPath?: string;
  title: string;
  modified: number;
  meetingTime?: MeetingTime | null;
  asr?: FileItem;
  internal?: FileItem;
  work?: FileItem;
  workDocx?: FileItem;
};

const fileFilterOptions: Array<{ id: FileFilter; label: string }> = [
  { id: "all", label: "全部" },
  { id: "audio", label: "录音" },
  { id: "image", label: "图片" },
  { id: "document", label: "文件" },
  { id: "output", label: "产出" }
];

const conversationStorageKey = "work-agent-conversation-history";
const pendingConversationTitle = "待命名对话";
const untitledConversationTitle = "待命名对话";
const toolCallMarkupPattern = /<\/?\s*(?:tool_calls?|工具调用(?:列表)?)(?=[\s>/])/i;

const dateFormatter = new Intl.DateTimeFormat("zh-CN", {
  month: "2-digit",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit"
});

const numberFormatter = new Intl.NumberFormat("zh-CN");
const voiceLevelBarCount = 64;
const realtimeTranscriptionMinSegmentMs = 7000;
const realtimeTranscriptionPauseMs = 1200;
const realtimeTranscriptionMaxSegmentMs = 22000;
const realtimeTranscriptionVadFrameMs = 100;
const realtimeTranscriptionBackendVadFrameMs = 30;
const realtimeTranscriptionBackendVadSampleRate = 16000;
const realtimeTranscriptionBackendVadBatchFrames = 3;
const realtimeTranscriptionStartSpeechFrames = 2;
const realtimeTranscriptionNoiseFloorInitial = 0.06;
const realtimeTranscriptionVoiceMargin = 0.028;
const realtimeTranscriptionVoiceRatio = 1.55;

function initialView(): View {
  const query = new URLSearchParams(window.location.search).get("view");
  if (
    query === "skills" ||
    query === "models" ||
    query === "projects" ||
    query === "artifacts" ||
    query === "agent" ||
    query === "transcribe" ||
    query === "more" ||
    query === "sync"
  ) {
    return query;
  }
  if (query === "meeting") return "artifacts";
  return "agent";
}

function loadReasoningEffort(): ReasoningEffort {
  const saved = window.localStorage.getItem(REASONING_STORAGE_KEY);
  return REASONING_OPTIONS.some((option) => option.value === saved)
    ? (saved as ReasoningEffort)
    : "medium";
}

function reasoningOption(value: ReasoningEffort) {
  return REASONING_OPTIONS.find((option) => option.value === value) ?? REASONING_OPTIONS[1];
}

function formatProfileLabel(profile?: ModelProfile) {
  if (!profile) return "选择模型";
  const raw = profile.model || profile.name;
  const normalized = raw.toLowerCase();
  if (normalized === "gpt-5.6-luna") return "5.6 Luna";
  if (normalized === "gpt-5.6-terra") return "5.6 Terra";
  if (normalized === "gpt-5.6-sol") return "5.6 Sol";
  if (normalized === "deepseek-v4-pro") return "DeepSeek V4 Pro";
  if (normalized === "deepseek-v4-flash") return "DeepSeek V4 Flash";
  return raw;
}

function formatProfileCompactLabel(profile?: ModelProfile) {
  const label = formatProfileLabel(profile);
  if (label === "DeepSeek V4 Pro") return "V4 Pro";
  if (label === "DeepSeek V4 Flash") return "V4 Flash";
  return label;
}

function presetForModelProfile(profile: ModelProfile): ModelProviderPresetId {
  if (profile.provider === "deepseek" || profile.base_url.includes("api.deepseek.com")) {
    return "deepseek";
  }
  if (profile.base_url.includes("openrouter.ai")) return "openrouter";
  if (profile.base_url.includes("api.openai.com")) return "openai";
  return "openai-compatible";
}

function nextCopiedModelName(name: string, profiles: ModelProfile[]) {
  const existing = new Set(profiles.map((profile) => profile.name));
  const base = `${name}-copy`;
  if (!existing.has(base)) return base;
  for (let index = 2; index < 100; index += 1) {
    const candidate = `${base}-${index}`;
    if (!existing.has(candidate)) return candidate;
  }
  return `${base}-${Date.now()}`;
}

function providerInitial(provider: string) {
  const normalized = provider.trim().toLowerCase();
  if (normalized.includes("deepseek")) return "DS";
  if (normalized.includes("openai")) return "AI";
  return normalized.slice(0, 2).toUpperCase() || "LL";
}

export default function App() {
  const [authState, setAuthState] = useState<"loading" | "anonymous" | "authenticated">("loading");
  const [currentUser, setCurrentUser] = useState<AuthUser | null>(null);
  const [authMode, setAuthMode] = useState<"login" | "register">("login");
  const [authForm, setAuthForm] = useState({ username: "", password: "", confirm: "" });
  const [authBusy, setAuthBusy] = useState(false);
  const [authError, setAuthError] = useState("");
  const [view, setView] = useState<View>(initialView);
  const [artifactTab, setArtifactTab] = useState<ArtifactTab>(
    new URLSearchParams(window.location.search).get("tab") === "files" ? "files" : "meeting"
  );
  const [models, setModels] = useState<ModelsPayload | null>(null);
  const [asrSettings, setAsrSettings] = useState<AsrSettingsPayload | null>(null);
  const [agentSettings, setAgentSettings] = useState<AgentSettingsPayload | null>(null);
  const [meetingMinutesSettings, setMeetingMinutesSettings] = useState<MeetingMinutesSettingsPayload | null>(null);
  const [crossChatMemories, setCrossChatMemories] = useState<CrossChatMemory[]>([]);
  const [memoryProfile, setMemoryProfile] = useState<{ content: string; updated_at: number } | null>(null);
  const [memoryQuery, setMemoryQuery] = useState("");
  const [memoryScope, setMemoryScope] = useState<"all" | "account" | "projects">("all");
  const [editingMemoryId, setEditingMemoryId] = useState<string | null>(null);
  const [memoryDraft, setMemoryDraft] = useState("");
  const [deleteConfirmMemoryId, setDeleteConfirmMemoryId] = useState<string | null>(null);
  const [tools, setTools] = useState<ToolsPayload | null>(null);
  const [skills, setSkills] = useState<SkillInfo[]>([]);
  const [skillInstructions, setSkillInstructions] = useState<SkillInstructionsPayload | null>(null);
  const [skillInstructionsDraft, setSkillInstructionsDraft] = useState("");
  const [skillInstructionsLoading, setSkillInstructionsLoading] = useState(false);
  const [workspace, setWorkspace] = useState("");
  const [files, setFiles] = useState<FileItem[]>([]);
  const [temporarySync, setTemporarySync] = useState<TemporarySyncPayload | null>(null);
  const [temporarySyncText, setTemporarySyncText] = useState("");
  const [temporarySyncTextDirty, setTemporarySyncTextDirty] = useState(false);
  const temporarySyncTextDirtyRef = useRef(false);
  const temporarySyncRequestRef = useRef(0);
  const [temporarySyncBusy, setTemporarySyncBusy] = useState(false);
  const [temporarySyncClock, setTemporarySyncClock] = useState(Date.now());
  const [meetingArchives, setMeetingArchives] = useState<MeetingArchive[]>([]);
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);
  const [activeProjectId, setActiveProjectId] = useState<string | null>(null);
  const activeProjectIdRef = useRef<string | null>(null);
  const [projectCreateOpen, setProjectCreateOpen] = useState(false);
  const [projectCreateForm, setProjectCreateForm] = useState({ name: "", instructions: "" });
  const [projectSettingsOpen, setProjectSettingsOpen] = useState(false);
  const [projectSettingsForm, setProjectSettingsForm] = useState({ name: "", instructions: "" });
  const [projectDetailTab, setProjectDetailTab] = useState<"chat" | "files">("chat");
  const [projectChatDraft, setProjectChatDraft] = useState("");
  const [meetingSyncOpen, setMeetingSyncOpen] = useState(false);
  const [meetingSyncProjectId, setMeetingSyncProjectId] = useState("");
  const [meetingSyncMessage, setMeetingSyncMessage] = useState("");
  const [selectedFile, setSelectedFile] = useState<FilePayload | null>(null);
  const [filesRoot, setFilesRoot] = useState("meet_files");
  const [fileQuery, setFileQuery] = useState("");
  const [fileFilter, setFileFilter] = useState<FileFilter>("all");
  const [fileActionText, setFileActionText] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<{ tone: StatusTone; text: string }>({
    tone: "idle",
    text: "就绪"
  });
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      role: "assistant",
      content:
        "你好，我是本地工作智能体。你可以直接和我对话，也可以让我读取工作区文件、生成会议纪要或整理当前项目。"
    }
  ]);
  const [chatInput, setChatInput] = useState("");
  const [editingMessageIndex, setEditingMessageIndex] = useState<number | null>(null);
  const [editMessageDraft, setEditMessageDraft] = useState("");
  const [conversationHistory, setConversationHistory] = useState<ConversationHistoryItem[]>([]);
  const conversationHistoryRef = useRef<ConversationHistoryItem[]>(conversationHistory);
  const [currentConversationId, setCurrentConversationId] = useState(() => createConversationId());
  const [conversationSummary, setConversationSummary] = useState("");
  const [conversationSummaryMessageCount, setConversationSummaryMessageCount] = useState(0);
  const [activeChatTurnIndex, setActiveChatTurnIndex] = useState<number | null>(null);
  const [searchOpen, setSearchOpen] = useState(false);
  const [conversationSearch, setConversationSearch] = useState("");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [historyMenu, setHistoryMenu] = useState<{ id: string; x: number; y: number } | null>(null);
  const [renamingConversationId, setRenamingConversationId] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState("");
  const [deleteConfirmConversationId, setDeleteConfirmConversationId] = useState<string | null>(null);
  const [activityOpen, setActivityOpen] = useState(false);
  const [activityRecords, setActivityRecords] = useState<ActivityRecordMap>({});
  const [activityPanelMessageIndex, setActivityPanelMessageIndex] = useState<number | null>(null);
  const [activityEvents, setActivityEvents] = useState<AgentActivityEvent[]>([]);
  const [activityElapsedMs, setActivityElapsedMs] = useState(0);
  const [activityMessageIndex, setActivityMessageIndex] = useState<number | null>(null);
  const [activityStartedAt, setActivityStartedAt] = useState<number | null>(null);
  const [activityRunning, setActivityRunning] = useState(false);
  const [activityNow, setActivityNow] = useState(Date.now());
  const [queuedChatCount, setQueuedChatCount] = useState(0);
  const currentConversationIdRef = useRef(currentConversationId);
  const activeChatRunsRef = useRef<Map<string, ActiveChatRun>>(new Map());
  const chatThreadRef = useRef<HTMLElement | null>(null);
  const pendingChatScrollToBottomRef = useRef(false);
  const queuedChatMessagesRef = useRef<Map<string, QueuedChatItem[]>>(new Map());
  const activityRunningRef = useRef(false);
  const chatMessagesRef = useRef<ChatMessage[]>(chatMessages);
  const titleGenerationInFlightRef = useRef<Set<string>>(new Set());
  const conversationArchiveReadyRef = useRef(false);
  const [speechSupported, setSpeechSupported] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const voiceChunksRef = useRef<BlobPart[]>([]);
  const voiceStreamRef = useRef<MediaStream | null>(null);
  const voiceAudioContextRef = useRef<AudioContext | null>(null);
  const voiceAnimationRef = useRef<number | null>(null);
  const voiceShouldTranscribeRef = useRef(false);
  const [voiceLevels, setVoiceLevels] = useState<number[]>(() => createIdleVoiceLevels());
  const [meetingLiveTitle, setMeetingLiveTitle] = useState("实时会议转写");
  const [meetingLiveSegments, setMeetingLiveSegments] = useState<RealtimeTranscriptSegment[]>([]);
  const [meetingLiveStatus, setMeetingLiveStatus] = useState<RealtimeTranscriptionStatus>("idle");
  const [meetingLiveLevels, setMeetingLiveLevels] = useState<number[]>(() => createIdleVoiceLevels());
  const [meetingLivePending, setMeetingLivePending] = useState(0);
  const [meetingLiveSavedPath, setMeetingLiveSavedPath] = useState("");
  const meetingLiveRecorderRef = useRef<MediaRecorder | null>(null);
  const meetingLiveStreamRef = useRef<MediaStream | null>(null);
  const meetingLiveAudioContextRef = useRef<AudioContext | null>(null);
  const meetingLiveAnimationRef = useRef<number | null>(null);
  const meetingLiveChunkIndexRef = useRef(0);
  const meetingLiveActiveRef = useRef(false);
  const meetingLivePendingRef = useRef(0);
  const meetingLiveTranscriptionQueueRef = useRef<Promise<void>>(Promise.resolve());
  const meetingLiveChunkPartsRef = useRef<BlobPart[]>([]);
  const meetingLiveOptionsRef = useRef<MediaRecorderOptions | undefined>(undefined);
  const meetingLiveSegmentStartedAtRef = useRef(0);
  const meetingLiveLastVoiceAtRef = useRef(0);
  const meetingLiveBoundaryTimerRef = useRef<number | null>(null);
  const meetingLiveCurrentLevelRef = useRef(0);
  const meetingLiveNoiseFloorRef = useRef(realtimeTranscriptionNoiseFloorInitial);
  const meetingLiveHasSpokenRef = useRef(false);
  const meetingLiveSpeechFramesRef = useRef(0);
  const meetingLiveSpeechCandidateFramesRef = useRef(0);
  const meetingLiveSilenceFramesRef = useRef(0);
  const meetingLiveBackendVadActiveRef = useRef(false);
  const meetingLiveBackendVadUnavailableRef = useRef(false);
  const meetingLiveVadAudioContextRef = useRef<AudioContext | null>(null);
  const meetingLiveVadSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const meetingLiveVadProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const meetingLiveVadSampleBufferRef = useRef<Float32Array>(new Float32Array(0));
  const meetingLiveVadFrameBatchRef = useRef<string[]>([]);
  const meetingLiveVadRequestActiveRef = useRef(false);
  const [selectedSkill, setSelectedSkill] = useState<SkillInfo | null>(null);
  const [attachments, setAttachments] = useState<AttachmentItem[]>([]);
  const [dragActive, setDragActive] = useState(false);
  const [actionMenuOpen, setActionMenuOpen] = useState(false);
  const [composerModelMenuOpen, setComposerModelMenuOpen] = useState(false);
  const [composerSubmenu, setComposerSubmenu] = useState<ComposerSubmenu>(null);
  const [reasoningEffort, setReasoningEffort] = useState<ReasoningEffort>(loadReasoningEffort);
  const composerModelMenuRef = useRef<HTMLDivElement | null>(null);
  const [meetingOutputKey, setMeetingOutputKey] = useState("");
  const [meetingResult, setMeetingResult] = useState<MeetingResult | null>(null);

  const defaultProfile = models?.default_profile ?? "deepseek-v4-pro";
  const profiles = models?.profiles ?? [];

  const [agentForm, setAgentForm] = useState({
    profile: defaultProfile
  });
  const currentProfile =
    profiles.find((profile) => profile.name === agentForm.profile) ??
    profiles.find((profile) => profile.name === defaultProfile);

  const [meetingForm, setMeetingForm] = useState({
    transcript_path: "",
    output_dir: "meet_files",
    meeting_name: "",
    confirmed_info: "",
    supplemental_paths: "",
    profile: defaultProfile
  });
  const [meetingMinutesSettingsForm, setMeetingMinutesSettingsForm] = useState({
    default_output_dir: "meet_files",
    custom_instructions: ""
  });

  const [modelForm, setModelForm] = useState(createDefaultModelForm);
  const [modelEditorMode, setModelEditorMode] = useState<ModelEditorMode>(null);
  const [editingModelName, setEditingModelName] = useState("");
  const [discoveredModelIds, setDiscoveredModelIds] = useState<string[]>([]);
  const [modelConnectionResult, setModelConnectionResult] = useState<{
    tone: "success" | "error";
    text: string;
  } | null>(null);
  const [deleteConfirmModelName, setDeleteConfirmModelName] = useState("");
  const [showModelApiKey, setShowModelApiKey] = useState(false);
  const [asrSettingsForm, setAsrSettingsForm] = useState({
    profile: "qwen3-asr-mlx-8bit",
    model_id: "meeting_audio_minutes/model_cache/mlx-community/Qwen3-ASR-1.7B-8bit",
    hotwords: ""
  });
  const [agentSettingsForm, setAgentSettingsForm] = useState({
    nickname: "",
    occupation: "",
    details: "",
    memory_enabled: true,
    work_background: "",
    company_document_format: ""
  });

  useEffect(() => {
    void api.authMe()
      .then((payload) => {
        if (payload.authenticated && payload.user) {
          setCurrentUser(payload.user);
          setAuthState("authenticated");
          return;
        }
        setAuthState("anonymous");
      })
      .catch(async () => {
        try {
          const health = await api.health();
          if (health.auth_enabled) {
            setAuthState("anonymous");
          } else {
            // Keeps the already-running pre-auth backend usable until its one-time restart.
            setCurrentUser({ id: 1, username: "admin", role: "admin", created_at: 0 });
            setAuthState("authenticated");
          }
        } catch {
          setAuthState("anonymous");
        }
      });
  }, []);

  useEffect(() => {
    if (authState !== "authenticated" || !currentUser) return;
    conversationArchiveReadyRef.current = false;
    const localHistory = loadConversationHistory(currentUser.username);
    conversationHistoryRef.current = localHistory;
    setConversationHistory(localHistory);
    setCurrentConversationId(createConversationId());
    void refreshAll();
    void restoreConversationArchive();
  }, [authState, currentUser?.id]);

  useEffect(() => {
    if (authState !== "authenticated" || view !== "sync") return;
    void refreshTemporarySync(true);
    const refreshTimer = window.setInterval(() => {
      void refreshTemporarySync(true);
    }, 10_000);
    const clockTimer = window.setInterval(() => {
      setTemporarySyncClock(Date.now());
    }, 1_000);
    return () => {
      window.clearInterval(refreshTimer);
      window.clearInterval(clockTimer);
    };
  }, [authState, view]);

  useEffect(() => {
    chatMessagesRef.current = chatMessages;
  }, [chatMessages]);

  useEffect(() => {
    currentConversationIdRef.current = currentConversationId;
  }, [currentConversationId]);

  useEffect(() => {
    activeProjectIdRef.current = activeProjectId;
  }, [activeProjectId]);

  useEffect(() => {
    activityRunningRef.current = activityRunning;
  }, [activityRunning]);

  useEffect(() => {
    conversationHistoryRef.current = conversationHistory;
    if (!currentUser) return;
    saveConversationHistory(conversationHistory, currentUser.username);
    if (conversationArchiveReadyRef.current) {
      void api.saveConversations({ items: conversationHistory }).catch(() => {
        // The browser copy is still kept if the workspace archive is temporarily unavailable.
      });
    }
  }, [conversationHistory, currentUser?.id]);

  useEffect(() => {
    const candidates = conversationHistory.filter(shouldGenerateModelTitle);
    for (const item of candidates) {
      if (titleGenerationInFlightRef.current.has(item.id)) continue;
      titleGenerationInFlightRef.current.add(item.id);
      void generateAndApplyConversationTitle(item);
    }
  }, [conversationHistory]);

  useEffect(() => {
    setSpeechSupported(
      typeof navigator.mediaDevices?.getUserMedia === "function" && typeof MediaRecorder !== "undefined"
    );
    return () => {
      voiceShouldTranscribeRef.current = false;
      if (mediaRecorderRef.current?.state === "recording") {
        mediaRecorderRef.current.stop();
      }
      if (meetingLiveRecorderRef.current?.state === "recording") {
        meetingLiveActiveRef.current = false;
        stopMeetingLiveBoundaryLoop();
        meetingLiveRecorderRef.current.stop();
      }
      stopVoiceLevelMeter(voiceAnimationRef, voiceAudioContextRef);
      stopVoiceStream(voiceStreamRef.current);
      stopVoiceLevelMeter(meetingLiveAnimationRef, meetingLiveAudioContextRef);
      stopVoiceStream(meetingLiveStreamRef.current);
    };
  }, []);

  useEffect(() => {
    if (!models) return;
    setAgentForm((form) => ({ ...form, profile: form.profile || models.default_profile }));
    setMeetingForm((form) => ({ ...form, profile: form.profile || models.default_profile }));
  }, [models]);

  useEffect(() => {
    window.localStorage.setItem(REASONING_STORAGE_KEY, reasoningEffort);
  }, [reasoningEffort]);

  useEffect(() => {
    if (!composerModelMenuOpen) return;
    const closeMenu = (event: PointerEvent) => {
      if (composerModelMenuRef.current?.contains(event.target as Node)) return;
      setComposerModelMenuOpen(false);
      setComposerSubmenu(null);
    };
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      setComposerModelMenuOpen(false);
      setComposerSubmenu(null);
    };
    window.addEventListener("pointerdown", closeMenu);
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("pointerdown", closeMenu);
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [composerModelMenuOpen]);

  useEffect(() => {
    if (!activityRunning) return;
    setActivityNow(Date.now());
    const timer = window.setInterval(() => setActivityNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [activityRunning]);

  useEffect(() => {
    if (view !== "agent" || !pendingChatScrollToBottomRef.current) return;
    pendingChatScrollToBottomRef.current = false;
    scheduleChatScrollToBottom();
  }, [view, currentConversationId, chatMessages.length]);

  useEffect(() => {
    if (!historyMenu) return;
    const closeMenu = () => {
      setHistoryMenu(null);
      setDeleteConfirmConversationId(null);
    };
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        closeMenu();
      }
    };
    window.addEventListener("pointerdown", closeMenu);
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("pointerdown", closeMenu);
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [historyMenu]);

  const keyReadyCount = useMemo(
    () => profiles.filter((profile) => profile.api_key_configured).length,
    [profiles]
  );
  const skillQuery = useMemo(() => parseSkillQuery(chatInput), [chatInput]);
  const suggestedSkills = useMemo(() => {
    if (skillQuery === null) return [];
    const query = skillQuery.toLowerCase();
    return skills
      .filter((skill) => {
        const haystack = `${skill.label} ${skill.mention} ${skill.description}`.toLowerCase();
        return haystack.includes(query);
      })
      .slice(0, 5);
  }, [skillQuery, skills]);
  const filteredFiles = useMemo(
    () => filterLibraryFiles(files, fileQuery, fileFilter),
    [files, fileQuery, fileFilter]
  );
  const libraryCounts = useMemo(() => countLibraryFiles(files), [files]);
  const filteredConversationHistory = useMemo(
    () => filterConversations(conversationHistory, conversationSearch),
    [conversationHistory, conversationSearch]
  );
  const filteredCrossChatMemories = useMemo(() => {
    const query = memoryQuery.trim().toLocaleLowerCase("zh-CN");
    return crossChatMemories.filter((memory) => {
      if (memoryScope === "account" && memory.project_id) return false;
      if (memoryScope === "projects" && !memory.project_id) return false;
      if (!query) return true;
      return `${memory.content} ${memory.conversation_title}`.toLocaleLowerCase("zh-CN").includes(query);
    });
  }, [crossChatMemories, memoryQuery, memoryScope]);
  const chatUserTurns = useMemo(
    () =>
      chatMessages.flatMap((message, messageIndex) =>
        message.role === "user"
          ? [{ messageIndex, label: chatTurnLabel(message.content) }]
          : []
      ),
    [chatMessages]
  );

  useEffect(() => {
    const thread = chatThreadRef.current;
    if (view !== "agent" || !thread || chatUserTurns.length === 0) {
      setActiveChatTurnIndex(null);
      return;
    }
    let animationFrame = 0;
    const updateActiveTurn = () => {
      animationFrame = 0;
      const threadRect = thread.getBoundingClientRect();
      const readingLine = threadRect.top + Math.min(180, Math.max(96, thread.clientHeight * 0.22));
      let activeIndex = chatUserTurns[0].messageIndex;
      for (const turn of chatUserTurns) {
        const element = thread.querySelector<HTMLElement>(
          `[data-chat-message-index="${turn.messageIndex}"]`
        );
        if (element && element.getBoundingClientRect().top <= readingLine) {
          activeIndex = turn.messageIndex;
        }
      }
      if (thread.scrollTop + thread.clientHeight >= thread.scrollHeight - 24) {
        activeIndex = chatUserTurns[chatUserTurns.length - 1].messageIndex;
      }
      setActiveChatTurnIndex((current) => (current === activeIndex ? current : activeIndex));
    };
    const scheduleUpdate = () => {
      if (animationFrame) return;
      animationFrame = window.requestAnimationFrame(updateActiveTurn);
    };
    thread.addEventListener("scroll", scheduleUpdate, { passive: true });
    window.addEventListener("resize", scheduleUpdate);
    scheduleUpdate();
    return () => {
      thread.removeEventListener("scroll", scheduleUpdate);
      window.removeEventListener("resize", scheduleUpdate);
      if (animationFrame) window.cancelAnimationFrame(animationFrame);
    };
  }, [view, currentConversationId, chatUserTurns.length]);
  const meetingGroups = useMemo(
    () => buildMeetingGroups(meetingArchives, files),
    [meetingArchives, files]
  );
  const activeMeetingGroup =
    meetingGroups.find((group) => group.key === meetingOutputKey) ?? meetingGroups[0] ?? null;
  const visibleActivityElapsedMs =
    activityRunning && activityStartedAt
      ? Math.max(activityElapsedMs, activityNow - activityStartedAt)
      : activityElapsedMs;
  const activityRecordForMessage = (messageIndex: number): ActivityRecord | null => {
    if (messageIndex === activityMessageIndex) {
      const existing = activityRecords[messageIndex];
      if (activityEvents.length === 0 && !activityStartedAt && !existing) return null;
      return {
        events: activityEvents.length > 0 ? activityEvents : existing?.events ?? [],
        elapsedMs: visibleActivityElapsedMs || existing?.elapsedMs || 0,
        completed: !activityRunning && (existing?.completed ?? true)
      };
    }
    return activityRecords[messageIndex] ?? null;
  };
  const panelActivityIndex = activityPanelMessageIndex ?? activityMessageIndex;
  const panelActivityRecord =
    panelActivityIndex === null ? null : activityRecordForMessage(panelActivityIndex);

  useEffect(() => {
    if (meetingGroups.length === 0) {
      if (meetingOutputKey) setMeetingOutputKey("");
      return;
    }
    if (!meetingGroups.some((group) => group.key === meetingOutputKey)) {
      setMeetingOutputKey(meetingGroups[0].key);
    }
  }, [meetingGroups, meetingOutputKey]);

  useEffect(() => {
    setMeetingSyncOpen(false);
    setMeetingSyncMessage("");
  }, [activeMeetingGroup?.key]);

  function setViewWithUrl(next: View) {
    setView(next);
    const url = new URL(window.location.href);
    url.searchParams.set("view", next);
    if (next !== "artifacts") {
      url.searchParams.delete("tab");
    } else {
      url.searchParams.set("tab", artifactTab);
    }
    window.history.replaceState(null, "", url);
  }

  function setArtifactTabWithUrl(next: ArtifactTab) {
    setArtifactTab(next);
    setView("artifacts");
    const url = new URL(window.location.href);
    url.searchParams.set("view", "artifacts");
    url.searchParams.set("tab", next);
    window.history.replaceState(null, "", url);
  }

  async function refreshTemporarySync(silent = false) {
    const requestId = ++temporarySyncRequestRef.current;
    if (!silent) {
      setTemporarySyncBusy(true);
      setStatus({ tone: "loading", text: "正在同步临时区…" });
    }
    try {
      const payload = await api.temporarySync();
      if (requestId !== temporarySyncRequestRef.current) return;
      setTemporarySync(payload);
      setTemporarySyncClock(Date.now());
      if (!temporarySyncTextDirtyRef.current) {
        setTemporarySyncText(payload.text.content);
      }
      if (!silent) {
        setStatus({ tone: "success", text: "临时同步区已刷新" });
      }
    } catch (error) {
      if (!silent) {
        setStatus({ tone: "error", text: explainError(error) });
      }
    } finally {
      if (!silent) setTemporarySyncBusy(false);
    }
  }

  async function saveTemporarySyncText() {
    ++temporarySyncRequestRef.current;
    setTemporarySyncBusy(true);
    setStatus({ tone: "loading", text: "正在同步文字…" });
    try {
      const payload = await api.saveTemporarySyncText(temporarySyncText);
      setTemporarySync((current) =>
        current
          ? { ...current, text: payload.text, server_time: Math.floor(Date.now() / 1000) }
          : {
              text: payload.text,
              files: [],
              file_ttl_seconds: 3600,
              server_time: Math.floor(Date.now() / 1000)
            }
      );
      temporarySyncTextDirtyRef.current = false;
      setTemporarySyncTextDirty(false);
      setStatus({ tone: "success", text: "文字已同步到同一账号的其他电脑" });
    } catch (error) {
      setStatus({ tone: "error", text: explainError(error) });
    } finally {
      setTemporarySyncBusy(false);
    }
  }

  async function uploadTemporarySyncFiles(fileList: FileList | File[]) {
    const selected = Array.from(fileList);
    if (selected.length === 0) return;
    const oversized = selected.find((file) => file.size > 100 * 1024 * 1024);
    if (oversized) {
      setStatus({ tone: "error", text: `${oversized.name} 超过 100 MB，未上传` });
      return;
    }
    setTemporarySyncBusy(true);
    setStatus({ tone: "loading", text: `正在上传 ${selected.length} 个文件…` });
    try {
      for (const file of selected) {
        await api.addTemporarySyncFile({
          name: file.name,
          mime_type: file.type || "application/octet-stream",
          content_base64: await fileToBase64(file)
        });
      }
      await refreshTemporarySync(true);
      setStatus({
        tone: "success",
        text: `${selected.length} 个文件已同步，1 小时后自动删除`
      });
    } catch (error) {
      setStatus({ tone: "error", text: explainError(error) });
    } finally {
      setTemporarySyncBusy(false);
    }
  }

  async function deleteTemporarySyncFile(id: string) {
    ++temporarySyncRequestRef.current;
    setTemporarySyncBusy(true);
    try {
      await api.deleteTemporarySyncFile(id);
      setTemporarySync((current) =>
        current ? { ...current, files: current.files.filter((file) => file.id !== id) } : current
      );
      setStatus({ tone: "success", text: "临时文件已删除" });
    } catch (error) {
      setStatus({ tone: "error", text: explainError(error) });
    } finally {
      setTemporarySyncBusy(false);
    }
  }

  async function refreshAll() {
    setBusy(true);
    setStatus({ tone: "loading", text: "正在刷新…" });
    try {
      const [
        health,
        nextModels,
        nextAsrSettings,
        nextAgentSettings,
        nextMeetingMinutesSettings,
        nextMemories,
        nextTools,
        nextSkills,
        nextFiles,
        nextMeetingArchives,
        nextProjects
      ] = await Promise.all([
        api.health(),
        api.models(),
        api.asrSettings(),
        api.agentSettings(),
        api.meetingMinutesSettings(),
        api.memories(),
        api.tools(),
        api.skills(),
        api.files(filesRoot),
        api.meetingArchives(),
        api.projects()
      ]);
      setWorkspace(health.workspace);
      setModels(nextModels);
      setAsrSettings(nextAsrSettings);
      setAsrSettingsForm({
        profile: nextAsrSettings.profile,
        model_id: nextAsrSettings.model_id,
        hotwords: nextAsrSettings.hotwords
      });
      setAgentSettings(nextAgentSettings);
      setAgentSettingsForm({
        nickname: nextAgentSettings.nickname,
        occupation: nextAgentSettings.occupation,
        details: nextAgentSettings.details,
        memory_enabled: nextAgentSettings.memory_enabled,
        work_background: nextAgentSettings.work_background,
        company_document_format: nextAgentSettings.company_document_format
      });
      setMeetingMinutesSettings(nextMeetingMinutesSettings);
      setMeetingMinutesSettingsForm({
        default_output_dir: nextMeetingMinutesSettings.default_output_dir,
        custom_instructions: nextMeetingMinutesSettings.custom_instructions
      });
      setCrossChatMemories(nextMemories.memories);
      setMemoryProfile(nextMemories.profile ? { content: nextMemories.profile.content, updated_at: nextMemories.profile.updated_at } : null);
      setTools(nextTools);
      setSkills(nextSkills.skills);
      setFiles(nextFiles.files);
      setMeetingArchives(nextMeetingArchives.meetings);
      setProjects(nextProjects.projects);
      setStatus({ tone: "success", text: "工作区已刷新" });
    } catch (error) {
      setStatus({ tone: "error", text: explainError(error) });
    } finally {
      setBusy(false);
    }
  }

  async function setSkillEnabled(skill: SkillInfo, enabled: boolean) {
    setBusy(true);
    try {
      const payload = await api.setSkillEnabled(skill.id, enabled);
      setSkills(payload.skills);
      setStatus({
        tone: "success",
        text: `${skill.label}已${enabled ? "启用" : "关闭"}；新对话会按此配置加载能力`
      });
    } catch (error) {
      setStatus({ tone: "error", text: explainError(error) });
    } finally {
      setBusy(false);
    }
  }

  async function openSkillInstructions(skill: SkillInfo) {
    setSkillInstructionsLoading(true);
    setSkillInstructions(null);
    try {
      const payload = await api.skillInstructions(skill.id);
      setSkillInstructions(payload);
      setSkillInstructionsDraft(payload.content);
    } catch (error) {
      setStatus({ tone: "error", text: explainError(error) });
    } finally {
      setSkillInstructionsLoading(false);
    }
  }

  async function saveSkillInstructions() {
    if (!skillInstructions) return;
    setBusy(true);
    setStatus({ tone: "loading", text: "正在保存技能说明…" });
    try {
      const payload = await api.saveSkillInstructions(skillInstructions.skill_id, skillInstructionsDraft);
      setSkillInstructions(payload);
      setSkillInstructionsDraft(payload.content);
      setStatus({ tone: "success", text: payload.message || "技能说明已保存" });
    } catch (error) {
      setStatus({ tone: "error", text: explainError(error) });
    } finally {
      setBusy(false);
    }
  }

  async function saveMeetingMinutesSettings(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setStatus({ tone: "loading", text: "正在保存会议纪要设置…" });
    try {
      const payload = await api.saveMeetingMinutesSettings(meetingMinutesSettingsForm);
      setMeetingMinutesSettings(payload);
      setMeetingMinutesSettingsForm({
        default_output_dir: payload.default_output_dir,
        custom_instructions: payload.custom_instructions
      });
      setStatus({ tone: "success", text: payload.message || "会议纪要设置已保存" });
    } catch (error) {
      setStatus({ tone: "error", text: explainError(error) });
    } finally {
      setBusy(false);
    }
  }

  async function refreshFiles(root = filesRoot) {
    setBusy(true);
    setStatus({ tone: "loading", text: "正在加载文件…" });
    try {
      const [nextFiles, nextMeetingArchives] = await Promise.all([
        api.files(root),
        api.meetingArchives()
      ]);
      setFiles(nextFiles.files);
      setMeetingArchives(nextMeetingArchives.meetings);
      setFilesRoot(root);
      setStatus({ tone: "success", text: `已加载 ${nextFiles.files.length} 个文件` });
    } catch (error) {
      setStatus({ tone: "error", text: explainError(error) });
    } finally {
      setBusy(false);
    }
  }

  async function openProject(projectId: string) {
    setBusy(true);
    setStatus({ tone: "loading", text: "正在打开项目…" });
    try {
      const payload = await api.project(projectId);
      setSelectedProject(payload.project);
      setProjectSettingsForm({
        name: payload.project.name,
        instructions: payload.project.instructions
      });
      setActiveProjectId(projectId);
      activeProjectIdRef.current = projectId;
      setProjectDetailTab("chat");
      setProjectChatDraft("");
      setProjectSettingsOpen(false);
      setViewWithUrl("projects");
      setStatus({ tone: "success", text: `已打开 ${payload.project.name}` });
    } catch (error) {
      setStatus({ tone: "error", text: explainError(error) });
    } finally {
      setBusy(false);
    }
  }

  function showProjectList() {
    setSelectedProject(null);
    setActiveProjectId(null);
    activeProjectIdRef.current = null;
    setProjectSettingsOpen(false);
    setViewWithUrl("projects");
  }

  async function createProject(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setStatus({ tone: "loading", text: "正在创建项目…" });
    try {
      const payload = await api.createProject(projectCreateForm);
      setProjects((items) => [payload.project, ...items]);
      setProjectCreateOpen(false);
      setProjectCreateForm({ name: "", instructions: "" });
      setSelectedProject(payload.project);
      setProjectSettingsForm({ name: payload.project.name, instructions: payload.project.instructions });
      setActiveProjectId(payload.project.id);
      activeProjectIdRef.current = payload.project.id;
      setProjectDetailTab("chat");
      setProjectChatDraft("");
      setViewWithUrl("projects");
      setStatus({ tone: "success", text: "项目已创建" });
    } catch (error) {
      setStatus({ tone: "error", text: explainError(error) });
    } finally {
      setBusy(false);
    }
  }

  async function saveProjectSettings(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedProject) return;
    setBusy(true);
    setStatus({ tone: "loading", text: "正在保存项目设置…" });
    try {
      const payload = await api.updateProject(selectedProject.id, projectSettingsForm);
      setSelectedProject(payload.project);
      setProjects((items) =>
        items.map((item) => (item.id === payload.project.id ? payload.project : item))
      );
      setProjectSettingsOpen(false);
      setStatus({ tone: "success", text: "项目设置已保存" });
    } catch (error) {
      setStatus({ tone: "error", text: explainError(error) });
    } finally {
      setBusy(false);
    }
  }

  function startProjectChatFromHome(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedProject) return;
    const draft = projectChatDraft.trim();
    startNewChat(selectedProject.id);
    setChatInput(draft);
    setProjectChatDraft("");
  }

  async function uploadProjectFiles(fileList: FileList | File[], projectId?: string) {
    const targetProjectId = projectId ?? selectedProject?.id ?? activeProjectIdRef.current;
    const nextFiles = Array.from(fileList);
    if (!targetProjectId || nextFiles.length === 0) return [];
    setBusy(true);
    setStatus({ tone: "loading", text: "正在加入项目资料…" });
    try {
      const uploaded: AttachmentItem[] = [];
      let latestProject: Project | null = null;
      for (const file of nextFiles) {
        const payload = await api.addProjectFile(targetProjectId, {
          name: file.name,
          mime_type: file.type || "application/octet-stream",
          content_base64: await fileToBase64(file)
        });
        uploaded.push(payload.attachment);
        latestProject = payload.project;
      }
      if (latestProject) {
        setSelectedProject((current) =>
          current?.id === latestProject?.id ? latestProject : current
        );
        setProjects((items) =>
          items.map((item) => (item.id === latestProject?.id ? latestProject! : item))
        );
      }
      setStatus({ tone: "success", text: `已加入 ${uploaded.length} 份项目资料` });
      return uploaded;
    } catch (error) {
      setStatus({ tone: "error", text: explainError(error) });
      return [];
    } finally {
      setBusy(false);
      setDragActive(false);
    }
  }

  async function deleteProjectFile(path: string) {
    if (!selectedProject) return;
    setBusy(true);
    setStatus({ tone: "loading", text: "正在移除项目资料…" });
    try {
      const payload = await api.deleteProjectFile(selectedProject.id, path);
      setSelectedProject(payload.project);
      setProjects((items) =>
        items.map((item) => (item.id === payload.project.id ? payload.project : item))
      );
      setStatus({ tone: "success", text: "项目资料已移除" });
    } catch (error) {
      setStatus({ tone: "error", text: explainError(error) });
    } finally {
      setBusy(false);
    }
  }

  async function syncActiveMeetingToProject() {
    if (!activeMeetingGroup?.manifestPath || !meetingSyncProjectId) return;
    const targetProject = projects.find((project) => project.id === meetingSyncProjectId);
    setBusy(true);
    setMeetingSyncMessage("");
    setStatus({ tone: "loading", text: "正在同步会议纪要…" });
    try {
      const payload = await api.syncMeetingToProject(
        meetingSyncProjectId,
        activeMeetingGroup.manifestPath
      );
      setProjects((items) =>
        items.map((item) => (item.id === payload.project.id ? payload.project : item))
      );
      setSelectedProject((current) =>
        current?.id === payload.project.id ? payload.project : current
      );
      const summary =
        payload.copied_count > 0
          ? `已同步 ${payload.copied_count} 份文件到“${targetProject?.name ?? payload.project.name}”`
          : `“${targetProject?.name ?? payload.project.name}”中的会议纪要已是最新`;
      setMeetingSyncMessage(summary);
      setStatus({ tone: "success", text: summary });
    } catch (error) {
      const message = explainError(error);
      setMeetingSyncMessage(message);
      setStatus({ tone: "error", text: message });
    } finally {
      setBusy(false);
    }
  }

  async function openFile(path: string) {
    setBusy(true);
    setStatus({ tone: "loading", text: "正在打开文件…" });
    try {
      const payload = await api.file(path);
      setSelectedFile(payload);
      setFileActionText("");
      setStatus({ tone: "success", text: "文件已打开" });
      return true;
    } catch (error) {
      setStatus({ tone: "error", text: explainError(error) });
      return false;
    } finally {
      setBusy(false);
    }
  }

  async function openLinkedFile(path: string) {
    await openFileInLibrary(path);
  }

  async function openFileInLibrary(path: string) {
    const opened = await openFile(path);
    if (!opened) return;
    setArtifactTabWithUrl("files");
  }

  function updateChatInput(value: string) {
    setChatInput(value);
    if (selectedSkill && !value.includes(selectedSkill.mention)) {
      setSelectedSkill(null);
    }
  }

  async function toggleVoiceInput() {
    if (isListening) {
      stopLocalVoiceRecording();
      return;
    }

    if (typeof navigator.mediaDevices?.getUserMedia !== "function" || typeof MediaRecorder === "undefined") {
      setSpeechSupported(false);
      setStatus({ tone: "error", text: "当前浏览器不支持本地录音，请用 Edge 或 Chrome 打开本地页面。" });
      return;
    }

    await startLocalVoiceRecording();
  }

  async function startLocalVoiceRecording() {
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });
    } catch (error) {
      setStatus({ tone: "error", text: explainMicrophonePermissionError(error) });
      return;
    }

    const options = preferredMediaRecorderOptions();
    let recorder: MediaRecorder;
    try {
      recorder = new MediaRecorder(stream, options);
    } catch (error) {
      stopVoiceStream(stream);
      setStatus({ tone: "error", text: explainRecorderStartError(error) });
      return;
    }

    voiceChunksRef.current = [];
    voiceStreamRef.current = stream;
    mediaRecorderRef.current = recorder;
    voiceShouldTranscribeRef.current = false;
    setActionMenuOpen(false);
    setIsListening(true);
    setVoiceLevels(createIdleVoiceLevels());
    setStatus({ tone: "loading", text: "正在录音，再次点击麦克风停止" });
    startVoiceLevelMeter(stream, setVoiceLevels, voiceAnimationRef, voiceAudioContextRef);

    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        voiceChunksRef.current.push(event.data);
      }
    };
    recorder.onerror = (event) => {
      console.warn("Local recorder error", event);
      voiceShouldTranscribeRef.current = false;
      setIsListening(false);
      stopVoiceLevelMeter(voiceAnimationRef, voiceAudioContextRef);
      stopVoiceStream(stream);
      voiceStreamRef.current = null;
      mediaRecorderRef.current = null;
      setStatus({ tone: "error", text: "录音失败，请检查浏览器和系统麦克风权限。" });
    };
    recorder.onstop = () => {
      const shouldTranscribe = voiceShouldTranscribeRef.current;
      voiceShouldTranscribeRef.current = false;
      if (shouldTranscribe) {
        void transcribeLocalVoice(recorder.mimeType || options?.mimeType || "audio/webm");
      } else {
        resetLocalVoiceCapture();
      }
    };

    try {
      recorder.start(1000);
    } catch (error) {
      stopVoiceStream(stream);
      stopVoiceLevelMeter(voiceAnimationRef, voiceAudioContextRef);
      setIsListening(false);
      setStatus({ tone: "error", text: explainRecorderStartError(error) });
    }
  }

  function stopLocalVoiceRecording() {
    const recorder = mediaRecorderRef.current;
    if (!recorder || recorder.state === "inactive") {
      resetLocalVoiceCapture();
      return;
    }
    voiceShouldTranscribeRef.current = true;
    setStatus({ tone: "loading", text: "正在结束录音…" });
    try {
      recorder.stop();
    } catch (error) {
      voiceShouldTranscribeRef.current = false;
      setIsListening(false);
      resetLocalVoiceCapture();
      setStatus({ tone: "error", text: explainRecorderStartError(error) });
    }
  }

  function cancelLocalVoiceRecording() {
    const recorder = mediaRecorderRef.current;
    voiceShouldTranscribeRef.current = false;
    if (recorder && recorder.state !== "inactive") {
      try {
        recorder.stop();
      } catch {
        resetLocalVoiceCapture();
      }
    } else {
      resetLocalVoiceCapture();
    }
    setStatus({ tone: "idle", text: "录音已取消" });
  }

  function resetLocalVoiceCapture() {
    voiceChunksRef.current = [];
    stopVoiceLevelMeter(voiceAnimationRef, voiceAudioContextRef);
    stopVoiceStream(voiceStreamRef.current);
    voiceStreamRef.current = null;
    mediaRecorderRef.current = null;
    setVoiceLevels(createIdleVoiceLevels());
    setIsListening(false);
  }

  async function transcribeLocalVoice(mimeType: string) {
    const chunks = voiceChunksRef.current;
    voiceChunksRef.current = [];
    stopVoiceLevelMeter(voiceAnimationRef, voiceAudioContextRef);
    stopVoiceStream(voiceStreamRef.current);
    voiceStreamRef.current = null;
    mediaRecorderRef.current = null;
    setVoiceLevels(createIdleVoiceLevels());
    setIsListening(false);

    if (chunks.length === 0) {
      setStatus({ tone: "error", text: "没有录到可转写的音频，请重新录制。" });
      return;
    }

    const blob = new Blob(chunks, { type: mimeType || "audio/webm" });
    if (blob.size === 0) {
      setStatus({ tone: "error", text: "录音内容为空，请重新录制。" });
      return;
    }

    setBusy(true);
    setStatus({ tone: "loading", text: "正在本地转写…" });
    try {
      const payload = await api.transcribeSpeech({
        name: `语音输入-${Date.now()}${extensionForMimeType(mimeType)}`,
        mime_type: mimeType || blob.type || "audio/webm",
        content_base64: await blobToBase64(blob)
      });
      const spokenText = payload.text.trim();
      if (!spokenText) {
        setStatus({ tone: "error", text: "本地 ASR 没有识别到文字，请靠近麦克风后再试。" });
        return;
      }
      setChatInput((value) => (value.trim() ? `${value.trimEnd()} ${spokenText}` : spokenText));
      setStatus({ tone: "success", text: "本地转写完成" });
    } catch (error) {
      setStatus({ tone: "error", text: explainError(error) });
    } finally {
      setBusy(false);
    }
  }

  async function startMeetingLiveTranscription() {
    if (meetingLiveStatus === "recording") return;
    if (typeof navigator.mediaDevices?.getUserMedia !== "function" || typeof MediaRecorder === "undefined") {
      setSpeechSupported(false);
      setMeetingLiveStatus("error");
      setStatus({ tone: "error", text: "当前浏览器不支持本地录音，请用 Edge 或 Chrome 打开本地页面。" });
      return;
    }

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          channelCount: 1
        }
      });
    } catch (error) {
      setMeetingLiveStatus("error");
      setStatus({ tone: "error", text: explainMicrophonePermissionError(error) });
      return;
    }

    const options = preferredMediaRecorderOptions();
    meetingLiveOptionsRef.current = options;
    meetingLiveStreamRef.current = stream;
    meetingLiveActiveRef.current = true;
    meetingLiveCurrentLevelRef.current = 0;
    meetingLiveNoiseFloorRef.current = realtimeTranscriptionNoiseFloorInitial;
    meetingLiveBackendVadActiveRef.current = false;
    meetingLiveBackendVadUnavailableRef.current = false;
    meetingLiveVadSampleBufferRef.current = new Float32Array(0);
    meetingLiveVadFrameBatchRef.current = [];
    meetingLiveVadRequestActiveRef.current = false;
    setMeetingLiveSavedPath("");
    setMeetingLiveStatus("recording");
    setMeetingLiveLevels(createIdleVoiceLevels());
    setStatus({ tone: "loading", text: "实时转写已开始，正在启用 WebRTC VAD" });
    startVoiceLevelMeter(
      stream,
      setMeetingLiveLevels,
      meetingLiveAnimationRef,
      meetingLiveAudioContextRef,
      meetingLiveCurrentLevelRef
    );
    startMeetingLiveBackendVad(stream);
    startMeetingLiveBoundaryLoop();
    startMeetingLiveSegment(stream);
  }

  function startMeetingLiveSegment(stream = meetingLiveStreamRef.current) {
    if (!stream || !meetingLiveActiveRef.current) return;
    const options = meetingLiveOptionsRef.current;
    let recorder: MediaRecorder;
    try {
      recorder = new MediaRecorder(stream, options);
    } catch (error) {
      meetingLiveActiveRef.current = false;
      cleanupMeetingLiveRecorder(stream);
      setMeetingLiveStatus("error");
      setStatus({ tone: "error", text: explainRecorderStartError(error) });
      return;
    }

    meetingLiveChunkPartsRef.current = [];
    meetingLiveSegmentStartedAtRef.current = Date.now();
    meetingLiveLastVoiceAtRef.current = Date.now();
    meetingLiveHasSpokenRef.current = false;
    meetingLiveSpeechFramesRef.current = 0;
    meetingLiveSpeechCandidateFramesRef.current = 0;
    meetingLiveSilenceFramesRef.current = 0;
    meetingLiveRecorderRef.current = recorder;
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        meetingLiveChunkPartsRef.current.push(event.data);
      }
    };
    recorder.onerror = (event) => {
      console.warn("Realtime recorder error", event);
      meetingLiveActiveRef.current = false;
      stopMeetingLiveBoundaryLoop();
      cleanupMeetingLiveRecorder(stream);
      setMeetingLiveStatus("error");
      setStatus({ tone: "error", text: "实时录音失败，请检查浏览器和系统麦克风权限。" });
    };
    recorder.onstop = () => {
      if (meetingLiveRecorderRef.current === recorder) {
        meetingLiveRecorderRef.current = null;
      }
      const parts = meetingLiveChunkPartsRef.current;
      meetingLiveChunkPartsRef.current = [];
      if (parts.length > 0) {
        const mimeType = recorder.mimeType || options?.mimeType || "audio/webm";
        const blob = new Blob(parts, { type: mimeType });
        void handleMeetingLiveChunk(blob, mimeType, meetingLiveSegmentStartedAtRef.current);
      }
      if (meetingLiveActiveRef.current && meetingLiveStreamRef.current) {
        window.setTimeout(() => startMeetingLiveSegment(), 40);
        return;
      }
      cleanupMeetingLiveRecorder(meetingLiveStreamRef.current);
      setMeetingLiveStatus(meetingLivePendingRef.current > 0 ? "processing" : "idle");
      setStatus({
        tone: meetingLivePendingRef.current > 0 ? "loading" : "success",
        text: meetingLivePendingRef.current > 0 ? "正在处理最后的音频片段…" : "实时转写已停止"
      });
    };

    try {
      recorder.start();
    } catch (error) {
      meetingLiveActiveRef.current = false;
      stopMeetingLiveBoundaryLoop();
      cleanupMeetingLiveRecorder(stream);
      setMeetingLiveStatus("error");
      setStatus({ tone: "error", text: explainRecorderStartError(error) });
    }
  }

  function startMeetingLiveBackendVad(stream: MediaStream) {
    stopMeetingLiveBackendVad();
    const AudioContextCtor =
      window.AudioContext ??
      (window as Window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AudioContextCtor) {
      meetingLiveBackendVadUnavailableRef.current = true;
      setStatus({ tone: "loading", text: "浏览器不支持音频帧采集，已回退到音量切段" });
      return;
    }

    try {
      const audioContext = new AudioContextCtor();
      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      processor.onaudioprocess = (event) => {
        const output = event.outputBuffer.getChannelData(0);
        output.fill(0);
        if (!meetingLiveActiveRef.current || meetingLiveBackendVadUnavailableRef.current) return;
        const input = event.inputBuffer.getChannelData(0);
        queueMeetingLiveVadSamples(input, audioContext.sampleRate);
      };
      source.connect(processor);
      processor.connect(audioContext.destination);
      meetingLiveVadAudioContextRef.current = audioContext;
      meetingLiveVadSourceRef.current = source;
      meetingLiveVadProcessorRef.current = processor;
      void audioContext.resume();
    } catch (error) {
      meetingLiveBackendVadUnavailableRef.current = true;
      setStatus({ tone: "loading", text: `WebRTC VAD 启动失败，已回退到音量切段：${explainError(error)}` });
    }
  }

  function queueMeetingLiveVadSamples(input: Float32Array, inputSampleRate: number) {
    const previous = meetingLiveVadSampleBufferRef.current;
    const merged = new Float32Array(previous.length + input.length);
    merged.set(previous);
    merged.set(input, previous.length);

    const targetSampleRate = realtimeTranscriptionBackendVadSampleRate;
    const targetFrameSamples = Math.round(
      (targetSampleRate * realtimeTranscriptionBackendVadFrameMs) / 1000
    );
    const ratio = inputSampleRate / targetSampleRate;
    const neededInputSamples = Math.ceil((targetFrameSamples + 1) * ratio);
    let offset = 0;

    while (merged.length - offset >= neededInputSamples) {
      const frame = new Int16Array(targetFrameSamples);
      for (let index = 0; index < targetFrameSamples; index += 1) {
        const sourceIndex = offset + index * ratio;
        const leftIndex = Math.floor(sourceIndex);
        const rightIndex = Math.min(merged.length - 1, leftIndex + 1);
        const fraction = sourceIndex - leftIndex;
        const sample = merged[leftIndex] * (1 - fraction) + merged[rightIndex] * fraction;
        const clipped = Math.max(-1, Math.min(1, sample));
        frame[index] = clipped < 0 ? clipped * 0x8000 : clipped * 0x7fff;
      }
      queueMeetingLiveVadFrame(int16FrameToBase64(frame));
      offset += Math.floor(targetFrameSamples * ratio);
    }

    meetingLiveVadSampleBufferRef.current = merged.slice(offset);
  }

  function queueMeetingLiveVadFrame(frameBase64: string) {
    if (!meetingLiveActiveRef.current || meetingLiveBackendVadUnavailableRef.current) return;
    const batch = meetingLiveVadFrameBatchRef.current;
    batch.push(frameBase64);
    if (batch.length > realtimeTranscriptionBackendVadBatchFrames * 10) {
      batch.splice(0, batch.length - realtimeTranscriptionBackendVadBatchFrames * 10);
    }
    if (batch.length >= realtimeTranscriptionBackendVadBatchFrames) {
      void flushMeetingLiveVadBatch();
    }
  }

  async function flushMeetingLiveVadBatch() {
    if (
      meetingLiveVadRequestActiveRef.current ||
      meetingLiveBackendVadUnavailableRef.current ||
      !meetingLiveActiveRef.current
    ) {
      return;
    }
    const batch = meetingLiveVadFrameBatchRef.current.splice(
      0,
      realtimeTranscriptionBackendVadBatchFrames
    );
    if (batch.length === 0) return;

    meetingLiveVadRequestActiveRef.current = true;
    try {
      const payload = await api.detectSpeechVad({
        sample_rate: realtimeTranscriptionBackendVadSampleRate,
        frame_ms: realtimeTranscriptionBackendVadFrameMs,
        aggressiveness: 3,
        frames_base64: batch
      });
      if (!payload.available) {
        meetingLiveBackendVadActiveRef.current = false;
        meetingLiveBackendVadUnavailableRef.current = true;
        setStatus({
          tone: "loading",
          text: payload.error
            ? `WebRTC VAD 不可用，已回退到音量切段：${payload.error}`
            : "WebRTC VAD 不可用，已回退到音量切段"
        });
        return;
      }
      meetingLiveBackendVadActiveRef.current = true;
      applyMeetingLiveVadFrames(payload.speech_frames);
    } catch (error) {
      meetingLiveBackendVadActiveRef.current = false;
      meetingLiveBackendVadUnavailableRef.current = true;
      setStatus({ tone: "loading", text: `WebRTC VAD 请求失败，已回退到音量切段：${explainError(error)}` });
    } finally {
      meetingLiveVadRequestActiveRef.current = false;
      if (
        meetingLiveActiveRef.current &&
        !meetingLiveBackendVadUnavailableRef.current &&
        meetingLiveVadFrameBatchRef.current.length >= realtimeTranscriptionBackendVadBatchFrames
      ) {
        void flushMeetingLiveVadBatch();
      }
    }
  }

  function applyMeetingLiveVadFrames(speechFrames: boolean[]) {
    const recorder = meetingLiveRecorderRef.current;
    if (!meetingLiveActiveRef.current || !recorder || recorder.state !== "recording") return;
    const now = Date.now();
    for (const isSpeech of speechFrames) {
      if (isSpeech) {
        meetingLiveSpeechCandidateFramesRef.current += 1;
        if (meetingLiveSpeechCandidateFramesRef.current >= realtimeTranscriptionStartSpeechFrames) {
          meetingLiveHasSpokenRef.current = true;
          meetingLiveSpeechFramesRef.current += 1;
          meetingLiveSilenceFramesRef.current = 0;
          meetingLiveLastVoiceAtRef.current = now;
        }
      } else {
        meetingLiveSpeechCandidateFramesRef.current = 0;
        if (meetingLiveHasSpokenRef.current) {
          meetingLiveSilenceFramesRef.current += 1;
        }
      }
    }
  }

  function stopMeetingLiveBackendVad() {
    meetingLiveBackendVadActiveRef.current = false;
    meetingLiveVadSampleBufferRef.current = new Float32Array(0);
    meetingLiveVadFrameBatchRef.current = [];
    meetingLiveVadRequestActiveRef.current = false;
    const processor = meetingLiveVadProcessorRef.current;
    const source = meetingLiveVadSourceRef.current;
    const audioContext = meetingLiveVadAudioContextRef.current;
    meetingLiveVadProcessorRef.current = null;
    meetingLiveVadSourceRef.current = null;
    meetingLiveVadAudioContextRef.current = null;
    if (processor) {
      processor.onaudioprocess = null;
      try {
        processor.disconnect();
      } catch {
        // Already disconnected.
      }
    }
    if (source) {
      try {
        source.disconnect();
      } catch {
        // Already disconnected.
      }
    }
    if (audioContext && audioContext.state !== "closed") {
      void audioContext.close();
    }
  }

  function startMeetingLiveBoundaryLoop() {
    stopMeetingLiveBoundaryLoop();
    meetingLiveBoundaryTimerRef.current = window.setInterval(() => {
      const recorder = meetingLiveRecorderRef.current;
      if (!meetingLiveActiveRef.current || !recorder || recorder.state !== "recording") return;
      const now = Date.now();
      const useBackendVad = meetingLiveBackendVadActiveRef.current;
      if (!useBackendVad) {
        const currentLevel = meetingLiveCurrentLevelRef.current;
        const voiceThreshold = realtimeTranscriptionVoiceThreshold(meetingLiveNoiseFloorRef.current);
        const isSpeechCandidate = currentLevel >= voiceThreshold;
        if (isSpeechCandidate) {
          meetingLiveSpeechCandidateFramesRef.current += 1;
          if (meetingLiveSpeechCandidateFramesRef.current >= realtimeTranscriptionStartSpeechFrames) {
            meetingLiveHasSpokenRef.current = true;
            meetingLiveSpeechFramesRef.current += 1;
            meetingLiveSilenceFramesRef.current = 0;
            meetingLiveLastVoiceAtRef.current = now;
          }
        } else {
          meetingLiveSpeechCandidateFramesRef.current = 0;
          meetingLiveNoiseFloorRef.current = updateRealtimeNoiseFloor(
            meetingLiveNoiseFloorRef.current,
            currentLevel
          );
          if (meetingLiveHasSpokenRef.current) {
            meetingLiveSilenceFramesRef.current += 1;
          }
        }
      }
      const elapsed = now - meetingLiveSegmentStartedAtRef.current;
      const silenceElapsed =
        meetingLiveSilenceFramesRef.current *
        (useBackendVad ? realtimeTranscriptionBackendVadFrameMs : realtimeTranscriptionVadFrameMs);
      if (elapsed >= realtimeTranscriptionMaxSegmentMs) {
        finishMeetingLiveSegment("max");
        return;
      }
      if (
        elapsed >= realtimeTranscriptionMinSegmentMs &&
        meetingLiveHasSpokenRef.current &&
        silenceElapsed >= realtimeTranscriptionPauseMs
      ) {
        finishMeetingLiveSegment("pause");
      }
    }, realtimeTranscriptionVadFrameMs);
  }

  function stopMeetingLiveBoundaryLoop() {
    if (meetingLiveBoundaryTimerRef.current !== null) {
      window.clearInterval(meetingLiveBoundaryTimerRef.current);
      meetingLiveBoundaryTimerRef.current = null;
    }
  }

  function finishMeetingLiveSegment(reason: "pause" | "max") {
    const recorder = meetingLiveRecorderRef.current;
    if (!recorder || recorder.state !== "recording") return;
    setStatus({
      tone: "loading",
      text: reason === "max" ? "当前片段较长，正在切段识别…" : "检测到语气停顿，正在切段识别…"
    });
    try {
      recorder.stop();
    } catch (error) {
      setStatus({ tone: "error", text: explainRecorderStartError(error) });
    }
  }

  function stopMeetingLiveTranscription() {
    const recorder = meetingLiveRecorderRef.current;
    meetingLiveActiveRef.current = false;
    stopMeetingLiveBoundaryLoop();
    if (!recorder || recorder.state === "inactive") {
      cleanupMeetingLiveRecorder(meetingLiveStreamRef.current);
      setMeetingLiveStatus(meetingLivePendingRef.current > 0 ? "processing" : "idle");
      return;
    }
    setMeetingLiveStatus("processing");
    setStatus({ tone: "loading", text: "正在结束实时转写…" });
    try {
      recorder.stop();
    } catch (error) {
      cleanupMeetingLiveRecorder(meetingLiveStreamRef.current);
      setMeetingLiveStatus("error");
      setStatus({ tone: "error", text: explainRecorderStartError(error) });
    }
  }

  function cleanupMeetingLiveRecorder(stream: MediaStream | null) {
    stopMeetingLiveBoundaryLoop();
    stopVoiceLevelMeter(meetingLiveAnimationRef, meetingLiveAudioContextRef);
    stopMeetingLiveBackendVad();
    stopVoiceStream(stream);
    meetingLiveStreamRef.current = null;
    meetingLiveRecorderRef.current = null;
    meetingLiveCurrentLevelRef.current = 0;
    setMeetingLiveLevels(createIdleVoiceLevels());
  }

  function updateMeetingLivePending(delta: number) {
    const next = Math.max(0, meetingLivePendingRef.current + delta);
    meetingLivePendingRef.current = next;
    setMeetingLivePending(next);
    return next;
  }

  async function handleMeetingLiveChunk(blob: Blob, mimeType: string, startedAt = Date.now()) {
    if (blob.size < 512) return;
    const index = meetingLiveChunkIndexRef.current + 1;
    meetingLiveChunkIndexRef.current = index;
    const now = Date.now();
    const segmentId = `live-${now}-${index}`;
    setMeetingLiveSegments((items) => [
      ...items,
      {
        id: segmentId,
        index,
        text: "",
        startedAt,
        finishedAt: 0,
        pending: true
      }
    ]);
    updateMeetingLivePending(1);
    const run = async () => {
      try {
        const payload = await api.transcribeSpeech({
          name: `实时转写-${index}-${now}${extensionForMimeType(mimeType)}`,
          mime_type: mimeType || blob.type || "audio/webm",
          content_base64: await blobToBase64(blob),
          use_denoise: true,
          skip_if_silent: true
        });
        const text = payload.text.trim();
        if (!text) {
          setMeetingLiveSegments((items) => items.filter((item) => item.id !== segmentId));
          if (!payload.skipped) {
            setStatus({ tone: "idle", text: `第 ${index} 段未识别到文字` });
          }
          return;
        }
        setMeetingLiveSegments((items) =>
          items.map((item) =>
            item.id === segmentId
              ? {
                  ...item,
                  text,
                  pending: false,
                  finishedAt: Date.now(),
                  audioPath: payload.audio_path,
                  transcriptPath: payload.transcript_path,
                  asrElapsedMs: payload.asr_elapsed_ms
                }
              : item
          )
        );
        setStatus({ tone: "success", text: `已追加第 ${index} 段转写` });
      } catch (error) {
        setMeetingLiveSegments((items) =>
          items.map((item) =>
            item.id === segmentId
              ? {
                  ...item,
                  pending: false,
                  finishedAt: Date.now(),
                  error: explainError(error)
                }
              : item
          )
        );
        if (!meetingLiveActiveRef.current) {
          setMeetingLiveStatus("error");
        }
        setStatus({ tone: "error", text: explainError(error) });
      } finally {
        const pending = updateMeetingLivePending(-1);
        if (!meetingLiveActiveRef.current && pending === 0) {
          setMeetingLiveStatus((current) => (current === "error" ? "error" : "idle"));
        }
      }
    };
    const queued = meetingLiveTranscriptionQueueRef.current.catch(() => undefined).then(run);
    meetingLiveTranscriptionQueueRef.current = queued.catch(() => undefined);
    await queued;
  }

  async function saveMeetingLiveTranscript() {
    const completedSegments = meetingLiveSegments.filter(
      (segment) => segment.text.trim() && !segment.pending && !segment.error
    );
    if (completedSegments.length === 0) {
      setStatus({ tone: "error", text: "还没有可保存的实时转写内容。" });
      return;
    }
    setBusy(true);
    setStatus({ tone: "loading", text: "正在保存实时转写稿…" });
    try {
      const payload = await api.saveRealtimeTranscript({
        title: meetingLiveTitle.trim() || "实时会议转写",
        segments: completedSegments.map((segment) => ({
          index: segment.index,
          text: segment.text,
          started_at: segment.startedAt,
          finished_at: segment.finishedAt
        }))
      });
      setMeetingLiveSavedPath(payload.path);
      await refreshFiles(filesRoot || "meet_files");
      setStatus({ tone: "success", text: `实时转写稿已保存：${payload.path}` });
    } catch (error) {
      setStatus({ tone: "error", text: explainError(error) });
    } finally {
      setBusy(false);
    }
  }

  function clearMeetingLiveTranscript() {
    if (meetingLiveStatus === "recording") return;
    meetingLiveChunkIndexRef.current = 0;
    meetingLivePendingRef.current = 0;
    setMeetingLivePending(0);
    setMeetingLiveSegments([]);
    setMeetingLiveSavedPath("");
    setMeetingLiveStatus("idle");
    setStatus({ tone: "idle", text: "实时转写内容已清空" });
  }

  function sendMeetingTranscriptToAgent() {
    if (!meetingLiveSavedPath) return;
    setChatInput(`请基于这份实时会议转写稿整理会议内容：${meetingLiveSavedPath}`);
    setViewWithUrl("agent");
  }

  function attachSkill(skill: SkillInfo) {
    setSelectedSkill(skill);
    setActionMenuOpen(false);
    setChatInput((value) => {
      if (value.includes(skill.mention)) return value;
      const trimmed = value.trimEnd();
      return trimmed ? `${trimmed} ${skill.mention} ` : `${skill.mention} `;
    });
  }

  function attachSkillById(id: string) {
    const skill = skills.find((item) => item.id === id);
    if (skill) {
      attachSkill(skill);
    }
  }

  function scheduleChatScrollToBottom(attempt = 0) {
    window.requestAnimationFrame(() => {
      const container = chatThreadRef.current;
      if (container) {
        container.scrollTop = container.scrollHeight;
      }
      if (attempt < 2) {
        window.setTimeout(() => scheduleChatScrollToBottom(attempt + 1), attempt === 0 ? 40 : 120);
      }
    });
  }

  function requestChatScrollToBottom() {
    pendingChatScrollToBottomRef.current = true;
    window.setTimeout(() => {
      if (!pendingChatScrollToBottomRef.current || view !== "agent") return;
      pendingChatScrollToBottomRef.current = false;
      scheduleChatScrollToBottom();
    }, 0);
  }

  function startNewChat(projectId: string | null = null) {
    requestChatScrollToBottom();
    const conversationId = createConversationId();
    currentConversationIdRef.current = conversationId;
    setCurrentConversationId(conversationId);
    setActiveProjectId(projectId);
    activeProjectIdRef.current = projectId;
    setConversationSummary("");
    setConversationSummaryMessageCount(0);
    setHistoryMenu(null);
    setRenamingConversationId(null);
    setDeleteConfirmConversationId(null);
    cancelEditingUserMessage();
    const initialMessages: ChatMessage[] = [
      {
        role: "assistant",
        content: projectId
          ? `已进入项目“${selectedProject?.name ?? "当前项目"}”。项目资料会在需要时由智能体自动读取。`
          : "新聊天已开始。你可以拖入录音或材料，也可以直接说要整理哪份会议。"
      }
    ];
    chatMessagesRef.current = initialMessages;
    setChatMessages(initialMessages);
    setAttachments([]);
    setSelectedSkill(null);
    setActionMenuOpen(false);
    setActivityRecords({});
    setActivityPanelMessageIndex(null);
    setActivityEvents([]);
    setActivityOpen(false);
    setActivityMessageIndex(null);
    setActivityStartedAt(null);
    activityRunningRef.current = false;
    setActivityRunning(false);
    setActivityElapsedMs(0);
    setQueuedChatCount(0);
    setBusy(false);
    setStatus({ tone: "idle", text: "就绪" });
    setSearchOpen(false);
    setViewWithUrl("agent");
  }

  function openConversation(item: ConversationHistoryItem) {
    requestChatScrollToBottom();
    const restoredActivityRecords = item.activities ?? {};
    const restoredActivityIndex =
      typeof item.activeActivityIndex === "number" && restoredActivityRecords[item.activeActivityIndex]
        ? item.activeActivityIndex
        : lastActivityRecordIndex(restoredActivityRecords);
    const restoredActivityRecord =
      restoredActivityIndex === null ? null : restoredActivityRecords[restoredActivityIndex] ?? null;
    const activeRun = activeChatRunsRef.current.get(item.id);
    currentConversationIdRef.current = item.id;
    setCurrentConversationId(item.id);
    setActiveProjectId(item.projectId ?? null);
    activeProjectIdRef.current = item.projectId ?? null;
    if (item.projectId && selectedProject?.id !== item.projectId) {
      void api.project(item.projectId).then((payload) => setSelectedProject(payload.project));
    }
    setConversationSummary(item.contextSummary ?? "");
    setConversationSummaryMessageCount(item.contextSummaryMessageCount ?? 0);
    setHistoryMenu(null);
    setRenamingConversationId(null);
    setDeleteConfirmConversationId(null);
    cancelEditingUserMessage();
    setSearchOpen(false);
    setConversationSearch("");
    setActivityRecords(restoredActivityRecords);
    setActivityPanelMessageIndex(restoredActivityIndex);
    setActivityEvents(restoredActivityRecord?.events ?? []);
    setActivityOpen(false);
    setActivityMessageIndex(restoredActivityIndex);
    setActivityStartedAt(activeRun?.startedAt ?? null);
    activityRunningRef.current = Boolean(activeRun);
    setActivityRunning(Boolean(activeRun));
    setActivityElapsedMs(restoredActivityRecord?.elapsedMs ?? 0);
    setQueuedChatCount(queuedChatMessagesRef.current.get(item.id)?.length ?? 0);
    setBusy(Boolean(activeRun));
    setViewWithUrl("agent");
    chatMessagesRef.current = item.messages;
    setChatMessages(item.messages);
    setStatus(
      activeRun
        ? { tone: "loading", text: "该对话仍在后台处理中…" }
        : { tone: "idle", text: "已打开对话" }
    );
  }

  function openHistoryMenu(event: MouseEvent<HTMLButtonElement>, id: string) {
    event.preventDefault();
    event.stopPropagation();
    const rect = event.currentTarget.getBoundingClientRect();
    const menuWidth = 184;
    const menuHeight = 146;
    const x = Math.min(
      Math.max(12, rect.right - menuWidth),
      Math.max(12, window.innerWidth - menuWidth - 12)
    );
    const y = Math.min(rect.bottom + 8, Math.max(12, window.innerHeight - menuHeight - 12));
    setHistoryMenu({ id, x, y });
    setDeleteConfirmConversationId(null);
  }

  function toggleConversationPin(id: string) {
    let pinned = false;
    setConversationHistory((items) =>
      orderConversationHistory(
        items.map((item) => {
          if (item.id !== id) return item;
          pinned = !item.pinned;
          return { ...item, pinned };
        })
      )
    );
    setHistoryMenu(null);
    setDeleteConfirmConversationId(null);
    setStatus({ tone: "success", text: pinned ? "已置顶对话" : "已取消置顶" });
  }

  function startRenameConversation(id: string) {
    const item = conversationHistory.find((conversation) => conversation.id === id);
    if (!item) return;
    setRenamingConversationId(id);
    setRenameDraft(item.title);
    setHistoryMenu(null);
    setDeleteConfirmConversationId(null);
  }

  function commitRenameConversation(id: string) {
    const title = renameDraft.replace(/\s+/g, " ").trim();
    setRenamingConversationId(null);
    setRenameDraft("");
    if (!title) return;
    setConversationHistory((items) =>
      items.map((item) => (item.id === id ? { ...item, title } : item))
    );
    setStatus({ tone: "success", text: "对话已重命名" });
  }

  function deleteConversation(id: string) {
    if (deleteConfirmConversationId !== id) {
      setDeleteConfirmConversationId(id);
      return;
    }
    setConversationHistory((items) => items.filter((item) => item.id !== id));
    setHistoryMenu(null);
    setDeleteConfirmConversationId(null);
    if (currentConversationId === id) {
      startNewChat();
    }
    setStatus({ tone: "success", text: "历史对话已删除" });
  }

  async function restoreConversationArchive() {
    try {
      const payload = await api.conversations();
      const archivedItems = payload.items
        .filter(isConversationHistoryItem)
        .map(sanitizeConversationHistoryItem);
      if (archivedItems.length > 0) {
        setConversationHistory((items) => mergeConversationHistories(archivedItems, items));
      }
    } catch {
      // Keep the browser-local copy when the workspace archive is not available.
    } finally {
      conversationArchiveReadyRef.current = true;
    }
  }

  async function attachDroppedFiles(fileList: FileList | File[]) {
    const nextFiles = Array.from(fileList);
    if (nextFiles.length === 0) return;
    if (activeProjectIdRef.current) {
      const uploaded = await uploadProjectFiles(nextFiles, activeProjectIdRef.current);
      setAttachments((items) => mergeAttachmentsByPath(items, uploaded));
      return;
    }
    setBusy(true);
    setStatus({ tone: "loading", text: "正在保存附件…" });
    try {
      const uploaded: AttachmentItem[] = [];
      for (const file of nextFiles) {
        const fileWithLocalHints = file as File & {
          path?: string;
          webkitRelativePath?: string;
        };
        const payload = await api.addAttachment({
          name: file.name,
          mime_type: file.type || "application/octet-stream",
          size: file.size,
          last_modified: file.lastModified,
          relative_path: fileWithLocalHints.webkitRelativePath || "",
          source_path: fileWithLocalHints.path || "",
          content_base64: await fileToBase64(file)
        });
        uploaded.push(payload.attachment);
      }
      setAttachments((items) => mergeAttachmentsByPath(items, uploaded));
      const [nextLibrary, nextMeetingArchives] = await Promise.all([
        api.files(filesRoot || "meet_files"),
        api.meetingArchives()
      ]);
      setFiles(nextLibrary.files);
      setFilesRoot(nextLibrary.root);
      setMeetingArchives(nextMeetingArchives.meetings);
      const deduplicatedCount = uploaded.filter((item) => item.deduplicated).length;
      setStatus({
        tone: "success",
        text:
          deduplicatedCount > 0
            ? `已处理 ${uploaded.length} 个附件，${deduplicatedCount} 个已存在`
            : `已添加 ${uploaded.length} 个附件`
      });
    } catch (error) {
      setStatus({ tone: "error", text: explainError(error) });
    } finally {
      setBusy(false);
      setDragActive(false);
    }
  }

  function removeAttachment(path: string) {
    setAttachments((items) => items.filter((item) => item.path !== path));
  }

  function handleAppDragEnter(event: DragEvent<HTMLDivElement>) {
    if (!hasDraggedFiles(event)) return;
    event.preventDefault();
    setDragActive(true);
  }

  function handleAppDragOver(event: DragEvent<HTMLDivElement>) {
    if (!hasDraggedFiles(event)) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
    setDragActive(true);
  }

  function handleAppDragLeave(event: DragEvent<HTMLDivElement>) {
    if (event.currentTarget.contains(event.relatedTarget as Node | null)) return;
    setDragActive(false);
  }

  function handleAppDrop(event: DragEvent<HTMLDivElement>) {
    if (!hasDraggedFiles(event)) return;
    event.preventDefault();
    setDragActive(false);
    void attachDroppedFiles(event.dataTransfer.files);
  }

  function startEditingUserMessage(index: number) {
    if (activityRunningRef.current) return;
    const message = chatMessagesRef.current[index];
    if (!message || message.role !== "user") return;
    setEditingMessageIndex(index);
    setEditMessageDraft(message.content);
  }

  function cancelEditingUserMessage() {
    setEditingMessageIndex(null);
    setEditMessageDraft("");
  }

  async function resendEditedUserMessage(event: FormEvent<HTMLFormElement>, index: number) {
    event.preventDefault();
    if (activityRunningRef.current) return;
    const content = editMessageDraft.trim();
    if (!content) return;
    const currentMessages = chatMessagesRef.current;
    if (currentMessages[index]?.role !== "user") return;

    const rewindUserMessageOrdinal = currentMessages
      .slice(0, index)
      .filter((message) => message.role === "user").length;
    const baseMessages = currentMessages.slice(0, index);
    const baseActivityRecords = Object.fromEntries(
      Object.entries(activityRecords).filter(([messageIndex]) => Number(messageIndex) < index)
    ) as ActivityRecordMap;
    const previousActivityIndex = lastActivityRecordIndex(baseActivityRecords);
    const previousActivityRecord =
      previousActivityIndex === null ? null : baseActivityRecords[previousActivityIndex] ?? null;

    cancelEditingUserMessage();
    setConversationSummary("");
    setConversationSummaryMessageCount(0);
    setActivityRecords(baseActivityRecords);
    setActivityPanelMessageIndex(previousActivityIndex);
    setActivityEvents(previousActivityRecord?.events ?? []);
    setActivityMessageIndex(previousActivityIndex);
    setActivityElapsedMs(previousActivityRecord?.elapsedMs ?? 0);
    setActivityOpen(false);
    chatMessagesRef.current = baseMessages;
    setChatMessages(baseMessages);

    await runChatMessage(
      {
        content,
        attachments: [],
        skill: inferSkillFromText(content, skills)
      },
      {
        baseMessages,
        activityRecords: baseActivityRecords,
        conversationSummary: "",
        conversationSummaryMessageCount: 0,
        rewindUserMessageOrdinal
      }
    );
  }

  async function continueFailedTurn() {
    if (activityRunningRef.current) return;
    const content = "继续刚才未完成的任务。请复用本轮已经成功的工具结果，不要重复读取已经成功读取的文件；完成剩余工作后直接给出最终结果。";
    await runChatMessage({
      content,
      attachments: [],
      skill: inferSkillFromText(chatMessagesRef.current.map((item) => item.content).join("\n"), skills)
    });
  }

  async function sendChatMessage(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const content = chatInput.trim();
    const queuedItem = {
      content,
      attachments,
      skill: selectedSkill ?? inferSkillFromText(content, skills)
    };
    if (!queuedItem.content && queuedItem.attachments.length === 0) return;
    setChatInput("");
    setSelectedSkill(null);
    setAttachments([]);
    const conversationId = currentConversationIdRef.current;
    if (activeChatRunsRef.current.has(conversationId)) {
      const queue = [...(queuedChatMessagesRef.current.get(conversationId) ?? []), queuedItem];
      queuedChatMessagesRef.current.set(conversationId, queue);
      setQueuedChatCount(queue.length);
      setStatus({
        tone: "loading",
        text: `当前轮仍在运行，已加入队列（${queue.length} 条等待）`
      });
      return;
    }
    await runChatMessage(queuedItem, { conversationId });
  }

  async function runChatMessage(
    queuedItem: QueuedChatItem,
    options: RunChatMessageOptions = {}
  ) {
    const content = queuedItem.content.trim();
    if (!content && queuedItem.attachments.length === 0) return;
    const outgoingSkill = queuedItem.skill ?? inferSkillFromText(content, skills);
    const messageContent = formatMessageWithAttachments(content, queuedItem.attachments);
    const conversationId = options.conversationId ?? currentConversationIdRef.current;
    const projectId =
      conversationHistoryRef.current.find((item) => item.id === conversationId)?.projectId ??
      activeProjectIdRef.current ??
      undefined;
    const isVisibleConversation = () => currentConversationIdRef.current === conversationId;
    const baseMessages = options.baseMessages ?? chatMessagesRef.current;
    const baseActivityRecords = options.activityRecords ?? activityRecords;
    const baseConversationSummary = options.conversationSummary ?? conversationSummary;
    const baseConversationSummaryMessageCount =
      options.conversationSummaryMessageCount ?? conversationSummaryMessageCount;
    const nextMessages: ChatMessage[] = [...baseMessages, { role: "user", content: messageContent }];
    const contextFilePaths = collectConversationFileReferences(
      nextMessages,
      queuedItem.attachments,
      baseActivityRecords
    );
    const assistantIndex = nextMessages.length;
    const shouldNameConversation =
      !conversationHistory.some((item) => item.id === conversationId) &&
      baseMessages.every((message) => message.role !== "user");
    const activityStartedAtMs = Date.now();
    let streamedContent = "";
    let finalReply = "";
    let assistantDraftContent = "";
    let waitingApprovalDraft = false;
    let lastDraftPersistAt = 0;
    let contextSummaryDraft = baseConversationSummary;
    let contextSummaryMessageCountDraft = baseConversationSummaryMessageCount;
    let activityEventsDraft: AgentActivityEvent[] = [];
    let activityElapsedDraft = 0;
    let activityRecordsDraft: ActivityRecordMap = {
      ...baseActivityRecords,
      [assistantIndex]: { events: [], elapsedMs: 0, completed: false }
    };
    const syncActivityRecord = (patch: Partial<ActivityRecord>) => {
      const previous = activityRecordsDraft[assistantIndex] ?? {
        events: activityEventsDraft,
        elapsedMs: activityElapsedDraft,
        completed: false
      };
      const nextRecord: ActivityRecord = {
        ...previous,
        ...patch
      };
      activityRecordsDraft = {
        ...activityRecordsDraft,
        [assistantIndex]: nextRecord
      };
      if (isVisibleConversation()) setActivityRecords(activityRecordsDraft);
      return activityRecordsDraft;
    };
    const persistDraftConversation = (options?: { force?: boolean; completed?: boolean }) => {
      const now = Date.now();
      if (!options?.force && now - lastDraftPersistAt < 1500) return;
      lastDraftPersistAt = now;
      const snapshotMessages: ChatMessage[] = [
        ...nextMessages,
        {
          role: "assistant",
          content: assistantDraftContent
        }
      ];
      const snapshotActivities = options?.completed
        ? {
            ...activityRecordsDraft,
            [assistantIndex]: {
              ...(activityRecordsDraft[assistantIndex] ?? {
                events: activityEventsDraft,
                elapsedMs: activityElapsedDraft,
                completed: false
              }),
              completed: true
            }
          }
        : activityRecordsDraft;
      const fallbackTitle = shouldNameConversation
        ? pendingConversationTitle
        : titleFromMessages(snapshotMessages);
      setConversationHistory((items) => {
        const next = updateConversationMessages(
          items,
          conversationId,
          snapshotMessages,
          snapshotActivities,
          assistantIndex,
          fallbackTitle,
          contextSummaryDraft,
          contextSummaryMessageCountDraft,
          projectId
        );
        conversationHistoryRef.current = next;
        saveConversationHistory(next, currentUser?.username);
        return next;
      });
    };
    const updateDraftActivityElapsed = (elapsedMs: number) => {
      activityElapsedDraft = Math.max(activityElapsedDraft, elapsedMs);
      if (isVisibleConversation()) setActivityElapsedMs(activityElapsedDraft);
      syncActivityRecord({ elapsedMs: activityElapsedDraft });
    };
    const appendDraftActivity = (item: AgentActivityEvent) => {
      activityEventsDraft = [...activityEventsDraft, item];
      if (isVisibleConversation()) setActivityEvents(activityEventsDraft);
      syncActivityRecord({
        events: activityEventsDraft,
        elapsedMs: activityElapsedDraft,
        completed: false
      });
      persistDraftConversation({ force: true });
    };
    const appendDraftActivityDelta = (
      item: Extract<AgentStreamEvent, { event: "activity_delta" }>
    ) => {
      activityEventsDraft = appendActivityDelta(activityEventsDraft, item);
      if (isVisibleConversation()) setActivityEvents(activityEventsDraft);
      syncActivityRecord({
        events: activityEventsDraft,
        elapsedMs: activityElapsedDraft,
        completed: false
      });
      persistDraftConversation();
    };
    const completeDraftActivity = () => {
      activityElapsedDraft = Math.max(activityElapsedDraft, Date.now() - activityStartedAtMs, 1);
      if (isVisibleConversation()) setActivityElapsedMs(activityElapsedDraft);
      return syncActivityRecord({
        events: activityEventsDraft,
        elapsedMs: activityElapsedDraft,
        completed: true
      });
    };
    const abortController = new AbortController();
    activeChatRunsRef.current.set(conversationId, {
      abortController,
      turnId: null,
      assistantIndex,
      startedAt: activityStartedAtMs
    });
    if (isVisibleConversation()) {
      setChatMessages([...nextMessages, { role: "assistant", content: "" }]);
      setActivityRecords(activityRecordsDraft);
      setActivityPanelMessageIndex(assistantIndex);
      setActivityEvents([]);
      setActivityElapsedMs(0);
      setActivityOpen(false);
      setActivityMessageIndex(assistantIndex);
      setActivityStartedAt(activityStartedAtMs);
      setActivityNow(activityStartedAtMs);
      activityRunningRef.current = true;
      setActivityRunning(true);
      setBusy(true);
      setStatus({ tone: "loading", text: "智能体处理中…" });
    }
    persistDraftConversation({ force: true });
    try {
      await api.streamChatAgent(
        {
          conversation_id: conversationId,
          project_id: projectId,
          messages: nextMessages,
          profile: agentForm.profile,
          reasoning_effort: reasoningEffort,
          skill_hint: outgoingSkill?.id ?? null,
          conversation_summary: baseConversationSummary || null,
          conversation_summary_message_count: baseConversationSummaryMessageCount,
          context_file_paths: contextFilePaths,
          rewind_user_message_ordinal: options.rewindUserMessageOrdinal
        },
        (streamEvent) => {
          if ("turn_id" in streamEvent && typeof streamEvent.turn_id === "string") {
            const activeRun = activeChatRunsRef.current.get(conversationId);
            if (activeRun?.abortController === abortController) {
              activeRun.turnId = streamEvent.turn_id;
            }
          }
          if ("elapsed_ms" in streamEvent && typeof streamEvent.elapsed_ms === "number") {
            updateDraftActivityElapsed(streamEvent.elapsed_ms);
          }
          if (streamEvent.event === "turn") {
            syncActivityRecord({ turnId: streamEvent.turn_id });
          } else if (streamEvent.event === "activity") {
            appendDraftActivity(streamEvent);
          } else if (streamEvent.event === "activity_delta") {
            appendDraftActivityDelta(streamEvent);
          } else if (streamEvent.event === "delta" || streamEvent.event === "draft_delta") {
            streamedContent += streamEvent.content;
            assistantDraftContent += streamEvent.content;
            if (isVisibleConversation()) {
              setChatMessages((items) =>
                items.map((message, index) =>
                  index === assistantIndex
                    ? { ...message, content: `${message.content}${streamEvent.content}` }
                    : message
                )
              );
            }
            persistDraftConversation();
          } else if (streamEvent.event === "draft_reset") {
            streamedContent = "";
            assistantDraftContent = "";
            if (isVisibleConversation()) {
              setChatMessages((items) =>
                items.map((message, index) =>
                  index === assistantIndex ? { ...message, content: "" } : message
                )
              );
            }
            persistDraftConversation({ force: true });
          } else if (streamEvent.event === "final") {
            finalReply = streamEvent.content;
            assistantDraftContent = streamEvent.content;
            waitingApprovalDraft = Boolean(streamEvent.waiting_approval);
            if (typeof streamEvent.context_summary === "string") {
              contextSummaryDraft = streamEvent.context_summary;
              if (isVisibleConversation()) setConversationSummary(contextSummaryDraft);
            }
            if (typeof streamEvent.context_summary_message_count === "number") {
              contextSummaryMessageCountDraft = streamEvent.context_summary_message_count;
              if (isVisibleConversation()) {
                setConversationSummaryMessageCount(contextSummaryMessageCountDraft);
              }
            }
            if (isVisibleConversation()) {
              setChatMessages((items) =>
                items.map((message, index) =>
                  index === assistantIndex ? { ...message, content: streamEvent.content } : message
                )
              );
              setStatus({
                tone: streamEvent.waiting_approval ? "loading" : "success",
                text: streamEvent.waiting_approval
                  ? "等待你确认终端命令"
                  : streamEvent.used_tools
                    ? "已完成智能处理"
                    : "已回复"
              });
            }
            persistDraftConversation({ force: true, completed: true });
          } else if (streamEvent.event === "error") {
            const errorMessage = streamErrorMessage(streamEvent);
            assistantDraftContent = `这次没有成功：${errorMessage}`;
            activityEventsDraft = mergeStreamErrorActivity(activityEventsDraft, streamEvent, errorMessage);
            if (isVisibleConversation()) setActivityEvents(activityEventsDraft);
            syncActivityRecord({ events: activityEventsDraft, completed: true });
            if (isVisibleConversation()) {
              setChatMessages((items) =>
                items.map((message, index) =>
                  index === assistantIndex
                    ? { ...message, content: assistantDraftContent }
                    : message
                )
              );
              setStatus({ tone: "error", text: errorMessage });
              activityRunningRef.current = false;
              setActivityRunning(false);
            }
            persistDraftConversation({ force: true, completed: true });
          } else if (streamEvent.event === "cancelled") {
            assistantDraftContent = streamEvent.message || "已停止当前轮处理。";
            appendDraftActivity({
              event: "activity",
              phase: "complete",
              title: "已停止",
              detail: assistantDraftContent
            });
            syncActivityRecord({ completed: true });
            if (isVisibleConversation()) {
              setChatMessages((items) =>
                items.map((message, index) =>
                  index === assistantIndex ? { ...message, content: assistantDraftContent } : message
                )
              );
              setStatus({ tone: "idle", text: "已停止当前轮" });
              activityRunningRef.current = false;
              setActivityRunning(false);
            }
            persistDraftConversation({ force: true, completed: true });
          }
        },
        { signal: abortController.signal }
      );
      if (isVisibleConversation()) {
        activityRunningRef.current = false;
        setActivityRunning(false);
      }
      const finalActivityRecords = completeDraftActivity();
      const assistantContent =
        assistantDraftContent.trim() ||
        finalReply.trim() ||
        streamedContent.trim() ||
        "这次没有生成可显示的回复。";
      assistantDraftContent = assistantContent;
      const completedMessages: ChatMessage[] = [
        ...nextMessages,
        {
          role: "assistant",
          content: assistantContent
        }
      ];
      if (isVisibleConversation()) {
        setChatMessages((items) =>
          items.map((message, index) =>
            index === assistantIndex && !message.content.trim()
              ? { ...message, content: "这次没有生成可显示的回复。" }
              : message
          )
        );
        setActivityElapsedMs((value) => value || 1);
        setStatus((current) =>
          current.tone === "loading" && !waitingApprovalDraft
            ? { tone: "success", text: "已回复" }
            : current
        );
      }
      if (shouldNameConversation) {
        saveNamedConversation({
          id: conversationId,
          messages: completedMessages,
          activities: finalActivityRecords,
          activeActivityIndex: assistantIndex,
          contextSummary: contextSummaryDraft,
          contextSummaryMessageCount: contextSummaryMessageCountDraft
        });
      } else {
        setConversationHistory((items) => {
          const next = updateConversationMessages(
            items,
            conversationId,
            completedMessages,
            finalActivityRecords,
            assistantIndex,
            untitledConversationTitle,
            contextSummaryDraft,
            contextSummaryMessageCountDraft
          );
          conversationHistoryRef.current = next;
          saveConversationHistory(next, currentUser?.username);
          return next;
        });
      }
    } catch (error) {
      const stoppedByUser = abortController.signal.aborted;
      const message = explainError(error);
      assistantDraftContent = stoppedByUser ? "已停止当前轮处理。" : `这次没有成功：${message}`;
      if (isVisibleConversation()) {
        activityRunningRef.current = false;
        setActivityRunning(false);
      }
      appendDraftActivity({
        event: "activity",
        phase: stoppedByUser ? "complete" : "error",
        title: stoppedByUser ? "已停止" : "连接失败",
        detail: stoppedByUser ? "用户停止了当前轮，后续排队消息会继续处理。" : message
      });
      completeDraftActivity();
      if (isVisibleConversation()) {
        setChatMessages((items) =>
          items.map((messageItem, index) =>
            index === assistantIndex ? { ...messageItem, content: assistantDraftContent } : messageItem
          )
        );
      }
      persistDraftConversation({ force: true, completed: true });
      if (isVisibleConversation()) {
        setStatus({ tone: stoppedByUser ? "idle" : "error", text: stoppedByUser ? "已停止当前轮" : message });
      }
    } finally {
      const activeRun = activeChatRunsRef.current.get(conversationId);
      if (activeRun?.abortController === abortController) {
        activeChatRunsRef.current.delete(conversationId);
      }
      if (isVisibleConversation()) {
        activityRunningRef.current = false;
        setActivityRunning(false);
        setBusy(false);
      }
      window.setTimeout(() => runNextQueuedChatMessage(conversationId), 0);
    }
  }

  function stopCurrentChatTurn() {
    const conversationId = currentConversationIdRef.current;
    const activeRun = activeChatRunsRef.current.get(conversationId);
    if (!activeRun) return;
    if (activeRun.turnId) {
      void api.cancelAgentTurn(activeRun.turnId).catch(() => {
        // Local abort still stops the visible stream; backend cancellation is best-effort.
      });
    }
    activeRun.abortController.abort();
    const queuedCount = queuedChatMessagesRef.current.get(conversationId)?.length ?? 0;
    setStatus({
      tone: "loading",
      text:
        queuedCount > 0
          ? `正在停止当前轮，随后处理队列（${queuedCount} 条等待）`
          : "正在停止当前轮…"
    });
  }

  function runNextQueuedChatMessage(conversationId: string) {
    if (activeChatRunsRef.current.has(conversationId)) return;
    const queue = queuedChatMessagesRef.current.get(conversationId) ?? [];
    const [next, ...rest] = queue;
    if (!next) {
      queuedChatMessagesRef.current.delete(conversationId);
      if (currentConversationIdRef.current === conversationId) setQueuedChatCount(0);
      return;
    }
    if (rest.length > 0) queuedChatMessagesRef.current.set(conversationId, rest);
    else queuedChatMessagesRef.current.delete(conversationId);
    if (currentConversationIdRef.current === conversationId) {
      setQueuedChatCount(rest.length);
      setStatus({
        tone: "loading",
        text: rest.length > 0 ? `正在处理队列消息，剩余 ${rest.length} 条` : "正在处理队列消息"
      });
    }
    const conversation = conversationHistoryRef.current.find((item) => item.id === conversationId);
    void runChatMessage(next, {
      conversationId,
      baseMessages: conversation?.messages,
      activityRecords: conversation?.activities,
      conversationSummary: conversation?.contextSummary,
      conversationSummaryMessageCount: conversation?.contextSummaryMessageCount
    });
  }

  async function approvePendingToolBatch(item: AgentActivityEvent, assistantIndex: number) {
    const conversationId = currentConversationIdRef.current;
    const isVisibleConversation = () => currentConversationIdRef.current === conversationId;
    if (activeChatRunsRef.current.has(conversationId)) return;
    const turnId = item.turn_id ?? activityRecords[assistantIndex]?.turnId;
    if (!turnId) {
      setStatus({ tone: "error", text: "没有找到待审批 turn，无法继续执行。" });
      return;
    }

    let assistantDraftContent = chatMessages[assistantIndex]?.content ?? "";
    let finalReply = "";
    let streamedContent = "";
    let contextSummaryDraft = conversationSummary;
    let contextSummaryMessageCountDraft = conversationSummaryMessageCount;
    let messagesDraft = chatMessages;
    let activityEventsDraft: AgentActivityEvent[] = [
      ...(activityRecords[assistantIndex]?.events ?? [])
    ];
    let activityElapsedDraft = activityRecords[assistantIndex]?.elapsedMs ?? 0;
    let activityRecordsDraft: ActivityRecordMap = {
      ...activityRecords,
      [assistantIndex]: {
        events: activityEventsDraft,
        elapsedMs: activityElapsedDraft,
        completed: false,
        turnId
      }
    };
    const activityStartedAtMs = Date.now();
    let lastDraftPersistAt = 0;
    const syncActivityRecord = (patch: Partial<ActivityRecord>) => {
      const previous = activityRecordsDraft[assistantIndex] ?? {
        events: activityEventsDraft,
        elapsedMs: activityElapsedDraft,
        completed: false,
        turnId
      };
      const nextRecord: ActivityRecord = {
        ...previous,
        ...patch,
        turnId
      };
      activityRecordsDraft = {
        ...activityRecordsDraft,
        [assistantIndex]: nextRecord
      };
      if (isVisibleConversation()) setActivityRecords(activityRecordsDraft);
      return activityRecordsDraft;
    };
    const writeAssistantContent = (content: string) => {
      assistantDraftContent = content;
      const base = messagesDraft;
      const next = base.map((message, index) =>
        index === assistantIndex ? { ...message, content } : message
      );
      messagesDraft = next;
      if (isVisibleConversation()) {
        chatMessagesRef.current = next;
        setChatMessages(next);
      }
    };
    const persistDraftConversation = (options?: { force?: boolean; completed?: boolean }) => {
      const now = Date.now();
      if (!options?.force && now - lastDraftPersistAt < 1500) return;
      lastDraftPersistAt = now;
      const snapshotActivities = options?.completed
        ? {
            ...activityRecordsDraft,
            [assistantIndex]: {
              ...(activityRecordsDraft[assistantIndex] ?? {
                events: activityEventsDraft,
                elapsedMs: activityElapsedDraft,
                completed: false,
                turnId
              }),
              completed: true,
              turnId
            }
          }
        : activityRecordsDraft;
      setConversationHistory((items) => {
        const next = updateConversationMessages(
          items,
          conversationId,
          messagesDraft,
          snapshotActivities,
          assistantIndex,
          untitledConversationTitle,
          contextSummaryDraft,
          contextSummaryMessageCountDraft
        );
        conversationHistoryRef.current = next;
        saveConversationHistory(next, currentUser?.username);
        return next;
      });
    };
    const updateDraftActivityElapsed = (elapsedMs: number) => {
      activityElapsedDraft = Math.max(activityElapsedDraft, elapsedMs);
      if (isVisibleConversation()) setActivityElapsedMs(activityElapsedDraft);
      syncActivityRecord({ elapsedMs: activityElapsedDraft });
    };
    const appendDraftActivity = (event: AgentActivityEvent) => {
      activityEventsDraft = [...activityEventsDraft, event];
      if (isVisibleConversation()) setActivityEvents(activityEventsDraft);
      syncActivityRecord({
        events: activityEventsDraft,
        elapsedMs: activityElapsedDraft,
        completed: false
      });
      persistDraftConversation({ force: true });
    };
    const appendDraftActivityDelta = (
      event: Extract<AgentStreamEvent, { event: "activity_delta" }>
    ) => {
      activityEventsDraft = appendActivityDelta(activityEventsDraft, event);
      if (isVisibleConversation()) setActivityEvents(activityEventsDraft);
      syncActivityRecord({
        events: activityEventsDraft,
        elapsedMs: activityElapsedDraft,
        completed: false
      });
      persistDraftConversation();
    };
    const completeDraftActivity = () => {
      activityElapsedDraft = Math.max(activityElapsedDraft, Date.now() - activityStartedAtMs, 1);
      if (isVisibleConversation()) setActivityElapsedMs(activityElapsedDraft);
      return syncActivityRecord({
        events: activityEventsDraft,
        elapsedMs: activityElapsedDraft,
        completed: true
      });
    };

    const abortController = new AbortController();
    activeChatRunsRef.current.set(conversationId, {
      abortController,
      turnId,
      assistantIndex,
      startedAt: activityStartedAtMs
    });
    if (isVisibleConversation()) {
      setActivityPanelMessageIndex(assistantIndex);
      setActivityMessageIndex(assistantIndex);
      setActivityEvents(activityEventsDraft);
      setActivityElapsedMs(activityElapsedDraft);
      setActivityStartedAt(activityStartedAtMs);
      setActivityNow(activityStartedAtMs);
      setActivityOpen(false);
      activityRunningRef.current = true;
      setActivityRunning(true);
      setBusy(true);
      setStatus({ tone: "loading", text: "正在继续执行已审批工具批次…" });
    }
    appendDraftActivity({
      event: "activity",
      phase: "action",
      title: "终端审批已确认",
      detail: "后端将恢复同一个 pending batch，不再让模型重新生成这批工具调用。",
      approval_resolved: true
    });

    try {
      await api.approveAgentTurn(
        turnId,
        { conversation_id: conversationId },
        (streamEvent) => {
          if ("turn_id" in streamEvent && typeof streamEvent.turn_id === "string") {
            const activeRun = activeChatRunsRef.current.get(conversationId);
            if (activeRun?.abortController === abortController) {
              activeRun.turnId = streamEvent.turn_id;
            }
          }
          if ("elapsed_ms" in streamEvent && typeof streamEvent.elapsed_ms === "number") {
            updateDraftActivityElapsed(streamEvent.elapsed_ms);
          }
          if (streamEvent.event === "turn") {
            syncActivityRecord({ turnId: streamEvent.turn_id });
          } else if (streamEvent.event === "activity") {
            appendDraftActivity(streamEvent);
          } else if (streamEvent.event === "activity_delta") {
            appendDraftActivityDelta(streamEvent);
          } else if (streamEvent.event === "delta" || streamEvent.event === "draft_delta") {
            streamedContent += streamEvent.content;
            writeAssistantContent(`${assistantDraftContent}${streamEvent.content}`);
            persistDraftConversation();
          } else if (streamEvent.event === "draft_reset") {
            streamedContent = "";
            writeAssistantContent("");
            persistDraftConversation({ force: true });
          } else if (streamEvent.event === "final") {
            finalReply = streamEvent.content;
            if (typeof streamEvent.context_summary === "string") {
              contextSummaryDraft = streamEvent.context_summary;
              if (isVisibleConversation()) setConversationSummary(contextSummaryDraft);
            }
            if (typeof streamEvent.context_summary_message_count === "number") {
              contextSummaryMessageCountDraft = streamEvent.context_summary_message_count;
              if (isVisibleConversation()) {
                setConversationSummaryMessageCount(contextSummaryMessageCountDraft);
              }
            }
            writeAssistantContent(streamEvent.content);
            if (isVisibleConversation()) {
              setStatus({
                tone: "success",
                text: streamEvent.used_tools ? "已完成智能处理" : "已回复"
              });
            }
            persistDraftConversation({ force: true, completed: true });
          } else if (streamEvent.event === "error") {
            const errorMessage = streamErrorMessage(streamEvent);
            writeAssistantContent(`这次没有成功：${errorMessage}`);
            activityEventsDraft = mergeStreamErrorActivity(activityEventsDraft, streamEvent, errorMessage);
            if (isVisibleConversation()) setActivityEvents(activityEventsDraft);
            syncActivityRecord({ events: activityEventsDraft, completed: true });
            if (isVisibleConversation()) setStatus({ tone: "error", text: errorMessage });
            persistDraftConversation({ force: true, completed: true });
          } else if (streamEvent.event === "cancelled") {
            writeAssistantContent(streamEvent.message || "已停止当前轮处理。");
            appendDraftActivity({
              event: "activity",
              phase: "complete",
              title: "已停止",
              detail: streamEvent.message || "已停止当前轮处理。"
            });
            syncActivityRecord({ completed: true });
            if (isVisibleConversation()) setStatus({ tone: "idle", text: "已停止当前轮" });
            persistDraftConversation({ force: true, completed: true });
          }
        },
        { signal: abortController.signal }
      );
      const finalActivityRecords = completeDraftActivity();
      const assistantContent =
        assistantDraftContent.trim() ||
        finalReply.trim() ||
        streamedContent.trim() ||
        "这次没有生成可显示的回复。";
      writeAssistantContent(assistantContent);
      setConversationHistory((items) => {
        const next = updateConversationMessages(
          items,
          conversationId,
          messagesDraft,
          finalActivityRecords,
          assistantIndex,
          untitledConversationTitle,
          contextSummaryDraft,
          contextSummaryMessageCountDraft
        );
        conversationHistoryRef.current = next;
        saveConversationHistory(next, currentUser?.username);
        return next;
      });
    } catch (error) {
      const stoppedByUser = abortController.signal.aborted;
      const message = explainError(error);
      writeAssistantContent(stoppedByUser ? "已停止当前轮处理。" : `这次没有成功：${message}`);
      appendDraftActivity({
        event: "activity",
        phase: stoppedByUser ? "complete" : "error",
        title: stoppedByUser ? "已停止" : "连接失败",
        detail: stoppedByUser ? "用户停止了当前轮。" : message
      });
      completeDraftActivity();
      persistDraftConversation({ force: true, completed: true });
      if (isVisibleConversation()) {
        setStatus({ tone: stoppedByUser ? "idle" : "error", text: stoppedByUser ? "已停止当前轮" : message });
      }
    } finally {
      const activeRun = activeChatRunsRef.current.get(conversationId);
      if (activeRun?.abortController === abortController) {
        activeChatRunsRef.current.delete(conversationId);
      }
      if (isVisibleConversation()) {
        activityRunningRef.current = false;
        setActivityRunning(false);
        setBusy(false);
      }
      window.setTimeout(() => runNextQueuedChatMessage(conversationId), 0);
    }
  }

  function saveNamedConversation({
    id,
    messages,
    activities,
    activeActivityIndex,
    contextSummary,
    contextSummaryMessageCount
  }: {
    id: string;
    messages: ChatMessage[];
    activities: ActivityRecordMap;
    activeActivityIndex: number;
    contextSummary: string;
    contextSummaryMessageCount: number;
  }) {
    setConversationHistory((items) => {
      const next = upsertConversation(items, {
        id,
        title: pendingConversationTitle,
        group: "最近",
        messages,
        contextSummary,
        contextSummaryMessageCount,
        activities,
        activeActivityIndex
      });
      conversationHistoryRef.current = next;
      return next;
    });
    void (async () => {
      try {
        const payload = await api.generateChatTitle({ messages, profile: agentForm.profile });
        const title = cleanConversationTitle(payload.title, pendingConversationTitle);
        setConversationHistory((items) => {
          const current = items.find((item) => item.id === id);
          if (!current || current.title !== pendingConversationTitle) return items;
          if (title === pendingConversationTitle) return items;
          return upsertConversation(items, {
            ...current,
            id,
            title,
            group: current?.group ?? "最近",
            messages: current?.messages ?? messages,
            contextSummary: current?.contextSummary || contextSummary,
            contextSummaryMessageCount:
              current?.contextSummaryMessageCount ?? contextSummaryMessageCount,
            activities: current?.activities ?? activities,
            activeActivityIndex: current?.activeActivityIndex ?? activeActivityIndex
          });
        });
      } catch {
        setConversationHistory((items) => {
          const current = items.find((item) => item.id === id);
          if (!current || current.title !== pendingConversationTitle) return items;
          return items;
        });
      }
    })();
  }

  async function generateAndApplyConversationTitle(item: ConversationHistoryItem) {
    try {
      const payload = await api.generateChatTitle({
        messages: item.messages,
        profile: agentForm.profile
      });
      const title = cleanConversationTitle(payload.title, pendingConversationTitle);
      if (title === pendingConversationTitle) return;
      setConversationHistory((items) =>
        orderConversationHistory(
          items.map((current) =>
            current.id === item.id && current.title === pendingConversationTitle
              ? { ...current, title }
              : current
          )
        )
      );
    } catch (error) {
      setStatus({ tone: "error", text: `对话命名失败：${explainError(error)}` });
    } finally {
      titleGenerationInFlightRef.current.delete(item.id);
    }
  }

  async function generateMinutes(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setMeetingResult(null);
    setStatus({ tone: "loading", text: "正在生成会议纪要…" });
    try {
      const supplemental_paths = meetingForm.supplemental_paths
        .split("\n")
        .map((path) => path.trim())
        .filter(Boolean);
      const payload = await api.generateMinutes({
        ...meetingForm,
        supplemental_paths
      });
      setMeetingResult(payload.result);
      setArtifactTabWithUrl("meeting");
      setStatus({ tone: "success", text: "会议纪要已生成" });
      await refreshFiles(meetingForm.output_dir);
    } catch (error) {
      setStatus({ tone: "error", text: explainError(error) });
    } finally {
      setBusy(false);
    }
  }

  async function switchModel(name: string) {
    setBusy(true);
    setStatus({ tone: "loading", text: "正在切换模型…" });
    try {
      const nextModels = await api.useModel(name);
      setModels(nextModels);
      setAgentForm((form) => ({ ...form, profile: name }));
      setMeetingForm((form) => ({ ...form, profile: name }));
      setStatus({ tone: "success", text: `当前模型：${name}` });
    } catch (error) {
      setStatus({ tone: "error", text: explainError(error) });
    } finally {
      setBusy(false);
    }
  }

  function closeModelEditor() {
    setModelEditorMode(null);
    setEditingModelName("");
    setModelForm(createDefaultModelForm());
    setDiscoveredModelIds([]);
    setModelConnectionResult(null);
    setShowModelApiKey(false);
  }

  function openAddModel() {
    setModelEditorMode("add");
    setEditingModelName("");
    setModelForm(createDefaultModelForm());
    setDiscoveredModelIds([]);
    setModelConnectionResult(null);
    setShowModelApiKey(false);
  }

  function openEditModel(profile: ModelProfile) {
    setModelEditorMode("edit");
    setEditingModelName(profile.name);
    setModelForm({
      name: profile.name,
      preset: presetForModelProfile(profile),
      provider: profile.provider,
      base_url: profile.base_url,
      model: profile.model,
      api_key: "",
      temperature: profile.temperature,
      max_tokens: profile.max_tokens,
      timeout_seconds: profile.timeout_seconds,
      set_default: profile.default,
      source_name: ""
    });
    setDiscoveredModelIds([]);
    setModelConnectionResult(null);
    setShowModelApiKey(false);
  }

  function copyModelProfile(profile: ModelProfile) {
    setModelEditorMode("add");
    setEditingModelName("");
    setModelForm({
      name: nextCopiedModelName(profile.name, profiles),
      preset: presetForModelProfile(profile),
      provider: profile.provider,
      base_url: profile.base_url,
      model: profile.model,
      api_key: "",
      temperature: profile.temperature,
      max_tokens: profile.max_tokens,
      timeout_seconds: profile.timeout_seconds,
      set_default: false,
      source_name: profile.name
    });
    setDiscoveredModelIds([]);
    setModelConnectionResult({
      tone: "success",
      text: `已复制 ${profile.name} 的参数和密钥引用，请确认新名称后保存。`
    });
    setShowModelApiKey(false);
  }

  async function saveModelConfiguration(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setStatus({
      tone: "loading",
      text: modelEditorMode === "edit" ? "正在更新模型配置…" : "正在添加模型配置…"
    });
    try {
      const nextModels = modelEditorMode === "edit"
        ? await api.updateModel({ ...modelForm, name: editingModelName })
        : await api.addModel(modelForm);
      setModels(nextModels);
      const savedName = modelEditorMode === "edit" ? editingModelName : modelForm.name;
      closeModelEditor();
      setStatus({
        tone: "success",
        text: modelEditorMode === "edit"
          ? `${savedName} 的配置已更新`
          : `${savedName} 已添加`
      });
    } catch (error) {
      setStatus({ tone: "error", text: explainError(error) });
    } finally {
      setBusy(false);
    }
  }

  async function testModelConfiguration(profile?: ModelProfile) {
    setBusy(true);
    setModelConnectionResult(null);
    const name = profile?.name || (modelEditorMode === "edit" ? editingModelName : modelForm.name);
    setStatus({ tone: "loading", text: `正在测试 ${name || "新配置"}…` });
    try {
      const result = await api.testModel(
        profile ? { name: profile.name } : {
          ...modelForm,
          name: modelEditorMode === "edit" ? editingModelName : modelForm.name
        }
      );
      const text = `连接可用 · ${result.latency_ms} ms · ${result.model}`;
      setModelConnectionResult({ tone: "success", text });
      setStatus({ tone: "success", text });
    } catch (error) {
      const text = explainError(error);
      setModelConnectionResult({ tone: "error", text });
      setStatus({ tone: "error", text });
    } finally {
      setBusy(false);
    }
  }

  async function discoverModels() {
    setBusy(true);
    setModelConnectionResult(null);
    setStatus({ tone: "loading", text: "正在获取可用模型…" });
    try {
      const result = await api.discoverModels({
        ...modelForm,
        name: modelEditorMode === "edit" ? editingModelName : modelForm.name
      });
      setDiscoveredModelIds(result.models);
      const text = result.count
        ? `已获取 ${result.count} 个模型 · ${result.latency_ms} ms`
        : "接口连接成功，但没有返回模型";
      setModelConnectionResult({ tone: "success", text });
      setStatus({ tone: "success", text });
    } catch (error) {
      const text = explainError(error);
      setModelConnectionResult({ tone: "error", text });
      setStatus({ tone: "error", text });
    } finally {
      setBusy(false);
    }
  }

  async function deleteModelConfiguration(profile: ModelProfile) {
    if (deleteConfirmModelName !== profile.name) {
      setDeleteConfirmModelName(profile.name);
      return;
    }
    setBusy(true);
    setStatus({ tone: "loading", text: `正在删除 ${profile.name}…` });
    try {
      const nextModels = await api.deleteModel(profile.name);
      setModels(nextModels);
      setDeleteConfirmModelName("");
      if (editingModelName === profile.name) closeModelEditor();
      setStatus({ tone: "success", text: `${profile.name} 已删除` });
    } catch (error) {
      setStatus({ tone: "error", text: explainError(error) });
    } finally {
      setBusy(false);
    }
  }

  async function copyPath(path: string) {
    await copyText(path, "路径已复制");
  }

  async function copyText(text: string, successText = "已复制") {
    try {
      await navigator.clipboard.writeText(text);
      setStatus({ tone: "success", text: successText });
    } catch {
      setStatus({ tone: "error", text: "剪贴板不可用，请手动选择复制。" });
    }
  }

  async function saveAsrSettings(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setStatus({ tone: "loading", text: "正在保存语音识别设置…" });
    try {
      const payload = await api.saveAsrSettings(asrSettingsForm);
      setAsrSettings(payload);
      setAsrSettingsForm({
        profile: payload.profile,
        model_id: payload.model_id,
        hotwords: payload.hotwords
      });
      setStatus({ tone: "success", text: "语音识别设置已保存" });
    } catch (error) {
      setStatus({ tone: "error", text: explainError(error) });
    } finally {
      setBusy(false);
    }
  }

  async function saveAgentSettings(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setStatus({ tone: "loading", text: "正在保存工作背景与文件格式…" });
    try {
      const payload = await api.saveAgentSettings(agentSettingsForm);
      setAgentSettings(payload);
      setAgentSettingsForm({
        nickname: payload.nickname,
        occupation: payload.occupation,
        details: payload.details,
        memory_enabled: payload.memory_enabled,
        work_background: payload.work_background,
        company_document_format: payload.company_document_format
      });
      setStatus({ tone: "success", text: "工作背景与文件格式已保存" });
    } catch (error) {
      setStatus({ tone: "error", text: explainError(error) });
    } finally {
      setBusy(false);
    }
  }

  async function refreshCrossChatMemories() {
    try {
      const payload = await api.memories();
      setCrossChatMemories(payload.memories);
      setMemoryProfile(payload.profile ? { content: payload.profile.content, updated_at: payload.profile.updated_at } : null);
      setStatus({ tone: "success", text: `已载入 ${payload.count} 条自动记忆` });
    } catch (error) {
      setStatus({ tone: "error", text: explainError(error) });
    }
  }

  function startEditMemory(memory: CrossChatMemory) {
    setEditingMemoryId(memory.id);
    setMemoryDraft(memory.content);
    setDeleteConfirmMemoryId(null);
  }

  async function saveMemoryCorrection(memoryId: string) {
    if (!memoryDraft.trim()) return;
    setBusy(true);
    setStatus({ tone: "loading", text: "正在保存纠正…" });
    try {
      const payload = await api.updateMemory(memoryId, memoryDraft);
      setCrossChatMemories((items) =>
        items.map((item) => (item.id === memoryId ? payload.memory : item))
      );
      setEditingMemoryId(null);
      setMemoryDraft("");
      setStatus({ tone: "success", text: "记忆已纠正，后续检索将优先使用纠正版" });
    } catch (error) {
      setStatus({ tone: "error", text: explainError(error) });
    } finally {
      setBusy(false);
    }
  }

  async function deleteCrossChatMemory(memoryId: string) {
    if (deleteConfirmMemoryId !== memoryId) {
      setDeleteConfirmMemoryId(memoryId);
      setEditingMemoryId(null);
      return;
    }
    setBusy(true);
    setStatus({ tone: "loading", text: "正在删除记忆…" });
    try {
      await api.deleteMemory(memoryId);
      setCrossChatMemories((items) => items.filter((item) => item.id !== memoryId));
      setDeleteConfirmMemoryId(null);
      setStatus({ tone: "success", text: "记忆已删除，不会在自动同步时恢复" });
    } catch (error) {
      setStatus({ tone: "error", text: explainError(error) });
    } finally {
      setBusy(false);
    }
  }

  function openMemorySource(memory: CrossChatMemory) {
    const conversation = conversationHistory.find((item) => item.id === memory.conversation_id);
    if (!conversation) {
      setStatus({ tone: "error", text: "来源聊天不在当前聊天列表中，可能已被删除。" });
      return;
    }
    openConversation(conversation);
  }

  async function openLocalFile(path: string) {
    setBusy(true);
    setStatus({ tone: "loading", text: "正在打开文件…" });
    try {
      await api.openLocalFile(path);
      setStatus({ tone: "success", text: "文件已交给系统打开" });
    } catch (error) {
      setStatus({ tone: "error", text: explainError(error) });
    } finally {
      setBusy(false);
    }
  }

  async function revealLocalFile(path: string) {
    setBusy(true);
    setStatus({ tone: "loading", text: "正在访达中定位…" });
    try {
      await api.revealLocalFile(path);
      setStatus({ tone: "success", text: "已在访达中显示" });
    } catch (error) {
      setStatus({ tone: "error", text: explainError(error) });
    } finally {
      setBusy(false);
    }
  }

  async function submitAuth(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setAuthError("");
    if (authMode === "register" && authForm.password !== authForm.confirm) {
      setAuthError("两次输入的密码不一致");
      return;
    }
    setAuthBusy(true);
    try {
      const payload = authMode === "login"
        ? await api.login(authForm.username, authForm.password)
        : await api.register(authForm.username, authForm.password);
      if (!payload.user) throw new Error("账户信息未返回，请重试");
      setCurrentUser(payload.user);
      setAuthState("authenticated");
      setAuthForm({ username: "", password: "", confirm: "" });
    } catch (error) {
      setAuthError(explainError(error));
    } finally {
      setAuthBusy(false);
    }
  }

  async function logout() {
    try {
      await api.logout();
    } finally {
      setCurrentUser(null);
      setAuthState("anonymous");
      setProjects([]);
      setSelectedProject(null);
      setActiveProjectId(null);
      activeProjectIdRef.current = null;
      setTemporarySync(null);
      setTemporarySyncText("");
      setTemporarySyncTextDirty(false);
      temporarySyncTextDirtyRef.current = false;
      setConversationHistory([]);
      setChatMessages([{
        role: "assistant",
        content: "你好，我是本地工作智能体。登录后即可继续使用你的独立工作区。"
      }]);
    }
  }

  if (authState !== "authenticated" || !currentUser) {
    return (
      <AuthScreen
        mode={authMode}
        form={authForm}
        busy={authBusy || authState === "loading"}
        error={authError}
        onModeChange={(mode) => {
          setAuthMode(mode);
          setAuthError("");
        }}
        onFormChange={setAuthForm}
        onSubmit={submitAuth}
      />
    );
  }

  const fileReaderOpen = view === "artifacts" && artifactTab === "files" && selectedFile !== null;
  const meetingArchiveOpen = view === "artifacts" && artifactTab === "meeting";

  return (
    <div
      className={`app-shell chatgpt-shell ${dragActive ? "is-file-dragging" : ""} ${
        sidebarCollapsed ? "is-sidebar-collapsed" : ""
      }`}
      onDragEnter={handleAppDragEnter}
      onDragOver={handleAppDragOver}
      onDragLeave={handleAppDragLeave}
      onDrop={handleAppDrop}
    >
      <a className="skip-link" href="#main-content">
        跳到主内容
      </a>

      <aside className="sidebar" id="workspace-sidebar" aria-label="工作台导航">
        <div className="brand-block">
          <div className="brand-copy">
            <h1>工作智能体</h1>
            <p translate="no">{workspace ? compactPath(workspace) : "本地工作区"}</p>
          </div>
          <div className="sidebar-header-actions">
            <button
              type="button"
              className="sidebar-icon-button sidebar-search-button"
              aria-label="搜索聊天"
              title="搜索聊天"
              onClick={() => {
                setSearchOpen(true);
                setConversationSearch("");
              }}
            >
              <Search aria-hidden="true" />
            </button>
            <button
              type="button"
              className="sidebar-icon-button sidebar-toggle-button"
              aria-label={sidebarCollapsed ? "展开侧边栏" : "收起侧边栏"}
              aria-expanded={!sidebarCollapsed}
              aria-controls="workspace-sidebar"
              title={sidebarCollapsed ? "展开侧边栏" : "收起侧边栏"}
              onClick={() => setSidebarCollapsed((collapsed) => !collapsed)}
            >
              <PanelLeft aria-hidden="true" />
            </button>
          </div>
        </div>

        <nav className="nav-list" aria-label="主要区域">
          <button
            type="button"
            className={`nav-button ${
              view === "agent" && chatMessages.length <= 1 && currentConversationId.startsWith("local")
                ? "is-active"
                : ""
            }`}
            aria-label="新聊天"
            title="新聊天"
            onClick={() => startNewChat()}
          >
            <MessageSquarePlus aria-hidden="true" />
            <span>新聊天</span>
          </button>
          <button
            type="button"
            className={`nav-button nav-section-start ${
              view === "artifacts" && artifactTab === "files" ? "is-active" : ""
            }`}
            aria-current={view === "artifacts" && artifactTab === "files" ? "page" : undefined}
            aria-label="文件库"
            title="文件库"
            onClick={() => setArtifactTabWithUrl("files")}
          >
            <Library aria-hidden="true" />
            <span>文件库</span>
          </button>
          <button
            type="button"
            className={`nav-button ${view === "projects" ? "is-active" : ""}`}
            aria-current={view === "projects" ? "page" : undefined}
            aria-label="项目"
            title="项目"
            onClick={showProjectList}
          >
            <Folder aria-hidden="true" />
            <span>项目</span>
          </button>
          <button
            type="button"
            className={`nav-button ${view === "skills" ? "is-active" : ""}`}
            aria-current={view === "skills" ? "page" : undefined}
            aria-label="技能"
            title="技能"
            onClick={() => setViewWithUrl("skills")}
          >
            <Wrench aria-hidden="true" />
            <span>技能</span>
          </button>
          <button
            type="button"
            className={`nav-button ${view === "models" ? "is-active" : ""}`}
            aria-current={view === "models" ? "page" : undefined}
            aria-label="设置"
            title="设置"
            onClick={() => setViewWithUrl("models")}
          >
            <Cpu aria-hidden="true" />
            <span>设置</span>
          </button>
          <button
            type="button"
            className={`nav-button ${view === "more" ? "is-active" : ""}`}
            aria-current={view === "more" ? "page" : undefined}
            aria-label="更多"
            title="更多"
            onClick={() => setViewWithUrl("more")}
          >
            <MoreHorizontal aria-hidden="true" />
            <span>更多</span>
          </button>
        </nav>

        <section className="recent-block" aria-label="历史对话">
          <h2>历史对话</h2>
          <div className="recent-list">
            {conversationHistory.length > 0 ? (
              conversationHistory.map((item) => {
                const isActive = currentConversationId === item.id;
                const isRenaming = renamingConversationId === item.id;
                const projectName = item.projectId
                  ? projects.find((project) => project.id === item.projectId)?.name
                  : undefined;
                return (
                  <div
                    key={item.id}
                    className={`recent-item ${isActive ? "is-active" : ""} ${
                      item.pinned ? "is-pinned" : ""
                    } ${historyMenu?.id === item.id ? "has-menu" : ""}`}
                  >
                    {isRenaming ? (
                      <form
                        className="recent-rename-form"
                        onSubmit={(event) => {
                          event.preventDefault();
                          commitRenameConversation(item.id);
                        }}
                      >
                        <input
                          autoFocus
                          value={renameDraft}
                          aria-label="重命名历史对话"
                          onFocus={(event) => event.currentTarget.select()}
                          onChange={(event) => setRenameDraft(event.target.value)}
                          onBlur={() => commitRenameConversation(item.id)}
                          onKeyDown={(event) => {
                            if (event.key === "Escape") {
                              setRenamingConversationId(null);
                              setRenameDraft("");
                            }
                          }}
                        />
                      </form>
                    ) : (
                      <button
                        type="button"
                        className="recent-title-button"
                        title={item.title}
                        onClick={() => openConversation(item)}
                      >
                        <span className="recent-conversation-label">
                          <span className="recent-conversation-title">{item.title}</span>
                          {projectName ? <span className="recent-project-name">{projectName}</span> : null}
                        </span>
                      </button>
                    )}
                    {!isRenaming ? (
                      <div className="recent-actions" aria-label={`${item.title} 的操作`}>
                        <button
                          type="button"
                          className={`recent-action-button recent-pin-button ${item.pinned ? "is-pinned" : ""}`}
                          aria-label={item.pinned ? "取消置顶" : "置顶对话"}
                          title={item.pinned ? "取消置顶" : "置顶对话"}
                          onClick={(event) => {
                            event.stopPropagation();
                            toggleConversationPin(item.id);
                          }}
                        >
                          {item.pinned ? <PinOff aria-hidden="true" /> : <Pin aria-hidden="true" />}
                        </button>
                        <button
                          type="button"
                          className="recent-action-button"
                          aria-label="更多操作"
                          aria-haspopup="menu"
                          aria-expanded={historyMenu?.id === item.id}
                          onClick={(event) => openHistoryMenu(event, item.id)}
                        >
                          <MoreHorizontal aria-hidden="true" />
                        </button>
                      </div>
                    ) : null}
                  </div>
                );
              })
            ) : (
              <span className="recent-empty">开始对话后会显示在这里</span>
            )}
          </div>
        </section>

        <div className="side-status" aria-label="当前状态">
          <div className="account-card">
            <span className="account-avatar" aria-hidden="true"><UserRound /></span>
            <span className="account-copy">
              <strong>{currentUser.username}</strong>
              <small>{currentUser.role === "admin" ? "管理员" : "独立账户"}</small>
            </span>
            <button type="button" className="account-logout" onClick={() => void logout()} title="退出登录" aria-label="退出登录">
              <LogOut aria-hidden="true" />
            </button>
          </div>
          <StatusLine label="当前模型" value={currentProfile?.name ?? "未加载"} />
          <StatusLine label="密钥状态" value={`${keyReadyCount}/${profiles.length} 已配置`} />
          <StatusLine label="可用技能" value={`${skills.length} 个`} />
        </div>
      </aside>

      <main
        className={`main ${view === "agent" ? "chat-main" : ""} ${
          fileReaderOpen ? "file-reader-main" : ""
        } ${meetingArchiveOpen ? "meeting-archive-main" : ""}`}
        id="main-content"
      >
        {!fileReaderOpen ? (
          <header className="topbar">
            <div>
              <h2>{titleForView(view, artifactTab)}</h2>
              <p className="overline">
                {view === "agent"
                  ? activeProjectId
                    ? `项目：${selectedProject?.name ?? "正在载入"} · 自动使用项目资料`
                    : "本地文件、技能和模型都在同一条对话里汇合"
                  : view === "projects"
                    ? selectedProject
                      ? "资料、指令和多轮对话共同组成项目上下文"
                      : "为每项长期工作建立独立资料空间"
                  : view === "skills"
                    ? "当前对话智能体可自动调用的能力"
                  : view === "transcribe"
                    ? "浏览器麦克风实时记录，本地降噪后转写并保存"
                  : view === "sync"
                    ? "同一账号的设备之间临时传递文字和文件"
                  : view === "more"
                    ? "临时同步、会议归档、实时转写等功能入口"
                  : view === "artifacts" && artifactTab === "files"
                    ? "拖入材料和生成产出都会沉淀在这里"
                    : "本地工作区"}
              </p>
            </div>
            <div className="topbar-actions">
              <StatusPill tone={status.tone}>{status.text}</StatusPill>
              <button type="button" className="secondary-button" onClick={refreshAll} disabled={busy}>
                <RefreshCw aria-hidden="true" className={busy ? "spin" : ""} />
                刷新
              </button>
            </div>
          </header>
        ) : null}

        <section className="workspace-surface" aria-busy={busy}>
          {view === "agent" && renderAgent()}
          {view === "projects" && renderProjects()}
          {view === "skills" && renderSkills()}
          {view === "transcribe" && renderRealtimeTranscription()}
          {view === "sync" && renderTemporarySync()}
          {view === "artifacts" && renderArtifacts()}
          {view === "models" && renderModels()}
          {view === "more" && renderMore()}
        </section>

        <div className="live-region" aria-live="polite" aria-atomic="true">
          {status.text}
        </div>
      </main>

      {searchOpen ? (
        <div className="chat-search-layer" role="presentation" onMouseDown={() => setSearchOpen(false)}>
          <section
            className="chat-search-dialog"
            role="dialog"
            aria-modal="true"
            aria-label="搜索聊天"
            onMouseDown={(event) => event.stopPropagation()}
          >
            <div className="chat-search-input">
              <Search aria-hidden="true" />
              <input
                autoFocus
                type="search"
                placeholder="搜索聊天..."
                value={conversationSearch}
                onChange={(event) => setConversationSearch(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Escape") {
                    setSearchOpen(false);
                  }
                }}
              />
              <button type="button" aria-label="关闭搜索" onClick={() => setSearchOpen(false)}>
                <X aria-hidden="true" />
              </button>
            </div>
            <div className="chat-search-results">
              <button type="button" className="search-new-chat-row" onClick={() => startNewChat()}>
                <MessageSquarePlus aria-hidden="true" />
                <span>新聊天</span>
              </button>
              {filteredConversationHistory.length > 0 ? (
                <HistorySearchGroups items={filteredConversationHistory} onOpen={openConversation} />
              ) : conversationHistory.length === 0 ? (
                <div className="search-empty-state">暂无历史对话</div>
              ) : (
                <div className="search-empty-state">没有找到匹配的历史对话</div>
              )}
            </div>
          </section>
        </div>
      ) : null}

      {projectCreateOpen ? renderProjectCreateDialog() : null}

      {historyMenu ? renderHistoryMenu() : null}

      {dragActive ? (
        <div className="drop-overlay" aria-hidden="true">
          <div className="drop-overlay-content">
            <div className="drop-illustration">
              <span className="drop-tile drop-tile-audio">
                <Music2 aria-hidden="true" />
              </span>
              <span className="drop-tile drop-tile-document">
                <FileText aria-hidden="true" />
              </span>
              <span className="drop-tile drop-tile-image">
                <ImageIcon aria-hidden="true" />
              </span>
            </div>
            <strong>添加任意内容</strong>
            <p>将任意文件拖放到此处，以将其添加到对话中</p>
          </div>
        </div>
      ) : null}
    </div>
  );

  function renderProjects() {
    if (!selectedProject) {
      return (
        <div className="projects-page">
          <header className="projects-heading">
            <div>
              <h2>项目</h2>
              <p>把一项长期工作的资料、要求和对话集中在同一个上下文中。</p>
            </div>
            <button type="button" className="primary-button" onClick={() => setProjectCreateOpen(true)}>
              <Plus aria-hidden="true" />
              新建项目
            </button>
          </header>
          {projects.length > 0 ? (
            <div className="project-table" role="list" aria-label="项目列表">
              <div className="project-table-head" aria-hidden="true">
                <span>名称</span>
                <span>资料</span>
                <span>最近更新</span>
              </div>
              {projects.map((project) => (
                <button
                  key={project.id}
                  type="button"
                  className="project-row"
                  role="listitem"
                  onClick={() => void openProject(project.id)}
                >
                  <span className="project-row-name">
                    <span className="project-folder-icon" aria-hidden="true"><Folder /></span>
                    <span><strong>{project.name}</strong><small>{project.root}</small></span>
                  </span>
                  <span>{project.file_count} 份</span>
                  <span>{formatProjectDate(project.updated_at)}</span>
                  <ChevronRight aria-hidden="true" />
                </button>
              ))}
            </div>
          ) : (
            <div className="project-empty-state">
              <span className="project-empty-icon" aria-hidden="true"><Folder /></span>
              <h3>为正在跟进的工作建立第一个项目</h3>
              <p>例如“某科技项目可行性报告”，把政策、企业材料、会议纪要和模板放进去，后续对话会自动使用这些资料。</p>
              <button type="button" className="primary-button" onClick={() => setProjectCreateOpen(true)}>
                <Plus aria-hidden="true" />
                新建项目
              </button>
            </div>
          )}
        </div>
      );
    }

    const projectChats = conversationHistory.filter((item) => item.projectId === selectedProject.id);
    return (
      <div className="project-detail-page project-home-page">
        <header className="project-home-header">
          <div className="project-home-title">
            <Folder aria-hidden="true" />
            <h2>{selectedProject.name}</h2>
          </div>
          <button
            type="button"
            className="icon-action-button"
            aria-label="项目设置"
            title="项目设置"
            onClick={() => setProjectSettingsOpen((open) => !open)}
          >
            <Settings2 aria-hidden="true" />
          </button>
        </header>

        {projectSettingsOpen ? (
          <form className="project-settings-panel" onSubmit={saveProjectSettings}>
            <label>
              <span>项目名称</span>
              <input
                value={projectSettingsForm.name}
                onChange={(event) => setProjectSettingsForm((form) => ({ ...form, name: event.target.value }))}
              />
            </label>
            <label>
              <span>项目指令</span>
              <textarea
                rows={5}
                placeholder="例如：写可行性报告时优先引用正式政策原文；不确定的数据要标注待核验。"
                value={projectSettingsForm.instructions}
                onChange={(event) =>
                  setProjectSettingsForm((form) => ({ ...form, instructions: event.target.value }))
                }
              />
            </label>
            <div className="project-settings-actions">
              <span>记忆范围固定为“仅项目”，不会引用其他项目资料。</span>
              <button type="submit" className="primary-button" disabled={busy}>保存设置</button>
            </div>
          </form>
        ) : null}

        <form className="project-chat-starter" onSubmit={startProjectChatFromHome}>
          <label className="project-chat-add-source" title="添加项目文件">
            <Plus aria-hidden="true" />
            <span className="sr-only">添加项目文件</span>
            <input
              type="file"
              multiple
              onChange={(event) => {
                if (event.currentTarget.files) void uploadProjectFiles(event.currentTarget.files, selectedProject.id);
                event.currentTarget.value = "";
              }}
            />
          </label>
          <input
            value={projectChatDraft}
            aria-label="项目中的新聊天"
            placeholder={`${selectedProject.name}中的新聊天`}
            onChange={(event) => setProjectChatDraft(event.target.value)}
          />
          <button type="submit" aria-label="开始项目聊天" title="开始项目聊天">
            <SendHorizontal aria-hidden="true" />
          </button>
        </form>

        <div className="project-content-tabs" role="tablist" aria-label="项目内容">
          <button
            type="button"
            role="tab"
            aria-selected={projectDetailTab === "chat"}
            className={projectDetailTab === "chat" ? "is-active" : ""}
            onClick={() => setProjectDetailTab("chat")}
          >
            聊天
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={projectDetailTab === "files"}
            className={projectDetailTab === "files" ? "is-active" : ""}
            onClick={() => setProjectDetailTab("files")}
          >
            文件
          </button>
        </div>

        {projectDetailTab === "chat" ? (
          <>
            {projectChats.length > 0 ? (
              <section className="project-home-chat-list" aria-label="项目聊天">
                {projectChats.map((item) => {
                  const preview = [...item.messages]
                    .reverse()
                    .find((message) => message.role === "user")?.content.trim() || "项目对话";
                  return (
                    <button type="button" key={item.id} onClick={() => openConversation(item)}>
                      <span>
                        <strong>{item.title}</strong>
                        <small>{preview}</small>
                      </span>
                      <time>{item.messages.length} 条</time>
                    </button>
                  );
                })}
              </section>
            ) : (
              <div className="project-home-empty">
                <p>还没有项目聊天。可以从可行性报告框架、材料缺口检查或政策依据梳理开始。</p>
                <button type="button" className="secondary-button" onClick={() => startNewChat(selectedProject.id)}>
                  <MessageSquarePlus aria-hidden="true" />开始第一个聊天
                </button>
              </div>
            )}
          </>
        ) : (
          <section className="project-section project-sources-section">
              <header>
                <div>
                  <h3>项目文件</h3>
                  <p>这些文件会由项目内 Agent 在需要时自动读取。</p>
                </div>
                <label className="secondary-button project-upload-button">
                  <Upload aria-hidden="true" />
                  添加文件
                  <input
                    type="file"
                    multiple
                    onChange={(event) => {
                      if (event.currentTarget.files) void uploadProjectFiles(event.currentTarget.files, selectedProject.id);
                      event.currentTarget.value = "";
                    }}
                  />
                </label>
              </header>
              {selectedProject.files.length > 0 ? (
                <div className="project-file-list">
                  {selectedProject.files.map((file) => (
                    <div className="project-file-row" key={file.path}>
                      <button
                        type="button"
                        className="project-file-main"
                        onClick={() =>
                          void openFile(file.path).then((opened) => {
                            if (opened) setArtifactTabWithUrl("files");
                          })
                        }
                      >
                        <span className="project-file-icon" aria-hidden="true">{iconForAttachment(file.kind)}</span>
                        <span><strong>{file.name}</strong><small>{formatBytes(file.size)} · {formatProjectDate(file.modified)}</small></span>
                      </button>
                      <button
                        type="button"
                        className="project-file-delete"
                        aria-label={`移除 ${file.name}`}
                        title="从项目中移除"
                        onClick={() => void deleteProjectFile(file.path)}
                      >
                        <Trash2 aria-hidden="true" />
                      </button>
                    </div>
                  ))}
                </div>
              ) : (
                <label className="project-source-empty">
                  <Upload aria-hidden="true" />
                  <strong>把相关材料放进项目</strong>
                  <span>支持 PDF、Word、Excel、PPT、图片、录音和文本文件</span>
                  <input
                    type="file"
                    multiple
                    onChange={(event) => {
                      if (event.currentTarget.files) void uploadProjectFiles(event.currentTarget.files, selectedProject.id);
                      event.currentTarget.value = "";
                    }}
                  />
                </label>
              )}
          </section>
        )}
      </div>
    );
  }

  function renderProjectCreateDialog() {
    return (
      <div className="project-dialog-layer" role="presentation" onMouseDown={() => setProjectCreateOpen(false)}>
        <form
          className="project-create-dialog"
          role="dialog"
          aria-modal="true"
          aria-label="创建项目"
          onSubmit={createProject}
          onMouseDown={(event) => event.stopPropagation()}
        >
          <header>
            <div><h2>创建项目</h2><p>项目会把资料、指令和对话限制在同一个工作范围内。</p></div>
            <button type="button" aria-label="关闭" onClick={() => setProjectCreateOpen(false)}><X aria-hidden="true" /></button>
          </header>
          <label>
            <span>项目名称</span>
            <input
              autoFocus
              placeholder="例如：某科技项目可行性报告"
              value={projectCreateForm.name}
              onChange={(event) => setProjectCreateForm((form) => ({ ...form, name: event.target.value }))}
            />
          </label>
          <label>
            <span>项目指令 <small>可稍后补充</small></span>
            <textarea
              rows={4}
              placeholder="说明项目背景、材料使用规则和希望 Agent 遵循的写作要求。"
              value={projectCreateForm.instructions}
              onChange={(event) =>
                setProjectCreateForm((form) => ({ ...form, instructions: event.target.value }))
              }
            />
          </label>
          <div className="project-memory-note">
            <CheckCircle2 aria-hidden="true" />
            <span><strong>仅项目范围</strong><small>项目聊天只自动使用本项目指令和资料，不会串用其他项目。</small></span>
          </div>
          <footer>
            <button type="button" className="secondary-button" onClick={() => setProjectCreateOpen(false)}>取消</button>
            <button type="submit" className="primary-button" disabled={busy || !projectCreateForm.name.trim()}>创建项目</button>
          </footer>
        </form>
      </div>
    );
  }

  function renderHistoryMenu() {
    if (!historyMenu) return null;
    const item = conversationHistory.find((conversation) => conversation.id === historyMenu.id);
    if (!item) return null;
    const confirmingDelete = deleteConfirmConversationId === item.id;
    return (
      <div
        className="history-menu-popover"
        role="menu"
        aria-label={`${item.title} 的历史对话操作`}
        style={{ left: historyMenu.x, top: historyMenu.y }}
        onPointerDown={(event) => event.stopPropagation()}
      >
        <button type="button" role="menuitem" onClick={() => toggleConversationPin(item.id)}>
          {item.pinned ? <PinOff aria-hidden="true" /> : <Pin aria-hidden="true" />}
          <span>{item.pinned ? "取消置顶" : "置顶"}</span>
        </button>
        <button type="button" role="menuitem" onClick={() => startRenameConversation(item.id)}>
          <Pencil aria-hidden="true" />
          <span>重命名</span>
        </button>
        <button
          type="button"
          role="menuitem"
          className={`history-menu-danger ${confirmingDelete ? "is-confirming" : ""}`}
          onClick={() => deleteConversation(item.id)}
        >
          <Trash2 aria-hidden="true" />
          <span>{confirmingDelete ? "确认删除" : "删除"}</span>
        </button>
      </div>
    );
  }

  function jumpToChatTurn(messageIndex: number) {
    const thread = chatThreadRef.current;
    const target = thread?.querySelector<HTMLElement>(
      `[data-chat-message-index="${messageIndex}"]`
    );
    if (!target) return;
    setActiveChatTurnIndex(messageIndex);
    target.scrollIntoView({
      behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth",
      block: "start"
    });
  }

  function renderChatTurnNavigator() {
    if (chatUserTurns.length < 2) return null;
    return (
      <aside className="chat-turn-navigator" aria-label="对话历史导航">
        <div className="chat-turn-rail" aria-hidden="true">
          {chatUserTurns.map((turn) => (
            <span
              key={turn.messageIndex}
              className={activeChatTurnIndex === turn.messageIndex ? "is-active" : ""}
            />
          ))}
        </div>
        <nav className="chat-turn-popover" aria-label="按你的提问跳转">
          <header>
            <strong>本次对话</strong>
            <span>{chatUserTurns.length} 个提问</span>
          </header>
          <div className="chat-turn-list">
            {chatUserTurns.map((turn) => (
              <button
                key={turn.messageIndex}
                type="button"
                className={activeChatTurnIndex === turn.messageIndex ? "is-active" : ""}
                aria-current={activeChatTurnIndex === turn.messageIndex ? "step" : undefined}
                title={turn.label}
                onClick={() => jumpToChatTurn(turn.messageIndex)}
              >
                <span>{turn.label}</span>
              </button>
            ))}
          </div>
        </nav>
      </aside>
    );
  }

  function renderAgent() {
    return (
      <div className={`chat-page ${activityOpen ? "has-activity" : ""}`}>
        <div className="chat-workarea">
          <section ref={chatThreadRef} className="chat-thread" aria-label="智能体对话">
            <div className="chat-messages" aria-live="polite">
              {chatMessages.map((message, index) => {
                const activityRecord =
                  message.role === "assistant" ? activityRecordForMessage(index) : null;
                const isActivityMessage = Boolean(activityRecord);
                const approvalEvent =
                  message.role === "assistant" && index === chatMessages.length - 1 && activityRecord
                    ? pendingApprovalEvent(activityRecord)
                    : null;
                const displayMessage = sanitizeChatMessage(message);
                const content =
                  displayMessage.content ||
                  (message.role === "assistant" && busy && !isActivityMessage ? "正在生成…" : "");
                return (
                  <article
                    key={`${message.role}-${index}`}
                    data-chat-message-index={index}
                    data-chat-role={message.role}
                    className={`chat-message chat-${message.role} ${
                      editingMessageIndex === index ? "is-editing" : ""
                    }`}
                  >
                    <span className="chat-role">{message.role === "user" ? "你" : "智能体"}</span>
                    {activityRecord ? renderActivityTrigger(index, activityRecord) : null}
                    {message.role === "user" && editingMessageIndex === index ? (
                      <form
                        className="chat-message-editor"
                        onSubmit={(event) => void resendEditedUserMessage(event, index)}
                      >
                        <textarea
                          autoFocus
                          value={editMessageDraft}
                          aria-label="编辑这条消息"
                          onChange={(event) => setEditMessageDraft(event.target.value)}
                          onKeyDown={(event) => {
                            if (event.key === "Escape") cancelEditingUserMessage();
                            if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
                              event.currentTarget.form?.requestSubmit();
                            }
                          }}
                        />
                        <div className="chat-message-editor-actions">
                          <span>发送后将从这里重新开始</span>
                          <button type="button" className="message-edit-cancel" onClick={cancelEditingUserMessage}>
                            取消
                          </button>
                          <button type="submit" className="message-edit-send" disabled={!editMessageDraft.trim()}>
                            发送
                          </button>
                        </div>
                      </form>
                    ) : content ? (
                      <>
                        <div className="chat-bubble">
                          {displayMessage.role === "assistant" ? (
                            <MarkdownContent content={content} onOpenFile={openLinkedFile} />
                          ) : (
                            <p>{content}</p>
                          )}
                        </div>
                        {displayMessage.role === "assistant" &&
                        index === chatMessages.length - 1 &&
                        isRetryableChatFailure(content) ? (
                          <button
                            type="button"
                            className="chat-retry-button"
                            disabled={busy}
                            onClick={() => void continueFailedTurn()}
                          >
                            <RefreshCw aria-hidden="true" />
                            继续本轮
                          </button>
                        ) : null}
                      </>
                    ) : null}
                    {message.role === "user" && editingMessageIndex !== index ? (
                      <div className="chat-message-actions" aria-label="消息操作">
                        <button
                          type="button"
                          title="复制提示词"
                          aria-label="复制提示词"
                          onClick={() => void copyText(message.content, "提示词已复制")}
                        >
                          <Copy aria-hidden="true" />
                        </button>
                        <button
                          type="button"
                          title={busy ? "当前轮结束后可编辑" : "编辑消息并从这里重新发送"}
                          aria-label="编辑消息并从这里重新发送"
                          disabled={busy}
                          onClick={() => startEditingUserMessage(index)}
                        >
                          <Pencil aria-hidden="true" />
                        </button>
                      </div>
                    ) : null}
                    {approvalEvent ? renderApprovalCard(approvalEvent, index) : null}
                  </article>
                );
              })}
            </div>
          </section>

          {renderChatTurnNavigator()}

          <section className="composer-dock" aria-label="消息输入区">
            <form
              className={`chat-compose ${dragActive ? "is-dragging" : ""} ${
                isListening ? "is-recording" : ""
              } ${attachments.length > 0 ? "has-attachments" : ""}`}
              onSubmit={sendChatMessage}
            >
              {attachments.length > 0 ? (
                <div className="attachment-list" aria-label="本轮参考附件">
                  {attachments.map((attachment) => (
                    <div
                      key={attachment.path}
                      className={`attachment-card attachment-${attachment.kind}`}
                      title={`${attachment.name}\n${attachment.path}`}
                    >
                      <button
                        type="button"
                        className="attachment-main"
                        onClick={() => copyPath(attachment.path)}
                        aria-label={`复制附件路径：${attachment.name}`}
                      >
                        <span className="attachment-thumb" aria-hidden="true">
                          {iconForAttachment(attachment.kind)}
                        </span>
                        <span className="attachment-copy">
                          <strong>{attachment.name}</strong>
                          <small>
                            {labelForAttachment(attachment.kind)} · {formatBytes(attachment.size)}
                          </small>
                          {attachment.recording_metadata?.recording_started_at ? (
                            <small>
                              录音开始：{formatRecordingStartedAt(attachment.recording_metadata.recording_started_at)}
                            </small>
                          ) : null}
                        </span>
                      </button>
                      <button
                        type="button"
                        className="attachment-remove"
                        aria-label={`移除附件：${attachment.name}`}
                        title="移除附件"
                        onClick={() => removeAttachment(attachment.path)}
                        disabled={isListening}
                      >
                        <X aria-hidden="true" />
                      </button>
                    </div>
                  ))}
                </div>
              ) : null}

              {(actionMenuOpen || suggestedSkills.length > 0) && (
                <div className="action-menu" role="menu" aria-label="添加文件和技能">
                  {suggestedSkills.length > 0 ? (
                    <div className="action-menu-section">
                      {suggestedSkills.map((skill) => (
                        <button
                          key={skill.id}
                          type="button"
                          role="menuitem"
                          onMouseDown={(event) => event.preventDefault()}
                          onClick={() => {
                            attachSkill(skill);
                            setActionMenuOpen(false);
                          }}
                        >
                          <AtSign aria-hidden="true" />
                          <span>
                            <strong>{skill.label}</strong>
                            <small>{skill.description}</small>
                          </span>
                        </button>
                      ))}
                    </div>
                  ) : (
                    <>
                      <label className="action-menu-item">
                        <Paperclip aria-hidden="true" />
                        <span>
                          <strong>添加照片和文件</strong>
                          <small>录音、图片、PDF、Word 都会保存为本地附件路径</small>
                        </span>
                        <input
                          type="file"
                          multiple
                          onChange={(event) => {
                            if (event.currentTarget.files) {
                              void attachDroppedFiles(event.currentTarget.files);
                            }
                            event.currentTarget.value = "";
                            setActionMenuOpen(false);
                          }}
                        />
                      </label>
                      <button type="button" onClick={() => attachSkillById("meeting-minutes")}>
                        <FileText aria-hidden="true" />
                        <span>
                          <strong>会议纪要</strong>
                          <small>生成内部留档版和工作提交版</small>
                        </span>
                      </button>
                    </>
                  )}
                  <div className="action-menu-search">输入 @ 可搜索已接入技能</div>
                </div>
              )}

              <label htmlFor="chat-input" className="sr-only">
                输入消息
              </label>
              <div className={`composer-input-row ${isListening ? "is-recording" : ""}`}>
                {isListening ? (
                  <>
                    <button
                      type="button"
                      className="composer-plus voice-plus"
                      aria-label="录音时暂不能添加文件"
                      disabled
                    >
                      <Plus aria-hidden="true" />
                    </button>
                    <div className="voice-waveform" aria-label="正在录音，音量波形实时显示">
                      <span className="sr-only">正在录音，点击勾号结束并转写，点击叉号取消。</span>
                      {voiceLevels.map((level, index) => (
                        <span
                          key={`${index}-${voiceLevels.length}`}
                          className="voice-waveform-bar"
                          style={{ height: `${Math.round(4 + level * 32)}px` }}
                          aria-hidden="true"
                        />
                      ))}
                    </div>
                    <button
                      type="button"
                      className="voice-cancel-button"
                      aria-label="取消录音"
                      title="取消录音"
                      onClick={cancelLocalVoiceRecording}
                    >
                      <X aria-hidden="true" />
                    </button>
                    <button
                      type="button"
                      className="voice-confirm-button"
                      aria-label="结束录音并转写"
                      title="结束录音并转写"
                      onClick={stopLocalVoiceRecording}
                    >
                      <Check aria-hidden="true" />
                    </button>
                  </>
                ) : (
                  <>
                    <button
                      type="button"
                      className={`composer-plus ${actionMenuOpen ? "is-active" : ""}`}
                      onClick={() => setActionMenuOpen((open) => !open)}
                      aria-label="添加文件和技能"
                      aria-expanded={actionMenuOpen}
                    >
                      <Plus aria-hidden="true" />
                    </button>
                    <div className="compose-input-wrap">
                      <textarea
                        id="chat-input"
                        name="chat-input"
                        rows={1}
                        value={chatInput}
                        autoComplete="off"
                        placeholder="有问题，尽管问"
                        onChange={(event) => updateChatInput(event.target.value)}
                        onPaste={(event) => {
                          if (event.clipboardData.files.length > 0) {
                            void attachDroppedFiles(event.clipboardData.files);
                          }
                        }}
                        onKeyDown={(event) => {
                          if (event.key === "Enter" && (event.metaKey || event.ctrlKey) && !event.nativeEvent.isComposing) {
                            event.preventDefault();
                            event.currentTarget.form?.requestSubmit();
                          }
                        }}
                      />
                    </div>
                    <div className="composer-model-control" ref={composerModelMenuRef}>
                      <button
                        type="button"
                        className={`composer-model-trigger ${composerModelMenuOpen ? "is-open" : ""}`}
                        onClick={() => {
                          setComposerModelMenuOpen((open) => !open);
                          setComposerSubmenu(null);
                        }}
                        aria-label="模型和推理强度"
                        aria-expanded={composerModelMenuOpen}
                      >
                        <span>{formatProfileCompactLabel(currentProfile)}</span>
                        <span className="composer-reasoning-short">
                          {reasoningOption(reasoningEffort).shortLabel}
                        </span>
                        <ChevronDown aria-hidden="true" />
                      </button>
                      {composerModelMenuOpen ? (
                        <div
                          className={`composer-model-menu ${composerSubmenu === "advanced" ? "is-advanced" : ""}`}
                          role="menu"
                        >
                          {composerSubmenu === "advanced" ? (
                            <div className="composer-advanced-panel">
                              <div className="composer-advanced-header">
                                <button
                                  type="button"
                                  onClick={() => setComposerSubmenu(null)}
                                  aria-label="收起高级设置"
                                >
                                  <strong>高级</strong>
                                  <ChevronRight aria-hidden="true" />
                                </button>
                              </div>
                              <div
                                className="composer-reasoning-slider-wrap"
                                style={{
                                  "--reasoning-progress": [
                                    "0px",
                                    "calc(20px + (100% - 40px) * 0.3333)",
                                    "calc(20px + (100% - 40px) * 0.6667)",
                                    "100%"
                                  ][REASONING_OPTIONS.findIndex((option) => option.value === reasoningEffort)],
                                  "--reasoning-thumb-position": [
                                    "20px",
                                    "calc(20px + (100% - 40px) * 0.3333)",
                                    "calc(20px + (100% - 40px) * 0.6667)",
                                    "calc(100% - 20px)"
                                  ][REASONING_OPTIONS.findIndex((option) => option.value === reasoningEffort)]
                                } as CSSProperties}
                              >
                                <input
                                  className="composer-reasoning-slider"
                                  type="range"
                                  min="0"
                                  max={REASONING_OPTIONS.length - 1}
                                  step="1"
                                  value={REASONING_OPTIONS.findIndex((option) => option.value === reasoningEffort)}
                                  onChange={(event) => {
                                    const option = REASONING_OPTIONS[Number(event.currentTarget.value)];
                                    if (option) setReasoningEffort(option.value);
                                  }}
                                  aria-label="推理强度"
                                />
                                <div className="composer-reasoning-dots" aria-hidden="true">
                                  {REASONING_OPTIONS.map((option, index) => (
                                    <span
                                      key={option.value}
                                      className={index <= REASONING_OPTIONS.findIndex((item) => item.value === reasoningEffort) ? "is-filled" : ""}
                                    />
                                  ))}
                                </div>
                                <span className="composer-reasoning-thumb" aria-hidden="true" />
                              </div>
                            </div>
                          ) : (
                            <>
                          <button
                            type="button"
                            className={composerSubmenu === "model" ? "is-active" : ""}
                            onClick={() => setComposerSubmenu((value) => (value === "model" ? null : "model"))}
                          >
                            <strong>模型</strong>
                            <span>{formatProfileLabel(currentProfile)}</span>
                            <ChevronRight aria-hidden="true" />
                          </button>
                          <button
                            type="button"
                            className={composerSubmenu === "reasoning" ? "is-active" : ""}
                            onClick={() =>
                              setComposerSubmenu((value) => (value === "reasoning" ? null : "reasoning"))
                            }
                          >
                            <strong>推理强度</strong>
                            <span>{reasoningOption(reasoningEffort).label}</span>
                            <ChevronRight aria-hidden="true" />
                          </button>
                          <div className="composer-model-menu-divider" />
                          <button
                            type="button"
                            className="composer-model-advanced"
                            onClick={() => setComposerSubmenu("advanced")}
                          >
                            <strong>高级</strong>
                            <ChevronDown aria-hidden="true" />
                          </button>

                          {composerSubmenu === "model" ? (
                            <div className="composer-model-submenu is-model" role="menu" aria-label="选择模型">
                              <div className="composer-submenu-title">模型</div>
                              {profiles.map((profile) => (
                                <button
                                  key={profile.name}
                                  type="button"
                                  onClick={() => {
                                    setAgentForm((form) => ({ ...form, profile: profile.name }));
                                    setComposerModelMenuOpen(false);
                                    setComposerSubmenu(null);
                                  }}
                                >
                                  <span>{formatProfileLabel(profile)}</span>
                                  {profile.name === agentForm.profile ? <Check aria-hidden="true" /> : null}
                                </button>
                              ))}
                            </div>
                          ) : null}

                          {composerSubmenu === "reasoning" ? (
                            <div className="composer-model-submenu" role="menu" aria-label="选择推理强度">
                              <div className="composer-submenu-title">推理强度</div>
                              {REASONING_OPTIONS.map((option) => (
                                <button
                                  key={option.value}
                                  type="button"
                                  onClick={() => {
                                    setReasoningEffort(option.value);
                                    setComposerModelMenuOpen(false);
                                    setComposerSubmenu(null);
                                  }}
                                >
                                  <span>{option.label}</span>
                                  {option.value === reasoningEffort ? <Check aria-hidden="true" /> : null}
                                </button>
                              ))}
                            </div>
                          ) : null}
                            </>
                          )}
                        </div>
                      ) : null}
                    </div>
                    {activityRunning ? (
                      <button
                        type="button"
                        className="composer-stop"
                        aria-label="停止当前轮"
                        title="停止当前轮，保留已输出内容并继续处理队列"
                        onClick={stopCurrentChatTurn}
                      >
                        <span aria-hidden="true" />
                      </button>
                    ) : (
                      <button
                        type="button"
                        className="composer-icon"
                        aria-label="本地录音转文字"
                        title={speechSupported ? "本地录音转文字" : "当前浏览器不支持本地录音"}
                        disabled={busy || !speechSupported}
                        onClick={toggleVoiceInput}
                      >
                        <Mic aria-hidden="true" />
                      </button>
                    )}
                    <button
                      type="submit"
                      className="composer-send"
                      disabled={!chatInput.trim() && attachments.length === 0}
                      aria-label={activityRunning ? "发送到等待队列" : "发送"}
                      title={activityRunning ? "当前轮运行中，本条会进入等待队列" : "发送"}
                    >
                      <SendHorizontal aria-hidden="true" />
                    </button>
                  </>
                )}
              </div>
              {selectedSkill ? (
                <div className="compose-meta">
                  <button
                    type="button"
                    className="active-skill"
                    onClick={() => {
                      setSelectedSkill(null);
                      setChatInput((value) => value.replace(selectedSkill.mention, "").trimStart());
                    }}
                    aria-label={`清除 ${selectedSkill.label} 技能`}
                  >
                    <AtSign aria-hidden="true" />
                    {selectedSkill.label}
                    <span>清除</span>
                  </button>
                </div>
              ) : null}
            </form>
            <div className="composer-underbar">
              <p>
                {queuedChatCount > 0
                  ? `队列中还有 ${queuedChatCount} 条消息等待处理。`
                  : "智能体存在幻觉，可能会犯错，请核查重要信息。"}
              </p>
              {chatMessages.length > 1 ? (
                <button
                  type="button"
                  className="clear-chat-button"
                  disabled={busy}
                  onClick={() => {
                    setChatMessages([
                      {
                        role: "assistant",
                        content:
                          "对话已清空。你可以继续让我读文件、生成会议纪要，或者整理当前工作。"
                      }
                    ]);
                    setAttachments([]);
                    setSelectedSkill(null);
                    setActionMenuOpen(false);
                    setActivityRecords({});
                    setActivityPanelMessageIndex(null);
                    setActivityEvents([]);
                    setActivityOpen(false);
                    setActivityMessageIndex(null);
                    setActivityStartedAt(null);
                    setActivityRunning(false);
                    setActivityElapsedMs(0);
                    setStatus({ tone: "success", text: "对话已清空" });
                  }}
                >
                  清空
                </button>
              ) : null}
            </div>
          </section>
        </div>
        {activityOpen ? renderActivityPanel() : null}
      </div>
    );
  }

  function renderActivityTrigger(messageIndex: number, record: ActivityRecord) {
    if (record.events.length === 0 && record.elapsedMs === 0 && messageIndex !== activityMessageIndex) return null;
    const isCurrentActivity = messageIndex === activityMessageIndex && activityRunning;
    const isSelected = activityOpen && panelActivityIndex === messageIndex;
    const label = isCurrentActivity ? "处理中" : "已处理";
    const elapsedLabel = formatActivityDuration(record.elapsedMs);
    return (
      <button
        type="button"
        className={`activity-trigger ${isSelected ? "is-open" : ""}`}
        aria-expanded={isSelected}
        aria-controls="activity-panel"
        onClick={() => {
          setActivityPanelMessageIndex(messageIndex);
          setActivityOpen((open) => (panelActivityIndex === messageIndex ? !open : true));
        }}
      >
        {isCurrentActivity ? (
          <Loader2 className="activity-trigger-spinner spin" aria-hidden="true" />
        ) : null}
        <span>
          {label} {elapsedLabel}
        </span>
        <ChevronRight className="activity-trigger-chevron" aria-hidden="true" />
      </button>
    );
  }

  function renderApprovalCard(item: AgentActivityEvent, messageIndex: number) {
    const commandText = item.command ?? item.tool_name ?? "命令";
    const preview = (item.approval_preview || item.content || "").trim();
    const batchCommands = item.approval_batch_commands ?? [];
    return (
      <div className="chat-approval-card" role="group" aria-label="需要确认执行终端命令">
        <div className="chat-approval-copy">
          <strong>需要你确认执行这批工具调用</strong>
          <span>
            {item.risk_category ? `${item.risk_category} · ` : ""}
            {item.detail || "该命令需要明确确认后才会继续。"}
          </span>
          {item.approval_batch_count ? (
            <span>
              本批共 {item.approval_batch_count} 个工具调用；确认后从当前命令继续执行剩余{" "}
              {item.approval_batch_remaining ?? 1} 个。
            </span>
          ) : null}
        </div>
        {batchCommands.length > 1 ? (
          <pre>
            <code>{batchCommands.map((entry) => entry.command || "").filter(Boolean).join("\n")}</code>
          </pre>
        ) : (
          <pre>
            <code>{preview || commandText}</code>
          </pre>
        )}
        <button
          type="button"
          disabled={activityRunning}
          onClick={() => void approvePendingToolBatch(item, messageIndex)}
        >
          {activityRunning ? "等待本轮结束" : "确认执行"}
        </button>
      </div>
    );
  }

  function renderActivityPanel() {
    const events = panelActivityRecord?.events ?? [];
    const displayItems = buildActivityDisplayItems(events);
    const elapsedLabel = formatActivityDuration(panelActivityRecord?.elapsedMs ?? 0);
    const isCurrentActivity = panelActivityIndex === activityMessageIndex && activityRunning;
    return (
      <aside className="activity-panel" id="activity-panel" aria-label="智能体活动">
        <header className="activity-header">
          <div>
            <h3>
              活动
              <span> · {elapsedLabel}</span>
            </h3>
            <p>{isCurrentActivity ? "正在处理当前请求" : "本轮处理过程"}</p>
          </div>
          <button type="button" onClick={() => setActivityOpen(false)} aria-label="关闭活动面板">
            <X aria-hidden="true" />
          </button>
        </header>

        <div className="activity-body">
          {displayItems.length === 0 ? (
            <div className="activity-empty">
              <Loader2 aria-hidden="true" className={isCurrentActivity ? "spin" : ""} />
              <span>{isCurrentActivity ? "正在准备活动记录…" : "暂无活动记录"}</span>
            </div>
          ) : (
            <div className="activity-timeline">
              {displayItems.map((item) =>
                item.kind === "group" ? renderActivityGroup(item) : renderActivityEvent(item.event, item.key)
              )}
            </div>
          )}
        </div>
      </aside>
    );
  }

  function renderActivityEvent(item: AgentActivityEvent, key: string) {
    return (
      <article key={key} className={`activity-item activity-${item.phase}`}>
        <span className="activity-marker" aria-hidden="true">
          {iconForActivity(item.phase)}
        </span>
        <div className="activity-content">
          {item.activity_type === "command" ? (
            renderCommandActivity(item)
          ) : item.activity_type === "file_edit" ? (
            renderFileEditActivity(item)
          ) : (
            <>
              <div className="activity-kicker">
                <span>{activityPhaseLabel(item.phase)}</span>
                {item.step ? <span>第 {item.step} 步</span> : null}
              </div>
              <h4>{item.title}</h4>
              {item.tool_name ? <code translate="no">{item.tool_name}</code> : null}
              {item.detail ? <p>{item.detail}</p> : null}
              {item.content ? <p className="activity-stream-text">{item.content}</p> : null}
            </>
          )}
        </div>
      </article>
    );
  }

  function renderActivityGroup(group: Extract<ActivityDisplayItem, { kind: "group" }>) {
    const phase: AgentActivityEvent["phase"] = group.status === "error" ? "error" : "observation";
    return (
      <article key={group.key} className={`activity-item activity-${phase}`}>
        <span className="activity-marker" aria-hidden="true">
          {iconForActivity(phase)}
        </span>
        <div className="activity-content">
          <details className={`activity-summary-card is-${group.status}`} open={group.open}>
            <summary>
              <span className="activity-summary-icon" aria-hidden="true">
                {group.icon}
              </span>
              <span className="activity-summary-copy">
                <strong>{group.title}</strong>
                <small>{group.detail}</small>
              </span>
              <ChevronRight aria-hidden="true" />
            </summary>
            <div className="activity-summary-body">
              {group.events.map((item, index) => renderActivityGroupRow(item, index))}
            </div>
          </details>
        </div>
      </article>
    );
  }

  function renderActivityGroupRow(item: AgentActivityEvent, index: number) {
    const output = (item.approval_preview || item.content || "").trimEnd();
    const detailText = activityRowDetail(item);
    const copyPayload = item.activity_type === "file_edit" ? item.content || "" : activityRowCopyText(item);
    return (
      <div className="activity-summary-row" key={`${item.id ?? item.title}-${index}`}>
        <div className="activity-summary-row-head">
          <span>
            <strong>{activityRowTitle(item)}</strong>
            {detailText ? <small>{detailText}</small> : null}
          </span>
          {copyPayload ? (
            <button type="button" aria-label="复制详情" onClick={() => copyText(copyPayload, "详情已复制")}>
              <Copy aria-hidden="true" />
            </button>
          ) : null}
        </div>
        {item.activity_type === "file_edit" ? (
          <div className="activity-summary-diff">
            <span className="activity-file-stats" aria-label={`新增 ${item.additions ?? 0} 行，删除 ${item.deletions ?? 0} 行`}>
              <b className="is-add">+{item.additions ?? 0}</b>
              <b className="is-del">-{item.deletions ?? 0}</b>
            </span>
            {item.content ? (
              <pre>
                <code>{compactActivityOutput(item.content)}</code>
              </pre>
            ) : null}
          </div>
        ) : output || item.command || item.tool_name ? (
          <pre>
            <code>
              {activityRowCommandLine(item)}
              {output ? `\n\n${compactActivityOutput(output)}` : ""}
            </code>
          </pre>
        ) : null}
      </div>
    );
  }

  function renderCommandActivity(item: AgentActivityEvent) {
    const status = item.command_status ?? "running";
    const statusLabel =
      status === "success"
        ? "成功"
        : status === "error"
          ? "失败"
          : status === "approval_required"
            ? "等待确认"
            : "运行中";
    const commandText = item.command ?? item.tool_name ?? "命令";
    const isModelRequest = commandText.startsWith("LLM tool planning");
    const isHeartbeat = item.title === "运行状态";
    const isWritePreview =
      item.title === "写入文件" ||
      item.title === "生成DOCX" ||
      commandText.startsWith("write_text_file") ||
      commandText.startsWith("create_docx_from_markdown");
    const isApproval = status === "approval_required" || item.approval_required;
    const cardTitle = isHeartbeat
      ? "正在处理"
      : isModelRequest
        ? "模型请求"
        : isApproval
          ? "等待终端确认"
          : isWritePreview
            ? item.title
            : "已运行命令";
    const activityLabel = isHeartbeat ? "Live" : isModelRequest ? "LLM" : isWritePreview ? "Write" : "Shell";
    const output = (item.approval_preview || item.content || "").trimEnd();
    const heartbeatSummary = isHeartbeat ? output.replace(/\s+/g, " ").trim() : "";
    const commandLine = activityRowCommandLine(item);
    return (
      <details
        className={`activity-command-card is-${status}`}
        open={status === "running" || status === "approval_required" || status === "error"}
      >
        <summary>
          <span className="activity-command-icon" aria-hidden="true">
            {isHeartbeat && status === "running" ? (
              <Loader2 className="spin" />
            ) : isModelRequest ? (
              "✦"
            ) : isApproval ? (
              "!"
            ) : isWritePreview ? (
              "✎"
            ) : (
              "$"
            )}
          </span>
          <span className="activity-command-summary">
            <strong>{cardTitle}</strong>
            <small>
              {item.risk_category ? `${item.risk_category} · ` : ""}
              {item.detail ? `${item.detail} · ` : ""}
              {heartbeatSummary ? `${heartbeatSummary} · ` : ""}
              {statusLabel}
            </small>
          </span>
          <ChevronRight aria-hidden="true" />
        </summary>
        <div className="activity-command-body">
          <div className="activity-command-toolbar">
            <span>
              {activityLabel}
              {item.risk_category ? <b className="activity-risk-pill">{item.risk_category}</b> : null}
            </span>
            <button
              type="button"
              aria-label={
                isModelRequest
                  ? "复制模型请求摘要"
                  : isApproval
                    ? "复制审批预览"
                    : isWritePreview
                      ? "复制写入摘要"
                      : "复制命令"
              }
              onClick={() =>
                copyText(
                  isApproval ? output || commandText : commandLine,
                  isModelRequest
                    ? "模型请求摘要已复制"
                    : isApproval
                      ? "审批预览已复制"
                      : isWritePreview
                        ? "写入摘要已复制"
                        : "命令已复制"
                )
              }
            >
              <Copy aria-hidden="true" />
            </button>
          </div>
          <pre>
            <code>
              {commandLine}
              {output ? `\n\n${output}` : ""}
            </code>
          </pre>
        </div>
      </details>
    );
  }

  function renderFileEditActivity(item: AgentActivityEvent) {
    const status = item.command_status ?? "running";
    const statusLabel =
      status === "success" ? "已保存" : status === "error" ? "写入失败" : "写入中";
    const filePath = item.file_path ?? item.detail ?? "文件";
    const fileName = filePath.split(/[\\/]/).filter(Boolean).pop() ?? filePath;
    const additions = item.additions ?? 0;
    const deletions = item.deletions ?? 0;
    const diff = item.content ?? "";
    return (
      <details className={`activity-file-card is-${status}`} open={status === "running"}>
        <summary>
          <span className="activity-file-icon" aria-hidden="true">
            ±
          </span>
          <span className="activity-file-summary">
            <strong>
              已编辑 <span>{fileName}</span>
            </strong>
            <small title={filePath}>
              {filePath} · {statusLabel}
            </small>
          </span>
          <span className="activity-file-stats" aria-label={`新增 ${additions} 行，删除 ${deletions} 行`}>
            <b className="is-add">+{additions}</b>
            <b className="is-del">-{deletions}</b>
          </span>
          <ChevronRight aria-hidden="true" />
        </summary>
        <div className="activity-file-body">
          <div className="activity-command-toolbar">
            <span>File diff</span>
            <button
              type="button"
              aria-label="复制 diff"
              onClick={() => copyText(diff, "diff 已复制")}
            >
              <Copy aria-hidden="true" />
            </button>
          </div>
          <pre className="activity-diff-preview">
            <code>
              {diff
                ? diff.split("\n").map((line, index) => {
                    const className = diffLineClassName(line);
                    return (
                      <span key={`${index}-${line.slice(0, 16)}`} className={className}>
                        <span className="activity-diff-line-number">{index + 1}</span>
                        <span className="activity-diff-line-text">{line || " "}</span>
                      </span>
                    );
                  })
                : "等待写入结果…"}
            </code>
          </pre>
        </div>
      </details>
    );
  }

  function diffLineClassName(line: string) {
    if (line.startsWith("+++") || line.startsWith("---")) return "activity-diff-line is-file";
    if (line.startsWith("@@")) return "activity-diff-line is-hunk";
    if (line.startsWith("+")) return "activity-diff-line is-add";
    if (line.startsWith("-")) return "activity-diff-line is-del";
    return "activity-diff-line";
  }

  function renderMore() {
    return (
      <div className="more-page">
        <header className="more-hero">
          <div>
            <h2>更多功能</h2>
            <p>把低频但重要的办公功能收在这里，常用工作仍然从对话和文件库开始。</p>
          </div>
        </header>

        <div className="more-grid" aria-label="功能入口">
          <button type="button" className="more-card" onClick={() => setViewWithUrl("sync")}>
            <span className="panel-icon">
              <Cloud aria-hidden="true" />
            </span>
            <span>
              <strong>临时同步区</strong>
              <small>同一账号的电脑间传文字和文件；文件保留 1 小时。</small>
            </span>
            <ChevronRight aria-hidden="true" />
          </button>

          <button
            type="button"
            className="more-card"
            onClick={() => setArtifactTabWithUrl("meeting")}
          >
            <span className="panel-icon">
              <FileText aria-hidden="true" />
            </span>
            <span>
              <strong>会议纪要</strong>
              <small>按会议查看 ASR 转写稿、内部留档版、工作提交版和 DOCX。</small>
            </span>
            <ChevronRight aria-hidden="true" />
          </button>

          <button type="button" className="more-card" onClick={() => setViewWithUrl("transcribe")}>
            <span className="panel-icon">
              <Mic aria-hidden="true" />
            </span>
            <span>
              <strong>实时转写</strong>
              <small>现场录音、自动断句、本地 ASR 转写并保存成稿。</small>
            </span>
            <ChevronRight aria-hidden="true" />
          </button>
        </div>
      </div>
    );
  }

  function renderTemporarySync() {
    const storedText = temporarySync?.text.content ?? "";
    const textChanged = temporarySyncTextDirty || temporarySyncText !== storedText;
    return (
      <div className="temporary-sync-page">
        <header className="temporary-sync-heading">
          <div>
            <h2>临时同步区</h2>
            <p>登录同一账号的设备会看到相同内容。页面每 10 秒自动刷新一次。</p>
          </div>
          <button
            type="button"
            className="secondary-button"
            onClick={() => void refreshTemporarySync()}
            disabled={temporarySyncBusy}
          >
            <RefreshCw aria-hidden="true" className={temporarySyncBusy ? "spin" : ""} />
            刷新
          </button>
        </header>

        <section className="sync-text-section" aria-labelledby="sync-text-title">
          <div className="sync-section-heading">
            <div>
              <h3 id="sync-text-title">同步文字</h3>
              <p>只保留最新一条，不限保存时长；再次保存会覆盖上一条。</p>
            </div>
            {temporarySync?.text.updated_at ? (
              <span>更新于 {formatProjectDate(temporarySync.text.updated_at)}</span>
            ) : null}
          </div>
          <textarea
            rows={8}
            value={temporarySyncText}
            placeholder="在一台电脑粘贴文字并保存，另一台电脑打开这里即可复制。"
            aria-label="要同步的文字"
            onChange={(event) => {
              setTemporarySyncText(event.target.value);
              setTemporarySyncTextDirty(true);
              temporarySyncTextDirtyRef.current = true;
            }}
          />
          <div className="sync-text-actions">
            <button
              type="button"
              className="primary-button"
              disabled={temporarySyncBusy || !textChanged}
              onClick={() => void saveTemporarySyncText()}
            >
              {temporarySyncBusy ? <Loader2 aria-hidden="true" className="spin" /> : <Cloud aria-hidden="true" />}
              保存并同步
            </button>
            <button
              type="button"
              className="secondary-button"
              disabled={!temporarySyncText}
              onClick={() => void copyText(temporarySyncText, "同步文字已复制")}
            >
              <Copy aria-hidden="true" />
              复制文字
            </button>
            {textChanged ? <span className="sync-unsaved-note">有尚未同步的修改</span> : null}
          </div>
        </section>

        <section className="sync-files-section" aria-labelledby="sync-files-title">
          <div className="sync-section-heading">
            <div>
              <h3 id="sync-files-title">临时文件</h3>
              <p>可同时传多个文件，单个不超过 100 MB；上传满 1 小时后自动删除。</p>
            </div>
            <span>{temporarySync?.files.length ?? 0} 个有效文件</span>
          </div>

          <label className={`sync-file-picker ${temporarySyncBusy ? "is-disabled" : ""}`}>
            <Upload aria-hidden="true" />
            <span>
              <strong>{temporarySyncBusy ? "正在处理文件…" : "选择要同步的文件"}</strong>
              <small>支持多选，上传后另一台电脑刷新即可下载</small>
            </span>
            <input
              type="file"
              multiple
              disabled={temporarySyncBusy}
              onChange={(event) => {
                if (event.currentTarget.files) {
                  void uploadTemporarySyncFiles(event.currentTarget.files);
                }
                event.currentTarget.value = "";
              }}
            />
          </label>

          <div className="sync-file-list" aria-live="polite">
            {temporarySync?.files.length ? (
              temporarySync.files.map((file) => (
                <article className="sync-file-row" key={file.id}>
                  <span className="sync-file-icon">
                    <FileText aria-hidden="true" />
                  </span>
                  <span className="sync-file-details">
                    <strong title={file.name}>{file.name}</strong>
                    <small>
                      {formatBytes(file.size)} · 上传于 {formatProjectDate(file.uploaded_at)}
                    </small>
                  </span>
                  <span className="sync-expiry">
                    <Clock3 aria-hidden="true" />
                    {formatTemporarySyncRemaining(file.expires_at, temporarySyncClock)}
                  </span>
                  <span className="sync-file-actions">
                    <a
                      className="secondary-button sync-download-button"
                      href={file.download_url}
                      download={file.name}
                    >
                      <Download aria-hidden="true" />
                      下载
                    </a>
                    <button
                      type="button"
                      className="text-button sync-delete-button"
                      disabled={temporarySyncBusy}
                      onClick={() => void deleteTemporarySyncFile(file.id)}
                    >
                      <Trash2 aria-hidden="true" />
                      删除
                    </button>
                  </span>
                </article>
              ))
            ) : (
              <div className="sync-empty-state">
                <Cloud aria-hidden="true" />
                <strong>还没有临时文件</strong>
                <span>从任一台已登录设备上传，文件会在这里保留 1 小时。</span>
              </div>
            )}
          </div>
        </section>
      </div>
    );
  }

  function renderSkills() {
    return (
      <div className="stack">
        <section className="panel">
          <PanelHeader icon={<Sparkles aria-hidden="true" />} title="技能目录" />
          <div className="context-strip">
            <span>逐项开关；关闭后不会向模型暴露，也不能调用</span>
            <span>点击任一技能可查看说明；管理员可直接编辑 SKILL.md</span>
          </div>
          <div className="skill-catalog">
            {skills.map((skill) => (
              <article key={skill.id} className="skill-card">
                <button
                  type="button"
                  className="skill-card-summary"
                  onClick={() => void openSkillInstructions(skill)}
                  aria-label={`查看 ${skill.label} 技能说明`}
                >
                  <div>
                    <h3>{skill.label}</h3>
                    <code translate="no">{skill.mention}</code>
                    <p>{skill.description}</p>
                  </div>
                  <ChevronRight aria-hidden="true" />
                </button>
                <div className="skill-card-actions">
                  <label className="skill-switch">
                    <input
                      type="checkbox"
                      checked={skill.enabled}
                      disabled={busy}
                      onChange={(event) => setSkillEnabled(skill, event.target.checked)}
                    />
                    <span aria-hidden="true" />
                    <b>{skill.enabled ? "已启用" : "已关闭"}</b>
                  </label>
                  <button type="button" className="text-button" onClick={() => void openSkillInstructions(skill)}>
                    <Eye aria-hidden="true" />查看说明
                  </button>
                <button
                  type="button"
                  className="secondary-button"
                  onClick={() => {
                    attachSkill(skill);
                    setViewWithUrl("agent");
                  }}
                >
                  <AtSign aria-hidden="true" />
                  在对话中使用
                </button>
                </div>
              </article>
            ))}
          </div>
          {skillInstructionsLoading ? <p className="muted skill-detail-loading">正在读取技能说明…</p> : null}
          {skillInstructions ? (
            <section className="skill-detail-panel" aria-label={`${skillInstructions.skill_id} 技能说明`}>
              <div className="skill-detail-head">
                <div>
                  <h3>{skills.find((item) => item.id === skillInstructions.skill_id)?.label || skillInstructions.skill_id}</h3>
                  <p>路径：<code translate="no">{skillInstructions.path}</code></p>
                </div>
                <button type="button" className="text-button" onClick={() => setSkillInstructions(null)}>
                  <X aria-hidden="true" />收起
                </button>
              </div>
              {skillInstructions.editable ? (
                <>
                  <textarea
                    className="skill-instructions-editor"
                    value={skillInstructionsDraft}
                    spellCheck={false}
                    aria-label="技能 Markdown 说明"
                    onChange={(event) => setSkillInstructionsDraft(event.target.value)}
                  />
                  <div className="form-actions">
                    <span className="muted">保存后，新一轮调用会读取此规则；业务偏好请放到对应的专用设置中。</span>
                    <button type="button" className="primary-button" disabled={busy} onClick={() => void saveSkillInstructions()}>
                      <Check aria-hidden="true" />保存 SKILL.md
                    </button>
                  </div>
                </>
              ) : (
                <pre className="skill-instructions-preview">{skillInstructions.content}</pre>
              )}
            </section>
          ) : null}
        </section>
      </div>
    );
  }

  function renderMeetingArtifacts() {
    return (
      <div className="meeting-archive-page">
        <section className="meeting-list-panel" aria-label="会议列表">
          <div className="meeting-list-head">
            <div>
              <h2>会议纪要</h2>
              <p>每个会议归档为一组文件，点击会议后查看对应产出。</p>
            </div>
            <Badge>{meetingGroups.length} 场会议</Badge>
          </div>

          <div className="meeting-group-list">
            {meetingGroups.length === 0 ? (
              <EmptyState text="还没有会议纪要产出。通过对话生成后会自动出现在这里。" />
            ) : (
              meetingGroups.map((group) => {
                const fileCount = [
                  group.asr,
                  group.internal,
                  group.work,
                  group.workDocx
                ].filter(Boolean).length;
                return (
                  <button
                    key={group.key}
                    type="button"
                    className={`meeting-group-row ${activeMeetingGroup?.key === group.key ? "is-active" : ""}`}
                    onClick={() => setMeetingOutputKey(group.key)}
                  >
                    <span>
                      <strong>{group.title}</strong>
                      <small>{formatMeetingDate(group.meetingTime)}</small>
                    </span>
                    <Badge tone={fileCount >= 4 ? "success" : "neutral"}>{fileCount}/4</Badge>
                  </button>
                );
              })
            )}
          </div>
        </section>

        <section className="meeting-detail-panel" aria-label="会议文件">
          {activeMeetingGroup ? (
            <>
              <div className="meeting-detail-top">
                <div className="meeting-detail-head">
                  <div>
                    <h3>{activeMeetingGroup.title}</h3>
                    <p>双击卡片或点“查看”进入文件库预览；也可以复制路径、系统打开或在访达中显示。</p>
                  </div>
                  <button
                    type="button"
                    className="secondary-button meeting-sync-trigger"
                    disabled={!activeMeetingGroup.manifestPath}
                    title={activeMeetingGroup.manifestPath ? "将整组会议纪要同步到项目资料" : "该旧归档缺少 manifest.json，暂不能一键同步"}
                    onClick={() => {
                      setMeetingSyncProjectId((current) =>
                        current && projects.some((project) => project.id === current)
                          ? current
                          : activeProjectId && projects.some((project) => project.id === activeProjectId)
                            ? activeProjectId
                            : projects[0]?.id ?? ""
                      );
                      setMeetingSyncOpen((open) => !open);
                      setMeetingSyncMessage("");
                    }}
                  >
                    <FolderOpen aria-hidden="true" />
                    同步到项目
                  </button>
                </div>
                {meetingSyncOpen ? (
                  <div className="meeting-sync-panel" aria-label="同步会议纪要到项目">
                    {projects.length > 0 ? (
                      <>
                        <label>
                          <span>目标项目</span>
                          <select
                            value={meetingSyncProjectId}
                            onChange={(event) => {
                              setMeetingSyncProjectId(event.target.value);
                              setMeetingSyncMessage("");
                            }}
                          >
                            {projects.map((project) => (
                              <option key={project.id} value={project.id}>{project.name}</option>
                            ))}
                          </select>
                        </label>
                        <button
                          type="button"
                          className="primary-button"
                          disabled={busy || !meetingSyncProjectId}
                          onClick={() => void syncActiveMeetingToProject()}
                        >
                          {busy ? <Loader2 aria-hidden="true" className="spin" /> : <FolderOpen aria-hidden="true" />}
                          {busy ? "同步中…" : "一键同步"}
                        </button>
                      </>
                    ) : (
                      <div className="meeting-sync-empty">
                        <span>还没有可接收纪要的项目。</span>
                        <button type="button" className="text-button" onClick={showProjectList}>先创建项目</button>
                      </div>
                    )}
                    {meetingSyncMessage ? (
                      <p className="meeting-sync-result" role="status">{meetingSyncMessage}</p>
                    ) : null}
                  </div>
                ) : null}
              </div>

              <div className="version-grid meeting-version-grid">
                <VersionCard
                  title="内部留档版"
                  description="给自己本地参考，允许保留不确定信息和详细沟通过程。"
                  path={activeMeetingGroup.internal?.path}
                  onView={openFileInLibrary}
                  onCopy={copyPath}
                  onOpen={openLocalFile}
                  onReveal={revealLocalFile}
                />
                <VersionCard
                  title="ASR转写稿"
                  description="录音转文本的标准命名副本，便于追溯和继续整理。"
                  path={activeMeetingGroup.asr?.path}
                  onView={openFileInLibrary}
                  onCopy={copyPath}
                  onOpen={openLocalFile}
                  onReveal={revealLocalFile}
                />
                <VersionCard
                  title="工作提交版Markdown"
                  description="用于工作提交，简要、保守，只写确认过的内容。"
                  path={activeMeetingGroup.work?.path}
                  onView={openFileInLibrary}
                  onCopy={copyPath}
                  onOpen={openLocalFile}
                  onReveal={revealLocalFile}
                />
                <VersionCard
                  title="工作提交版DOCX"
                  description="按提交格式导出的 Word 文件，可直接交付或继续编辑。"
                  path={activeMeetingGroup.workDocx?.path}
                  onView={openFileInLibrary}
                  onCopy={copyPath}
                  onOpen={openLocalFile}
                  onReveal={revealLocalFile}
                />
              </div>
            </>
          ) : (
            <EmptyState text="还没有可展示的会议归档。" />
          )}
        </section>
      </div>
    );
  }

  function renderRealtimeTranscription() {
    const completedSegments = meetingLiveSegments.filter(
      (segment) => segment.text.trim() && !segment.pending && !segment.error
    );
    const errorSegments = meetingLiveSegments.filter((segment) => segment.error);
    const isRecording = meetingLiveStatus === "recording";
    const isProcessing = meetingLiveStatus === "processing" || meetingLivePending > 0;
    const transcriptText = completedSegments.map((segment) => segment.text).join("\n\n");
    return (
      <div className="realtime-page">
        <section className="realtime-control-panel" aria-label="实时转写控制">
          <div className="realtime-control-head">
            <span className="panel-icon">
              <Mic aria-hidden="true" />
            </span>
            <div>
              <h3>会议现场转写</h3>
              <p>浏览器录音，WebRTC VAD 断句，本地轻量降噪，Qwen3-ASR 中文转写。</p>
            </div>
          </div>

          <Field label="转写稿名称" htmlFor="realtime-title">
            <input
              id="realtime-title"
              type="text"
              value={meetingLiveTitle}
              disabled={isRecording}
              onChange={(event) => setMeetingLiveTitle(event.target.value)}
            />
          </Field>

          <div className={`realtime-recorder ${isRecording ? "is-recording" : ""}`}>
            {isRecording ? (
              <div className="realtime-meter">
                <div className="voice-waveform" aria-label="实时麦克风音量">
                  {meetingLiveLevels.map((level, index) => (
                    <span
                      key={`meeting-live-${index}`}
                      className="voice-waveform-bar"
                      style={{ height: `${Math.round(4 + level * 36)}px` }}
                      aria-hidden="true"
                    />
                  ))}
                </div>
              </div>
            ) : null}
            <div className="realtime-state-row">
              <Badge tone={isRecording ? "success" : isProcessing ? "warning" : "neutral"}>
                {isRecording ? "正在记录" : isProcessing ? "正在识别" : "未开始"}
              </Badge>
              <span>{completedSegments.length} 段已转写</span>
              {meetingLivePending > 0 ? <span>{meetingLivePending} 段处理中</span> : null}
            </div>
          </div>

          <div className="realtime-actions">
            {isRecording ? (
              <button type="button" className="primary-button" onClick={stopMeetingLiveTranscription}>
                <Check aria-hidden="true" />
                停止并处理
              </button>
            ) : (
              <button
                type="button"
                className="primary-button"
                onClick={startMeetingLiveTranscription}
                disabled={busy || isProcessing}
              >
                <Mic aria-hidden="true" />
                开始实时转写
              </button>
            )}
            <button
              type="button"
              className="secondary-button"
              onClick={saveMeetingLiveTranscript}
              disabled={busy || isRecording || completedSegments.length === 0}
            >
              {busy ? <Loader2 aria-hidden="true" className="spin" /> : <FileText aria-hidden="true" />}
              保存转写稿
            </button>
            <button
              type="button"
              className="text-button"
              onClick={clearMeetingLiveTranscript}
              disabled={busy || isRecording || meetingLiveSegments.length === 0}
            >
              <Trash2 aria-hidden="true" />
              清空
            </button>
          </div>

          <div className="realtime-notes">
            <span>
              满 {Math.round(realtimeTranscriptionMinSegmentMs / 1000)} 秒后由 WebRTC VAD 判断语气停顿自动成段；最长约{" "}
              {Math.round(realtimeTranscriptionMaxSegmentMs / 1000)} 秒兜底切段。
            </span>
            <span>当前片段会先做高通、低通、频谱降噪和音量归一化。</span>
          </div>

          {meetingLiveSavedPath ? (
            <div className="realtime-saved">
              <strong>已保存</strong>
              <button type="button" className="path-button" onClick={() => openLinkedFile(meetingLiveSavedPath)}>
                {meetingLiveSavedPath}
              </button>
              <div>
                <button type="button" className="text-button" onClick={() => copyPath(meetingLiveSavedPath)}>
                  <Copy aria-hidden="true" />
                  复制路径
                </button>
                <button type="button" className="text-button" onClick={sendMeetingTranscriptToAgent}>
                  <SendHorizontal aria-hidden="true" />
                  交给智能体
                </button>
              </div>
            </div>
          ) : null}
        </section>

        <section className="realtime-transcript-panel" aria-label="实时转写结果">
          <div className="realtime-transcript-head">
            <div>
              <h3>实时转写结果</h3>
              <p>
                {transcriptText
                  ? `${numberFormatter.format(transcriptText.length)} 字`
                  : "开始后会在这里持续追加文本"}
              </p>
            </div>
            {errorSegments.length > 0 ? <Badge tone="warning">{errorSegments.length} 段异常</Badge> : null}
          </div>

          <div className="realtime-segment-list">
            {meetingLiveSegments.length === 0 ? (
              <EmptyState text="还没有转写内容。开始后保持页面打开，会议语音会持续追加在这里。" />
            ) : (
              meetingLiveSegments.map((segment) => (
                <article
                  key={segment.id}
                  className={`realtime-segment ${segment.pending ? "is-pending" : ""} ${
                    segment.error ? "is-error" : ""
                  }`}
                >
                  <header>
                    <span>片段 {segment.index}</span>
                    <small>
                      {formatClockTime(segment.startedAt)}
                      {segment.asrElapsedMs ? ` · ASR ${formatActivityDuration(segment.asrElapsedMs)}` : ""}
                    </small>
                  </header>
                  {segment.pending ? (
                    <p className="realtime-pending">
                      <Loader2 aria-hidden="true" className="spin" />
                      正在识别这一段…
                    </p>
                  ) : segment.error ? (
                    <p className="realtime-error-text">{segment.error}</p>
                  ) : (
                    <p>{segment.text}</p>
                  )}
                </article>
              ))
            )}
          </div>
        </section>
      </div>
    );
  }

  function renderModels() {
    return (
      <div className="stack settings-page">
        <form className="panel settings-work-background" onSubmit={saveAgentSettings}>
          <PanelHeader icon={<MessageCircle aria-hidden="true" />} title="关于你" />
          <p className="muted">
            这是你主动填写的个性化资料，独立于系统自动生成的记忆。只写希望长期生效的称呼、身份与偏好。
          </p>
          <div className="form-grid">
            <Field label="昵称" htmlFor="agent-nickname">
              <input id="agent-nickname" value={agentSettingsForm.nickname} onChange={(event) => setAgentSettingsForm({ ...agentSettingsForm, nickname: event.target.value })} placeholder="希望智能体如何称呼你？" />
            </Field>
            <Field label="职业" htmlFor="agent-occupation">
              <input id="agent-occupation" value={agentSettingsForm.occupation} onChange={(event) => setAgentSettingsForm({ ...agentSettingsForm, occupation: event.target.value })} placeholder="例如：机器人产业研究员" />
            </Field>
          </div>
          <Field label="你的详情" htmlFor="agent-details">
            <textarea
              id="agent-details"
              rows={8}
              value={agentSettingsForm.details}
              onChange={(event) => setAgentSettingsForm({ ...agentSettingsForm, details: event.target.value })}
              placeholder="需要长期记住的兴趣、价值观、偏好或固定表达要求。临时项目进展不要写在这里。"
            />
          </Field>
          <Field label="文档排版偏好" htmlFor="company-document-format">
            <textarea
              id="company-document-format"
              name="company-document-format"
              rows={7}
              value={agentSettingsForm.company_document_format}
              onChange={(event) =>
                setAgentSettingsForm({
                  ...agentSettingsForm,
                  company_document_format: event.target.value
                })
              }
              placeholder={"页面设置：上3.5厘米、下3.1厘米、左2.65厘米、右2.65厘米\n行间距：固定值29.6磅\n标题：2号字，方正小标宋简体"}
            />
          </Field>
          <p className="muted">
            只写纯文字排版规则。它是文档偏好，不会被系统当作你的自动记忆。
          </p>
          <div className="form-actions">
            <button type="submit" className="primary-button" disabled={busy}>
              <Check aria-hidden="true" />
              保存设置
            </button>
            {agentSettings?.work_background || agentSettings?.company_document_format ? (
              <span className="muted">已配置，会在新一轮对话中生效。</span>
            ) : null}
          </div>
        </form>

        <section className="panel memory-manager" aria-labelledby="cross-chat-memory-title">
          <div className="memory-manager-heading">
            <div className="panel-header">
              <span className="panel-icon"><Brain aria-hidden="true" /></span>
              <div>
                <h3 id="cross-chat-memory-title">记忆</h3>
                <p>系统从聊天中提炼长期信息，并在回答前按相关性取回；聊天原文仍可用于核验。</p>
              </div>
            </div>
            <div className="memory-manager-status">
              <label className="skill-switch">
                <input type="checkbox" checked={agentSettingsForm.memory_enabled} onChange={(event) => setAgentSettingsForm({ ...agentSettingsForm, memory_enabled: event.target.checked })} />
                <span aria-hidden="true" />
                <b>{agentSettingsForm.memory_enabled ? "启用记忆" : "关闭记忆"}</b>
              </label>
              <button type="button" className="text-button" onClick={() => void saveAgentSettings({ preventDefault() {} } as FormEvent<HTMLFormElement>)} disabled={busy}>保存开关</button>
              <button type="button" className="text-button" onClick={() => void refreshCrossChatMemories()} disabled={busy}>
                <RefreshCw aria-hidden="true" />重新同步
              </button>
            </div>
          </div>

          <div className="memory-manager-toolbar">
            <label className="memory-search-field">
              <Search aria-hidden="true" />
              <span className="sr-only">搜索跨聊天记忆</span>
              <input
                type="search"
                value={memoryQuery}
                placeholder="搜索记忆或来源聊天"
                onChange={(event) => setMemoryQuery(event.target.value)}
              />
            </label>
            <div className="memory-scope-switch" role="group" aria-label="记忆范围">
              {([
                ["all", "全部"],
                ["account", "普通聊天"],
                ["projects", "项目"]
              ] as const).map(([value, label]) => (
                <button
                  key={value}
                  type="button"
                  className={memoryScope === value ? "is-active" : ""}
                  aria-pressed={memoryScope === value}
                  onClick={() => setMemoryScope(value)}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          <div className="memory-structure-note">
            <span><strong>关于你</strong> 由你主动填写</span>
            <ChevronRight aria-hidden="true" />
            <span><strong>自动记忆</strong> 从聊天提炼</span>
            <ChevronRight aria-hidden="true" />
            <span><strong>原文回想</strong> 用于核验细节</span>
          </div>

          {agentSettingsForm.memory_enabled ? null : <p className="muted">关闭后不会生成、更新或在回答中参考自动记忆；现有记忆不会被删除。</p>}
          {crossChatMemories.length > 0 ? null : <p className="muted">记忆会在聊天达到一定信息量后后台整理；不会把每句闲聊都保存下来。</p>}
          {memoryProfile ? (
            <article className="memory-profile" aria-label="自动记忆摘要">
              <header><strong>记忆摘要</strong><time>{formatProjectDate(memoryProfile.updated_at)}</time></header>
              <p>{memoryProfile.content}</p>
            </article>
          ) : null}

          {filteredCrossChatMemories.length > 0 ? (
            <div className="memory-list" role="list">
              {filteredCrossChatMemories.map((memory) => {
                const projectName = projects.find((project) => project.id === memory.project_id)?.name;
                const editing = editingMemoryId === memory.id;
                const confirmingDelete = deleteConfirmMemoryId === memory.id;
                return (
                  <article className="memory-row" role="listitem" key={memory.id}>
                    <header>
                      <div className="memory-source-title">
                        <strong>{memory.conversation_title || "未命名聊天"}</strong>
                        {memory.state === "corrected" ? <span className="memory-corrected-badge">已纠正</span> : null}
                        {projectName ? <span className="memory-project-badge">{projectName}</span> : null}
                      </div>
                      <time>{formatProjectDate(memory.updated_at)}</time>
                    </header>
                    {editing ? (
                      <div className="memory-editor">
                        <textarea
                          autoFocus
                          rows={6}
                          value={memoryDraft}
                          aria-label={`纠正来自${memory.conversation_title}的记忆`}
                          onChange={(event) => setMemoryDraft(event.target.value)}
                        />
                        <div>
                          <button type="button" className="secondary-button" onClick={() => { setEditingMemoryId(null); setMemoryDraft(""); }}>取消</button>
                          <button type="button" className="primary-button" disabled={busy || !memoryDraft.trim()} onClick={() => void saveMemoryCorrection(memory.id)}>
                            <Check aria-hidden="true" />保存纠正
                          </button>
                        </div>
                      </div>
                    ) : (
                      <p>{memory.content}</p>
                    )}
                    {!editing ? (
                      <footer>
                        <button type="button" className="text-button" onClick={() => openMemorySource(memory)}>
                          <Eye aria-hidden="true" />查看来源聊天
                        </button>
                        <span className="memory-row-spacer" />
                        <button type="button" className="text-button" onClick={() => startEditMemory(memory)}>
                          <Pencil aria-hidden="true" />纠正
                        </button>
                        <button
                          type="button"
                          className={`text-button memory-delete-button ${confirmingDelete ? "is-confirming" : ""}`}
                          onClick={() => void deleteCrossChatMemory(memory.id)}
                        >
                          <Trash2 aria-hidden="true" />{confirmingDelete ? "确认删除" : "删除"}
                        </button>
                      </footer>
                    ) : null}
                  </article>
                );
              })}
            </div>
          ) : (
            <div className="memory-empty-state">
              <Brain aria-hidden="true" />
              <strong>{crossChatMemories.length > 0 ? "没有匹配的记忆" : "还没有自动记忆"}</strong>
              <p>{crossChatMemories.length > 0 ? "换一个关键词或记忆范围试试。" : "系统会在后续对话中提炼有长期价值的信息；聊天原文仍可随时回想和核验。"}</p>
            </div>
          )}
        </section>

        <form className="panel settings-meeting-minutes" onSubmit={saveMeetingMinutesSettings}>
          <PanelHeader icon={<FileText aria-hidden="true" />} title="会议纪要设置" />
          <p className="muted">
            这里保存本账户的会议纪要偏好，会在每次生成时附加到会议纪要流程；它不修改技能的通用说明。
          </p>
          <Field label="默认输出目录" htmlFor="meeting-minutes-output-dir">
            <input
              id="meeting-minutes-output-dir"
              name="meeting-minutes-output-dir"
              type="text"
              spellCheck={false}
              value={meetingMinutesSettingsForm.default_output_dir}
              onChange={(event) =>
                setMeetingMinutesSettingsForm({
                  ...meetingMinutesSettingsForm,
                  default_output_dir: event.target.value
                })
              }
              placeholder="meet_files"
            />
          </Field>
          <Field label="补充写作要求" htmlFor="meeting-minutes-custom-instructions">
            <textarea
              id="meeting-minutes-custom-instructions"
              name="meeting-minutes-custom-instructions"
              rows={6}
              value={meetingMinutesSettingsForm.custom_instructions}
              onChange={(event) =>
                setMeetingMinutesSettingsForm({
                  ...meetingMinutesSettingsForm,
                  custom_instructions: event.target.value
                })
              }
              placeholder="例如：内部留档版保留待核实项；提交版使用本单位固定标题与分节方式。不要填写某次会议的临时事实。"
            />
          </Field>
          <div className="form-actions">
            <button type="submit" className="primary-button" disabled={busy}>
              <Check aria-hidden="true" />保存会议纪要设置
            </button>
            {meetingMinutesSettings?.custom_instructions ? <span className="muted">已配置，会在下次生成时生效。</span> : null}
          </div>
        </form>

        <form className="panel settings-asr" onSubmit={saveAsrSettings}>
          <PanelHeader icon={<Mic aria-hidden="true" />} title="语音识别设置" />
          <p className="muted">
            这里只配置本地 ASR 模型和识别热词。稳定的单位名称和专名请在上方“工作背景”维护。
          </p>
          <div className="form-grid">
            <Field label="本地ASR模型" htmlFor="asr-profile">
              <select
                id="asr-profile"
                name="asr-profile"
                value={asrSettingsForm.profile}
                onChange={(event) => {
                  const profile = event.target.value;
                  const option = asrSettings?.available_profiles.find((item) => item.name === profile);
                  setAsrSettingsForm({
                    ...asrSettingsForm,
                    profile,
                    model_id: option?.default_model_id || ""
                  });
                }}
              >
                {(asrSettings?.available_profiles ?? [
                  {
                    name: "qwen3-asr-mlx-8bit",
                    label: "Qwen3-ASR MLX 8bit（Mac推荐）",
                    default_model_id: "meeting_audio_minutes/model_cache/mlx-community/Qwen3-ASR-1.7B-8bit"
                  }
                ]).map((profile) => (
                  <option key={profile.name} value={profile.name}>
                    {profile.label}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="模型ID" htmlFor="asr-model-id">
              <input
                id="asr-model-id"
                name="asr-model-id"
                type="text"
                autoComplete="off"
                spellCheck={false}
                value={asrSettingsForm.model_id}
                onChange={(event) => setAsrSettingsForm({ ...asrSettingsForm, model_id: event.target.value })}
              />
            </Field>
          </div>
          <Field label="语音识别热词" htmlFor="asr-hotwords">
            <textarea
              id="asr-hotwords"
              name="asr-hotwords"
              rows={4}
              value={asrSettingsForm.hotwords}
              onChange={(event) => setAsrSettingsForm({ ...asrSettingsForm, hotwords: event.target.value })}
              placeholder="公司名、人名、客户名、项目名、技术词，例如：示例客户 智能座舱 SMT料盘 上料"
            />
          </Field>
          <button type="submit" className="primary-button" disabled={busy}>
            <Check aria-hidden="true" />
            保存语音识别设置
          </button>
        </form>

        <section className="panel model-manager" aria-labelledby="model-manager-title">
          <div className="model-manager-heading">
            <div className="panel-header">
              <span className="panel-icon"><Cpu aria-hidden="true" /></span>
              <div>
                <h3 id="model-manager-title">模型接入</h3>
                <p>集中切换、测试、复制和修改 OpenAI-compatible 模型配置。</p>
                {models?.env_override ? (
                  <p className="model-env-override-note">
                    当前模型由环境变量锁定为 <code translate="no">{models.env_override}</code>，网页内可编辑和测试，但不能切换。
                  </p>
                ) : null}
              </div>
            </div>
            <button type="button" className="primary-button" onClick={openAddModel} disabled={busy}>
              <Plus aria-hidden="true" />
              新增配置
            </button>
          </div>

          <div className="model-profile-list" role="list">
            {profiles.map((profile) => {
              const confirmingDelete = deleteConfirmModelName === profile.name;
              return (
                <article
                  className={`model-profile-row ${profile.default ? "is-current" : ""}`}
                  role="listitem"
                  key={profile.name}
                >
                  <span className="model-profile-mark" aria-hidden="true">
                    {providerInitial(profile.provider)}
                  </span>
                  <span className="model-profile-main">
                    <span className="model-profile-title">
                      <strong translate="no">{profile.name}</strong>
                      {profile.default ? <Badge tone="success">当前</Badge> : null}
                      <Badge tone={profile.api_key_configured ? "neutral" : "warning"}>
                        {profile.api_key_configured ? "密钥已配置" : "缺少密钥"}
                      </Badge>
                    </span>
                    <span className="model-profile-route">
                      <code translate="no">{profile.model}</code>
                      <small translate="no">{profile.base_url}</small>
                    </span>
                  </span>
                  <span className="model-profile-params">
                    <small>温度 {profile.temperature}</small>
                    <small>{numberFormatter.format(profile.max_tokens)} tokens</small>
                    <small>{profile.timeout_seconds}s</small>
                  </span>
                  <span className="model-profile-actions">
                    <button
                      type="button"
                      className="secondary-button"
                      disabled={busy || profile.default || Boolean(models?.env_override)}
                      onClick={() => void switchModel(profile.name)}
                      title={models?.env_override ? "当前模型由 WORK_AGENT_MODEL_PROFILE 环境变量锁定" : undefined}
                    >
                      {profile.default ? <Check aria-hidden="true" /> : null}
                      {profile.default ? "使用中" : models?.env_override ? "环境锁定" : "设为当前"}
                    </button>
                    <button
                      type="button"
                      className="text-button"
                      disabled={busy || !profile.api_key_configured}
                      onClick={() => void testModelConfiguration(profile)}
                    >
                      <RefreshCw aria-hidden="true" />
                      测试
                    </button>
                    <button type="button" className="text-button" disabled={busy} onClick={() => openEditModel(profile)}>
                      <Pencil aria-hidden="true" />
                      编辑
                    </button>
                    <button type="button" className="text-button" disabled={busy} onClick={() => copyModelProfile(profile)}>
                      <Copy aria-hidden="true" />
                      复制
                    </button>
                    <button
                      type="button"
                      className={`text-button model-profile-delete ${confirmingDelete ? "is-confirming" : ""}`}
                      disabled={busy || profile.default || profiles.length <= 1}
                      title={profile.default ? "请先切换到其他模型再删除" : "删除模型配置"}
                      onClick={() => void deleteModelConfiguration(profile)}
                    >
                      <Trash2 aria-hidden="true" />
                      {confirmingDelete ? "确认删除" : "删除"}
                    </button>
                  </span>
                </article>
              );
            })}
          </div>

          {modelEditorMode ? (
            <form className="model-editor" onSubmit={saveModelConfiguration}>
              <div className="model-editor-heading">
                <div>
                  <h4>{modelEditorMode === "edit" ? `编辑 ${editingModelName}` : "新增模型配置"}</h4>
                  <p>
                    {modelEditorMode === "edit"
                      ? "API 密钥留空表示保持原密钥不变。"
                      : modelForm.source_name
                        ? `复制自 ${modelForm.source_name}，默认沿用原密钥。`
                        : "选择预设会自动填写接口地址和推荐参数。"}
                  </p>
                </div>
                <button type="button" className="icon-action-button" aria-label="关闭模型编辑器" onClick={closeModelEditor}>
                  <X aria-hidden="true" />
                </button>
              </div>

              <div className="model-preset-strip" role="group" aria-label="供应商预设">
                {(Object.entries(MODEL_PROVIDER_PRESETS) as Array<[ModelProviderPresetId, typeof MODEL_PROVIDER_PRESETS[ModelProviderPresetId]]>).map(([id, preset]) => (
                  <button
                    type="button"
                    key={id}
                    className={modelForm.preset === id ? "is-active" : ""}
                    aria-pressed={modelForm.preset === id}
                    onClick={() => {
                      setModelForm((current) => ({
                        ...current,
                        preset: id,
                        provider: preset.provider,
                        base_url: preset.base_url,
                        model: preset.model,
                        temperature: preset.temperature,
                        max_tokens: preset.max_tokens,
                        timeout_seconds: preset.timeout_seconds
                      }));
                      setDiscoveredModelIds([]);
                      setModelConnectionResult(null);
                    }}
                  >
                    <strong>{preset.label}</strong>
                    <small>{preset.description}</small>
                  </button>
                ))}
              </div>

              <div className="form-grid">
                <Field label="配置名称" htmlFor="profile-name">
                  <input
                    id="profile-name"
                    name="profile-name"
                    type="text"
                    autoComplete="off"
                    readOnly={modelEditorMode === "edit"}
                    placeholder="例如：deepseek-v4-flash"
                    value={modelEditorMode === "edit" ? editingModelName : modelForm.name}
                    onChange={(event) => setModelForm({ ...modelForm, name: event.target.value })}
                  />
                </Field>
                <Field
                  label={modelEditorMode === "edit" || modelForm.source_name ? "API 密钥（留空则沿用）" : "API 密钥"}
                  htmlFor="profile-api-key"
                >
                  <div className="secret-input">
                    <input
                      id="profile-api-key"
                      name="profile-api-key"
                      type={showModelApiKey ? "text" : "password"}
                      autoComplete="new-password"
                      spellCheck={false}
                      placeholder={modelEditorMode === "edit" || modelForm.source_name ? "••••••••（不更换）" : "输入供应商提供的 API Key"}
                      value={modelForm.api_key}
                      onChange={(event) => setModelForm({ ...modelForm, api_key: event.target.value })}
                    />
                    <button
                      type="button"
                      aria-label={showModelApiKey ? "隐藏 API 密钥" : "显示 API 密钥"}
                      title={showModelApiKey ? "隐藏 API 密钥" : "显示 API 密钥"}
                      onClick={() => setShowModelApiKey((visible) => !visible)}
                    >
                      {showModelApiKey ? <EyeOff aria-hidden="true" /> : <Eye aria-hidden="true" />}
                    </button>
                  </div>
                </Field>
              </div>

              <Field label="接口地址" htmlFor="profile-base-url">
                <input
                  id="profile-base-url"
                  name="profile-base-url"
                  type="url"
                  inputMode="url"
                  autoComplete="off"
                  spellCheck={false}
                  placeholder="例如：https://api.example.com/v1"
                  value={modelForm.base_url}
                  onChange={(event) => {
                    setModelForm({ ...modelForm, base_url: event.target.value });
                    setDiscoveredModelIds([]);
                    setModelConnectionResult(null);
                  }}
                />
              </Field>

              <Field label="模型名称" htmlFor="profile-model">
                <div className="model-id-control">
                  <input
                    id="profile-model"
                    name="profile-model"
                    type="text"
                    list={discoveredModelIds.length ? "discovered-model-list" : undefined}
                    autoComplete="off"
                    spellCheck={false}
                    placeholder="输入模型 ID，或从接口获取"
                    value={modelForm.model}
                    onChange={(event) => setModelForm({ ...modelForm, model: event.target.value })}
                  />
                  <button
                    type="button"
                    className="secondary-button"
                    disabled={busy || !modelForm.base_url.trim() || (!modelForm.api_key.trim() && modelEditorMode !== "edit" && !modelForm.source_name)}
                    onClick={() => void discoverModels()}
                  >
                    <Download aria-hidden="true" />
                    获取模型
                  </button>
                  <datalist id="discovered-model-list">
                    {discoveredModelIds.map((modelId) => <option value={modelId} key={modelId} />)}
                  </datalist>
                </div>
              </Field>

              <details className="model-advanced">
                <summary>高级参数</summary>
                <div className="form-grid three">
                  <Field label="温度" htmlFor="profile-temperature">
                    <input
                      id="profile-temperature"
                      type="number"
                      min={0}
                      max={2}
                      step={0.1}
                      value={modelForm.temperature}
                      onChange={(event) => setModelForm({ ...modelForm, temperature: Number(event.target.value) })}
                    />
                  </Field>
                  <Field label="最大输出 Token" htmlFor="profile-max-tokens">
                    <input
                      id="profile-max-tokens"
                      type="number"
                      min={1}
                      value={modelForm.max_tokens}
                      onChange={(event) => setModelForm({ ...modelForm, max_tokens: Number(event.target.value) })}
                    />
                  </Field>
                  <Field label="超时时间（秒）" htmlFor="profile-timeout">
                    <input
                      id="profile-timeout"
                      type="number"
                      min={10}
                      value={modelForm.timeout_seconds}
                      onChange={(event) => setModelForm({ ...modelForm, timeout_seconds: Number(event.target.value) })}
                    />
                  </Field>
                </div>
              </details>

              <label className="checkbox-row model-default-toggle">
                <input
                  name="set-default-profile"
                  type="checkbox"
                  checked={modelForm.set_default}
                  disabled={Boolean(models?.env_override)}
                  onChange={(event) => setModelForm({ ...modelForm, set_default: event.target.checked })}
                />
                <span>{modelEditorMode === "edit" ? "保存后设为当前模型" : "添加后设为当前模型"}</span>
              </label>
              {models?.env_override ? (
                <p className="model-env-override-note model-editor-override-note">
                  如需切换当前模型，请先移除服务环境中的 <code>WORK_AGENT_MODEL_PROFILE</code> 覆盖。
                </p>
              ) : null}

              {modelConnectionResult ? (
                <div className={`model-test-result is-${modelConnectionResult.tone}`} role="status">
                  {modelConnectionResult.tone === "success"
                    ? <CheckCircle2 aria-hidden="true" />
                    : <AlertCircle aria-hidden="true" />}
                  <span>{modelConnectionResult.text}</span>
                </div>
              ) : null}

              <div className="model-editor-actions">
                <button
                  type="button"
                  className="secondary-button"
                  disabled={
                    busy ||
                    !modelForm.base_url.trim() ||
                    !modelForm.model.trim() ||
                    (!modelForm.api_key.trim() && modelEditorMode !== "edit" && !modelForm.source_name)
                  }
                  onClick={() => void testModelConfiguration()}
                >
                  <RefreshCw aria-hidden="true" className={busy ? "spin" : ""} />
                  测试连接
                </button>
                <span className="model-editor-action-spacer" />
                <button type="button" className="secondary-button" disabled={busy} onClick={closeModelEditor}>
                  取消
                </button>
                <button
                  type="submit"
                  className="primary-button"
                  disabled={
                    busy ||
                    !(modelEditorMode === "edit" ? editingModelName : modelForm.name).trim() ||
                    !modelForm.base_url.trim() ||
                    !modelForm.model.trim() ||
                    (!modelForm.api_key.trim() && modelEditorMode !== "edit" && !modelForm.source_name)
                  }
                >
                  <Check aria-hidden="true" />
                  {modelEditorMode === "edit" ? "保存修改" : "添加配置"}
                </button>
              </div>
            </form>
          ) : null}
        </section>
      </div>
    );
  }

  function renderArtifacts() {
    return artifactTab === "meeting" ? renderMeetingArtifacts() : renderFileArtifacts();
  }

  function renderFileArtifacts() {
    if (selectedFile) {
      return renderFileReader(selectedFile);
    }

    return (
      <div className="library-page">
        <header className="library-header">
          <div>
            <h2>文件库</h2>
            <p>拖进来的录音、图片、PDF、Word，以及智能体生成的纪要和 Markdown 产出都会保存在这里。</p>
          </div>
          <div className="library-actions">
            <label className="library-search" htmlFor="library-search">
              <Search aria-hidden="true" />
              <input
                id="library-search"
                type="search"
                placeholder="搜索"
                value={fileQuery}
                onChange={(event) => setFileQuery(event.target.value)}
              />
            </label>
            <label className="library-upload-button">
              <Plus aria-hidden="true" />
              添加文件
              <input
                type="file"
                multiple
                onChange={(event) => {
                  if (event.currentTarget.files) {
                    void attachDroppedFiles(event.currentTarget.files);
                  }
                  event.currentTarget.value = "";
                }}
              />
            </label>
          </div>
        </header>

        <div className="library-filter-row" role="tablist" aria-label="文件分类">
          {fileFilterOptions.map((option) => (
            <button
              key={option.id}
              type="button"
              role="tab"
              aria-selected={fileFilter === option.id}
              className={fileFilter === option.id ? "is-active" : ""}
              onClick={() => setFileFilter(option.id)}
            >
              {option.label}
              <span>{libraryCounts[option.id]}</span>
            </button>
          ))}
        </div>

        <div className="library-layout">
          <section className="library-list-panel" aria-label="文件列表">
            <div className="library-table-head" aria-hidden="true">
              <span>名称</span>
              <span>修改时间</span>
              <span>大小</span>
            </div>
            <div className="library-file-list">
              {filteredFiles.length === 0 ? (
                <EmptyState text={files.length === 0 ? "还没有文件，拖入录音或材料后会显示在这里。" : "没有匹配的文件。"} />
              ) : (
                filteredFiles.map((file) => (
                  <button
                    key={file.path}
                    type="button"
                    className="library-file-row"
                    onClick={() => openFile(file.path)}
                  >
                    <span className="library-name-cell">
                      <span className={`library-file-icon file-kind-${getLibraryKind(file)}`}>
                        {iconForLibraryFile(file)}
                      </span>
                      <span>
                        <strong>{file.name}</strong>
                        <code translate="no">{file.path}</code>
                      </span>
                    </span>
                    <span className="library-date-cell">{formatFileDate(file.modified)}</span>
                    <span className="library-size-cell">{formatBytes(file.size)}</span>
                  </button>
                ))
              )}
            </div>
          </section>
        </div>
      </div>
    );
  }

  function renderFileReader(file: FilePayload) {
    const kind = getLibraryKind(file);
    const widePreview = ["pdf", "image", "video"].includes(file.preview_mode);
    return (
      <div className={`file-reader-page file-preview-${file.preview_mode}`}>
        <header className="file-reader-toolbar" aria-label="文件阅读导航">
          <div className="file-reader-breadcrumb">
            <button
              type="button"
              className="file-reader-close"
              aria-label="关闭文件"
              title="关闭文件"
              onClick={() => setSelectedFile(null)}
            >
              <X aria-hidden="true" />
            </button>
            <span>库</span>
            <ChevronRight aria-hidden="true" />
            <strong>{fileTitleForReader(file.name)}</strong>
          </div>
          <div className="file-reader-actions">
            <button
              type="button"
              className="icon-button"
              aria-label="复制文件路径"
              title="复制文件路径"
              onClick={() => copyPath(file.path)}
            >
              <Copy aria-hidden="true" />
            </button>
            <button
              type="button"
              className="icon-button"
              aria-label="用系统打开"
              title="用系统打开"
              onClick={() => openLocalFile(file.path)}
            >
              <ExternalLink aria-hidden="true" />
            </button>
            <button
              type="button"
              className="icon-button"
              aria-label="在访达中显示"
              title="在访达中显示"
              onClick={() => revealLocalFile(file.path)}
            >
              <FolderOpen aria-hidden="true" />
            </button>
          </div>
        </header>

        <section className={`file-reader-body ${widePreview ? "is-wide" : ""}`} aria-label={`${file.name} 文件内容`}>
          <div className="file-reader-title">
            <span className={`library-file-icon file-kind-${kind}`}>
              {iconForLibraryFile(file)}
            </span>
            <div>
              <h2>{fileTitleForReader(file.name)}</h2>
              <code translate="no">{file.path}</code>
              {file.rendered_path ? <small>已生成本地预览：{file.rendered_path}</small> : null}
            </div>
          </div>

          {renderFilePreview(file)}
          {file.truncated ? <p className="file-reader-hint">预览内容已截断，可让智能体读取完整文件。</p> : null}
        </section>

        <form
          className="file-reader-compose"
          onSubmit={(event) => {
            event.preventDefault();
            sendSelectedFileToAgent(file);
          }}
        >
          <label htmlFor="file-reader-prompt" className="sr-only">
            描述修改内容
          </label>
          <textarea
            id="file-reader-prompt"
            rows={1}
            value={fileActionText}
            placeholder={file.editable ? "描述修改内容" : "问问关于此文件的问题"}
            onChange={(event) => setFileActionText(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && (event.metaKey || event.ctrlKey) && !event.nativeEvent.isComposing) {
                event.preventDefault();
                event.currentTarget.form?.requestSubmit();
              }
            }}
          />
          <button type="submit" className="composer-send" aria-label="交给智能体">
            <SendHorizontal aria-hidden="true" />
          </button>
        </form>
      </div>
    );
  }

  function renderFilePreview(file: FilePayload) {
    if (file.preview_mode === "markdown") {
      return (
        <article className="file-reader-markdown">
          <MarkdownContent content={file.content} onOpenFile={openLinkedFile} />
        </article>
      );
    }
    if (file.preview_mode === "text") {
      return <pre className="file-reader-plain">{file.content}</pre>;
    }
    if (file.preview_mode === "pdf" && file.preview_url) {
      return (
        <div className="file-reader-document-frame">
          <iframe title={file.name} src={file.preview_url} />
        </div>
      );
    }
    if (file.preview_mode === "image" && file.preview_url) {
      return (
        <figure className="file-reader-image-frame">
          <img src={file.preview_url} alt={file.name} />
        </figure>
      );
    }
    if (file.preview_mode === "audio" && file.preview_url) {
      return (
        <div className="file-reader-media-frame">
          <audio controls src={file.preview_url} />
        </div>
      );
    }
    if (file.preview_mode === "video" && file.preview_url) {
      return (
        <div className="file-reader-video-frame">
          <video controls src={file.preview_url} />
        </div>
      );
    }
    return (
      <div className="file-reader-unpreviewable">
        <span className={`library-file-icon file-kind-${getLibraryKind(file)}`}>
          {iconForLibraryFile(file)}
        </span>
        <strong>这个文件暂不支持网页预览</strong>
        <p>可以用系统打开，或在访达中定位后查看。</p>
        <div className="file-reader-unpreviewable-actions">
          <button type="button" className="secondary-button" onClick={() => openLocalFile(file.path)}>
            <ExternalLink aria-hidden="true" />
            用系统打开
          </button>
          <button type="button" className="secondary-button" onClick={() => revealLocalFile(file.path)}>
            <FolderOpen aria-hidden="true" />
            在访达中显示
          </button>
        </div>
      </div>
    );
  }

  function sendSelectedFileToAgent(file: FilePayload) {
    const prompt = fileActionText.trim();
    const reference = prompt
      ? `请读取并处理这个文件：${file.path}\n\n处理要求：${prompt}`
      : `请读取并概括这个文件：${file.path}`;
    setChatInput(reference);
    setSelectedFile(null);
    setFileActionText("");
    setViewWithUrl("agent");
  }
}

function AuthScreen({
  mode,
  form,
  busy,
  error,
  onModeChange,
  onFormChange,
  onSubmit
}: {
  mode: "login" | "register";
  form: { username: string; password: string; confirm: string };
  busy: boolean;
  error: string;
  onModeChange: (mode: "login" | "register") => void;
  onFormChange: Dispatch<SetStateAction<{ username: string; password: string; confirm: string }>>;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
}) {
  return (
    <main className="auth-shell">
      <section className="auth-intro" aria-labelledby="auth-title">
        <span className="auth-kicker">WORK AGENT / LOCAL</span>
        <h1 id="auth-title">你的工作记录，<br />只留在你的账户里。</h1>
        <p>对话、个人工作背景和上传文件按账户分开保存；模型与技能由管理员统一维护。</p>
        <div className="auth-signal" aria-label="服务状态">
          <span aria-hidden="true" /> 局域网服务已连接
        </div>
      </section>
      <section className="auth-panel" aria-label={mode === "login" ? "登录" : "注册账户"}>
        <div className="auth-mode-tabs" role="tablist" aria-label="账户操作">
          <button type="button" role="tab" aria-selected={mode === "login"} className={mode === "login" ? "is-active" : ""} onClick={() => onModeChange("login")}>登录</button>
          <button type="button" role="tab" aria-selected={mode === "register"} className={mode === "register" ? "is-active" : ""} onClick={() => onModeChange("register")}>注册</button>
        </div>
        <form className="auth-form" onSubmit={onSubmit}>
          <div>
            <h2>{mode === "login" ? "欢迎回来" : "建立独立工作区"}</h2>
            <p>{mode === "login" ? "输入账户信息继续。" : "注册完成后会自动登录。"}</p>
          </div>
          <label>
            <span>用户名</span>
            <input autoFocus autoComplete="username" value={form.username} onChange={(event) => onFormChange((value) => ({ ...value, username: event.target.value }))} placeholder="3–32 位字母或数字" required minLength={3} maxLength={32} />
          </label>
          <label>
            <span>密码</span>
            <input type="password" autoComplete={mode === "login" ? "current-password" : "new-password"} value={form.password} onChange={(event) => onFormChange((value) => ({ ...value, password: event.target.value }))} placeholder="至少 8 位" required minLength={8} />
          </label>
          {mode === "register" ? (
            <label>
              <span>确认密码</span>
              <input type="password" autoComplete="new-password" value={form.confirm} onChange={(event) => onFormChange((value) => ({ ...value, confirm: event.target.value }))} placeholder="再输入一次" required minLength={8} />
            </label>
          ) : null}
          {error ? <p className="auth-error" role="alert"><AlertCircle aria-hidden="true" />{error}</p> : null}
          <button type="submit" className="auth-submit" disabled={busy}>
            {busy ? <Loader2 className="spin" aria-hidden="true" /> : null}
            {busy ? "正在连接…" : mode === "login" ? "进入工作台" : "注册并进入"}
          </button>
        </form>
        <p className="auth-footnote">密码只保存为不可逆摘要，不会写入对话或浏览器存储。</p>
      </section>
    </main>
  );
}

function Field({
  label,
  htmlFor,
  children
}: {
  label: string;
  htmlFor: string;
  children: ReactNode;
}) {
  return (
    <div className="field">
      <label htmlFor={htmlFor}>{label}</label>
      {children}
    </div>
  );
}

function PanelHeader({ icon, title }: { icon: ReactNode; title: string }) {
  return (
    <div className="panel-header">
      <span className="panel-icon">{icon}</span>
      <h3>{title}</h3>
    </div>
  );
}

function OutputPanel({
  title,
  empty,
  content
}: {
  title: string;
  empty: string;
  content: ReactNode;
}) {
  return (
    <section className="panel output-panel">
      <PanelHeader icon={<CheckCircle2 aria-hidden="true" />} title={title} />
      {content ?? <EmptyState text={empty} />}
    </section>
  );
}

function EmptyState({ text }: { text: string }) {
  return (
    <div className="empty-state">
      <AlertCircle aria-hidden="true" />
      <span>{text}</span>
    </div>
  );
}

type MarkdownBlock =
  | { type: "heading"; level: number; text: string }
  | { type: "paragraph"; lines: string[] }
  | { type: "ul"; items: string[] }
  | { type: "ol"; items: string[] }
  | { type: "code"; language: string; content: string }
  | { type: "table"; headers: string[]; rows: string[][] };

type MarkdownContentProps = {
  content: string;
  onOpenFile?: (path: string) => void | Promise<void>;
};

function MarkdownContent({ content, onOpenFile }: MarkdownContentProps) {
  const blocks = parseMarkdownBlocks(content);
  return (
    <div className="markdown-content">
      {blocks.map((block, index) => renderMarkdownBlock(block, index, onOpenFile))}
    </div>
  );
}

function renderMarkdownBlock(
  block: MarkdownBlock,
  index: number,
  onOpenFile?: MarkdownContentProps["onOpenFile"]
) {
  if (block.type === "heading") {
    const HeadingTag = `h${Math.min(Math.max(block.level, 2), 4)}` as "h2" | "h3" | "h4";
    return <HeadingTag key={`heading-${index}`}>{renderInlineMarkdown(block.text, `h-${index}`, onOpenFile)}</HeadingTag>;
  }
  if (block.type === "ul") {
    return (
      <ul key={`ul-${index}`}>
        {block.items.map((item, itemIndex) => (
          <li key={`ul-${index}-${itemIndex}`}>{renderInlineMarkdown(item, `ul-${index}-${itemIndex}`, onOpenFile)}</li>
        ))}
      </ul>
    );
  }
  if (block.type === "ol") {
    return (
      <ol key={`ol-${index}`}>
        {block.items.map((item, itemIndex) => (
          <li key={`ol-${index}-${itemIndex}`}>{renderInlineMarkdown(item, `ol-${index}-${itemIndex}`, onOpenFile)}</li>
        ))}
      </ol>
    );
  }
  if (block.type === "code") {
    return (
      <pre key={`code-${index}`}>
        <code>{block.content}</code>
      </pre>
    );
  }
  if (block.type === "table") {
    return (
      <div key={`table-${index}`} className="markdown-table-wrap">
        <table>
          <thead>
            <tr>
              {block.headers.map((header, cellIndex) => (
                <th key={`table-${index}-head-${cellIndex}`} scope="col">
                  {renderInlineMarkdown(header, `table-${index}-head-${cellIndex}`, onOpenFile)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {block.rows.map((row, rowIndex) => (
              <tr key={`table-${index}-row-${rowIndex}`}>
                {row.map((cell, cellIndex) => (
                  <td key={`table-${index}-cell-${rowIndex}-${cellIndex}`}>
                    {renderInlineMarkdown(cell, `table-${index}-cell-${rowIndex}-${cellIndex}`, onOpenFile)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }
  return (
    <p key={`paragraph-${index}`}>
      {block.lines.map((line, lineIndex) => (
        <span key={`paragraph-${index}-${lineIndex}`}>
          {lineIndex > 0 ? <br /> : null}
          {renderInlineMarkdown(line, `p-${index}-${lineIndex}`, onOpenFile)}
        </span>
      ))}
    </p>
  );
}

function parseMarkdownBlocks(content: string): MarkdownBlock[] {
  const lines = content.replace(/\r\n/g, "\n").split("\n");
  const blocks: MarkdownBlock[] = [];
  let paragraph: string[] = [];
  let listType: "ul" | "ol" | null = null;
  let listItems: string[] = [];
  let inCode = false;
  let codeLanguage = "";
  let codeLines: string[] = [];

  const flushParagraph = () => {
    if (paragraph.length > 0) {
      blocks.push({ type: "paragraph", lines: paragraph });
      paragraph = [];
    }
  };
  const flushList = () => {
    if (listType && listItems.length > 0) {
      blocks.push({ type: listType, items: listItems });
    }
    listType = null;
    listItems = [];
  };

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index] ?? "";
    const trimmed = line.trim();

    if (trimmed.startsWith("```")) {
      if (inCode) {
        blocks.push({ type: "code", language: codeLanguage, content: codeLines.join("\n") });
        inCode = false;
        codeLanguage = "";
        codeLines = [];
      } else {
        flushParagraph();
        flushList();
        inCode = true;
        codeLanguage = trimmed.slice(3).trim();
      }
      continue;
    }

    if (inCode) {
      codeLines.push(line);
      continue;
    }

    if (!trimmed) {
      flushParagraph();
      flushList();
      continue;
    }

    const headingMatch = trimmed.match(/^(#{1,4})\s+(.+)$/);
    if (headingMatch) {
      flushParagraph();
      flushList();
      blocks.push({
        type: "heading",
        level: headingMatch[1].length,
        text: headingMatch[2].trim()
      });
      continue;
    }

    if (isMarkdownTableStart(lines, index)) {
      flushParagraph();
      flushList();
      const parsedTable = parseMarkdownTable(lines, index);
      blocks.push(parsedTable.block);
      index = parsedTable.nextIndex - 1;
      continue;
    }

    const unorderedMatch = line.match(/^\s*[-*]\s+(.+)$/);
    const orderedMatch = line.match(/^\s*\d+[.)]\s+(.+)$/);
    if (unorderedMatch || orderedMatch) {
      flushParagraph();
      const nextType = unorderedMatch ? "ul" : "ol";
      if (listType && listType !== nextType) flushList();
      listType = nextType;
      listItems.push((unorderedMatch?.[1] ?? orderedMatch?.[1] ?? "").trim());
      continue;
    }

    flushList();
    paragraph.push(trimmed);
  }

  if (inCode) {
    blocks.push({ type: "code", language: codeLanguage, content: codeLines.join("\n") });
  }
  flushParagraph();
  flushList();
  return blocks.length > 0 ? blocks : [{ type: "paragraph", lines: [content] }];
}

function renderInlineMarkdown(
  text: string,
  keyPrefix: string,
  onOpenFile?: MarkdownContentProps["onOpenFile"]
): ReactNode[] {
  const nodes: ReactNode[] = [];
  const pattern = /(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))/g;
  const pushText = (value: string) => {
    if (!value) return;
    nodes.push(...renderTextWithLocalFileLinks(value, `${keyPrefix}-text-${nodes.length}`, onOpenFile));
  };
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIndex) {
      pushText(text.slice(lastIndex, match.index));
    }
    const token = match[0];
    const key = `${keyPrefix}-${nodes.length}`;
    if (token.startsWith("**") && token.endsWith("**")) {
      nodes.push(<strong key={key}>{renderInlineMarkdown(token.slice(2, -2), `${key}-strong`, onOpenFile)}</strong>);
    } else if (token.startsWith("`") && token.endsWith("`")) {
      const codeContent = token.slice(1, -1);
      const localPath = normalizeLocalFileReference(codeContent);
      nodes.push(
        localPath && onOpenFile ? (
          <button
            key={key}
            type="button"
            className="markdown-file-link markdown-file-code"
            onClick={() => void onOpenFile(localPath)}
          >
            <code>{codeContent}</code>
          </button>
        ) : (
          <code key={key}>{codeContent}</code>
        )
      );
    } else {
      const linkMatch = token.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
      if (linkMatch) {
        const localPath = normalizeLocalFileReference(linkMatch[2]);
        if (localPath && onOpenFile) {
          nodes.push(
            <button
              key={key}
              type="button"
              className="markdown-file-link"
              onClick={() => void onOpenFile(localPath)}
            >
              {renderInlineMarkdown(linkMatch[1], `${key}-link`, onOpenFile)}
            </button>
          );
        } else {
          const href = safeMarkdownHref(linkMatch[2]);
          nodes.push(
            <a key={key} href={href} target={href.startsWith("#") ? undefined : "_blank"} rel="noreferrer">
              {renderInlineMarkdown(linkMatch[1], `${key}-link`, onOpenFile)}
            </a>
          );
        }
      } else {
        nodes.push(token);
      }
    }
    lastIndex = match.index + token.length;
  }
  if (lastIndex < text.length) {
    pushText(text.slice(lastIndex));
  }
  return nodes;
}

function renderTextWithLocalFileLinks(
  text: string,
  keyPrefix: string,
  onOpenFile?: MarkdownContentProps["onOpenFile"]
): ReactNode[] {
  if (!onOpenFile) return [text];
  const nodes: ReactNode[] = [];
  const pathPattern =
    /((?:(?:file:\/\/)?\/[^\s`'"<>|]*\/)?(?:\.\/)?(?:meet_files|meeting_audio_minutes|work_agent_skills|web_frontend|work_agent_core|config|schemas|tmp|产出材料|分析材料|学习笔记)\/[^\s`'"<>|]+?\.(?:md|txt|json|ya?ml|csv|log|srt|vtt|py|tsx?|jsx?|css|html|pdf|docx?|pptx?|xlsx?|png|jpe?g|webp|gif|heic|tiff?|m4a|mp3|wav|aac|flac|ogg|opus|wma|amr|aiff?|caf)(?::\d+(?::\d+)?|#L\d+(?:-L\d+)?)?)/giu;
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = pathPattern.exec(text)) !== null) {
    if (match.index > lastIndex) {
      nodes.push(text.slice(lastIndex, match.index));
    }
    const rawPath = match[1];
    const localPath = normalizeLocalFileReference(rawPath);
    if (localPath) {
      nodes.push(
        <button
          key={`${keyPrefix}-${nodes.length}`}
          type="button"
          className="markdown-file-link"
          onClick={() => void onOpenFile(localPath)}
        >
          {rawPath}
        </button>
      );
    } else {
      nodes.push(rawPath);
    }
    lastIndex = match.index + rawPath.length;
  }
  if (lastIndex < text.length) {
    nodes.push(text.slice(lastIndex));
  }
  return nodes;
}

function normalizeLocalFileReference(value: string) {
  const allowedPrefixes = [
    "meet_files/",
    "meeting_audio_minutes/",
    "work_agent_skills/",
    "web_frontend/",
    "work_agent_core/",
    "config/",
    "schemas/",
    "tmp/",
    "产出材料/",
    "分析材料/",
    "学习笔记/"
  ];
  let candidate = value
    .trim()
    .replace(/^<|>$/g, "")
    .replace(/^file:\/\//i, "")
    .replace(/[，。；;、,.!?！？:：)\]}]+$/u, "");
  if (!candidate || candidate.includes("\n")) return null;
  try {
    candidate = decodeURIComponent(candidate);
  } catch {
    // Keep the original path when it is not URL-encoded.
  }
  const rootMatch = candidate.match(/(?:^|\/)(meet_files|meeting_audio_minutes|work_agent_skills|web_frontend|work_agent_core|config|schemas|tmp|产出材料|分析材料|学习笔记)\//u);
  if (rootMatch?.index !== undefined) {
    candidate = candidate.slice(rootMatch.index + (rootMatch[0].startsWith("/") ? 1 : 0));
  }
  const sourceLocation = candidate.match(/^(.*\.[a-z0-9]+)(?::\d+(?::\d+)?|#L\d+(?:-L\d+)?)$/i);
  if (sourceLocation) {
    candidate = sourceLocation[1];
  }
  candidate = candidate.replace(/^\.\/+/, "");
  if (!allowedPrefixes.some((prefix) => candidate.startsWith(prefix))) return null;
  if (!/\.[a-z0-9]+$/i.test(candidate)) return null;
  return candidate;
}

function isMarkdownTableStart(lines: string[], index: number) {
  const current = lines[index]?.trim() ?? "";
  const next = lines[index + 1]?.trim() ?? "";
  return current.includes("|") && /^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$/.test(next);
}

function parseMarkdownTable(lines: string[], startIndex: number) {
  const headers = splitMarkdownTableRow(lines[startIndex]);
  const rows: string[][] = [];
  let index = startIndex + 2;
  while (index < lines.length && lines[index].trim().includes("|")) {
    rows.push(splitMarkdownTableRow(lines[index]));
    index += 1;
  }
  return { block: { type: "table" as const, headers, rows }, nextIndex: index };
}

function splitMarkdownTableRow(line: string) {
  return line
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((cell) => cell.trim());
}

function safeMarkdownHref(rawHref: string) {
  const href = rawHref.trim();
  if (/^(https?:|mailto:|#|\/)/i.test(href)) return href;
  return "#";
}

function HistorySearchGroups({
  items,
  onOpen
}: {
  items: ConversationHistoryItem[];
  onOpen: (item: ConversationHistoryItem) => void;
}) {
  const groups = groupConversations(items);
  return (
    <>
      {groups.map((group) => (
        <div key={group.label} className="search-history-group">
          <h3>{group.label}</h3>
          <div>
            {group.items.map((item) => (
              <button key={item.id} type="button" onClick={() => onOpen(item)}>
                <MessageCircle aria-hidden="true" />
                <span>{item.title}</span>
              </button>
            ))}
          </div>
        </div>
      ))}
    </>
  );
}

function PathRow({ label, path }: { label: string; path: string }) {
  return (
    <div className="path-row">
      <span>{label}</span>
      <code translate="no">{path}</code>
    </div>
  );
}

function VersionCard({
  title,
  description,
  path,
  onView,
  onCopy,
  onOpen,
  onReveal
}: {
  title: string;
  description: string;
  path?: string;
  onView: (path: string) => void | Promise<void>;
  onCopy: (path: string) => void | Promise<void>;
  onOpen?: (path: string) => void | Promise<void>;
  onReveal?: (path: string) => void | Promise<void>;
}) {
  return (
    <article
      className={`version-card ${path ? "is-openable" : ""}`}
      onDoubleClick={() => path && onView(path)}
      title={path ? "双击查看" : undefined}
    >
      <div>
        <h4>{title}</h4>
        <p>{description}</p>
      </div>
      {path ? (
        <code translate="no">{path}</code>
      ) : (
        <span className="missing-output">暂未找到这一版文件</span>
      )}
      <div className="version-actions">
        <button
          type="button"
          className="secondary-button"
          disabled={!path}
          onClick={() => path && onView(path)}
        >
          <FileText aria-hidden="true" />
          查看
        </button>
        {onOpen ? (
          <button
            type="button"
            className="text-button"
            disabled={!path}
            onClick={() => path && onOpen(path)}
          >
            <ExternalLink aria-hidden="true" />
            打开
          </button>
        ) : null}
        {onReveal ? (
          <button
            type="button"
            className="text-button"
            disabled={!path}
            onClick={() => path && onReveal(path)}
          >
            <FolderOpen aria-hidden="true" />
            访达
          </button>
        ) : null}
        <button
          type="button"
          className="text-button"
          disabled={!path}
          onClick={() => path && onCopy(path)}
        >
          <Copy aria-hidden="true" />
          复制路径
        </button>
      </div>
    </article>
  );
}

function Badge({ children, tone = "neutral" }: { children: ReactNode; tone?: "neutral" | "success" | "warning" }) {
  return <span className={`badge badge-${tone}`}>{children}</span>;
}

function StatusPill({ children, tone }: { children: ReactNode; tone: StatusTone }) {
  const title = typeof children === "string" ? children : undefined;
  return <span className={`status-pill status-${tone}`} title={title}>{children}</span>;
}

function StatusLine({ label, value }: { label: string; value: string }) {
  return (
    <div className="status-line">
      <span>{label}</span>
      <strong translate="no">{value}</strong>
    </div>
  );
}

function titleForView(view: View, artifactTab: ArtifactTab) {
  if (view === "projects") return "项目";
  if (view === "skills") return "技能";
  if (view === "transcribe") return "实时转写";
  if (view === "artifacts") return artifactTab === "files" ? "文件库" : "会议纪要";
  if (view === "models") return "模型与设置";
  if (view === "sync") return "临时同步区";
  if (view === "more") return "更多";
  return "智能体对话";
}

function formatProjectDate(epochSeconds: number) {
  if (!epochSeconds) return "刚刚";
  return new Intl.DateTimeFormat("zh-CN", {
    month: "numeric",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit"
  }).format(new Date(epochSeconds * 1000));
}

function explainMicrophonePermissionError(error: unknown) {
  if (error instanceof DOMException) {
    if (error.name === "NotAllowedError" || error.name === "SecurityError") {
      return "麦克风权限被浏览器拒绝，请在地址栏左侧的网站权限里允许麦克风。";
    }
    if (error.name === "NotFoundError" || error.name === "DevicesNotFoundError") {
      return "没有检测到可用麦克风，请检查系统输入设备。";
    }
    if (error.name === "NotReadableError" || error.name === "TrackStartError") {
      return "麦克风被其他应用占用，请关闭占用麦克风的软件后重试。";
    }
  }
  return "无法获取麦克风权限，请检查浏览器和系统麦克风权限。";
}

function explainRecorderStartError(error: unknown) {
  if (error instanceof DOMException) {
    if (error.name === "InvalidStateError") {
      return "录音已经在运行，请先停止后再开始。";
    }
    if (error.name === "NotSupportedError") {
      return "当前浏览器不支持可用的录音格式，请换 Edge 或 Chrome 再试。";
    }
    return `录音启动失败：${error.message || error.name}`;
  }
  return "录音启动失败，请刷新页面后重试。";
}

function preferredMediaRecorderOptions(): MediaRecorderOptions | undefined {
  const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"];
  for (const mimeType of candidates) {
    if (MediaRecorder.isTypeSupported(mimeType)) {
      return { mimeType };
    }
  }
  return undefined;
}

function createIdleVoiceLevels() {
  return Array.from({ length: voiceLevelBarCount }, (_, index) => {
    if (index < Math.floor(voiceLevelBarCount * 0.45)) return 0.018;
    return 0.06 + Math.abs(Math.sin(index * 0.72)) * 0.045;
  });
}

function realtimeTranscriptionVoiceThreshold(noiseFloor: number) {
  const floor = Math.min(0.14, Math.max(0.025, noiseFloor || realtimeTranscriptionNoiseFloorInitial));
  return Math.min(
    0.22,
    Math.max(0.075, floor + realtimeTranscriptionVoiceMargin, floor * realtimeTranscriptionVoiceRatio)
  );
}

function updateRealtimeNoiseFloor(previous: number, level: number) {
  const current = Math.min(0.14, Math.max(0.018, level || 0.018));
  const base = Math.min(0.14, Math.max(0.025, previous || realtimeTranscriptionNoiseFloorInitial));
  const smoothing = current < base ? 0.18 : 0.05;
  return Math.min(0.14, Math.max(0.025, base * (1 - smoothing) + current * smoothing));
}

function startVoiceLevelMeter(
  stream: MediaStream,
  setLevels: Dispatch<SetStateAction<number[]>>,
  animationRef: MutableRefObject<number | null>,
  audioContextRef: MutableRefObject<AudioContext | null>,
  latestLevelRef?: MutableRefObject<number>
) {
  stopVoiceLevelMeter(animationRef, audioContextRef);

  const AudioContextCtor =
    window.AudioContext ??
    (window as Window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!AudioContextCtor) {
    setLevels(createIdleVoiceLevels());
    return;
  }

  const audioContext = new AudioContextCtor();
  const analyser = audioContext.createAnalyser();
  analyser.fftSize = 1024;
  analyser.smoothingTimeConstant = 0.76;

  const source = audioContext.createMediaStreamSource(stream);
  source.connect(analyser);
  audioContextRef.current = audioContext;

  const samples = new Uint8Array(analyser.fftSize);
  const tick = () => {
    analyser.getByteTimeDomainData(samples);
    let sum = 0;
    for (const sample of samples) {
      const normalized = (sample - 128) / 128;
      sum += normalized * normalized;
    }
    const rms = Math.sqrt(sum / samples.length);
    const nextLevel = Math.min(1, Math.max(0.018, Math.pow(rms * 7.5, 0.72)));
    if (latestLevelRef) {
      latestLevelRef.current = nextLevel;
    }
    setLevels((levels) => [...levels.slice(1), nextLevel]);
    animationRef.current = window.requestAnimationFrame(tick);
  };

  void audioContext.resume();
  tick();
}

function stopVoiceLevelMeter(
  animationRef: MutableRefObject<number | null>,
  audioContextRef: MutableRefObject<AudioContext | null>
) {
  if (animationRef.current !== null) {
    window.cancelAnimationFrame(animationRef.current);
    animationRef.current = null;
  }
  const audioContext = audioContextRef.current;
  audioContextRef.current = null;
  if (audioContext && audioContext.state !== "closed") {
    void audioContext.close();
  }
}

function extensionForMimeType(mimeType: string) {
  const normalized = mimeType.split(";", 1)[0].toLowerCase();
  if (normalized === "audio/mp4") return ".m4a";
  if (normalized === "audio/ogg") return ".ogg";
  if (normalized === "audio/wav" || normalized === "audio/x-wav") return ".wav";
  if (normalized === "audio/mpeg") return ".mp3";
  return ".webm";
}

function stopVoiceStream(stream: MediaStream | null) {
  stream?.getTracks().forEach((track) => track.stop());
}

function explainError(error: unknown) {
  if (error instanceof Error) {
    const message = error.message;
    if (message.includes("Missing API key env var")) {
      return "模型密钥还没有配置，请检查项目根目录的 .env 文件并重启本地服务。";
    }
    if (message.includes("Failed to fetch")) {
      return "无法连接本地服务，请确认后端服务正在运行。";
    }
    return friendlyRuntimeError(message);
  }
  return "出现未知错误，请查看本地服务日志。";
}

function friendlyRuntimeError(message: string) {
  const normalized = String(message || "").toLowerCase();
  if (normalized.includes("http 402") || normalized.includes("insufficient balance")) {
    return "模型服务余额不足。充值或切换到可用模型后再继续。";
  }
  if (
    normalized.includes("timed out") ||
    normalized.includes("timeout") ||
    normalized.includes("超时") ||
    normalized.includes("总时限")
  ) {
    return "模型响应超时，本轮已完成的文件读取和工具结果仍然保留。可点击“继续本轮”接着处理。";
  }
  if (
    normalized.includes("503") ||
    normalized.includes("service unavailable") ||
    normalized.includes("tunnel connection failed")
  ) {
    return "当前模型网络连接失败（503），本轮进度已保留。请检查网络后点击“继续本轮”；如需更换模型，请在设置中手动选择。";
  }
  if (
    normalized.includes("没有返回正文或工具调用") ||
    normalized.includes("没有可显示的最终回复") ||
    normalized.includes("emptymodelresponse")
  ) {
    return "模型没有生成可显示的回复。系统只对当前模型尝试了恢复，不会自动更换模型；可点击“继续本轮”重试。";
  }
  if (normalized.includes("后端没有返回完成、失败或停止状态")) {
    return "连接意外中断，但已完成的工具结果可能仍已保留。可点击“继续本轮”接着处理。";
  }
  return message;
}

function isRetryableChatFailure(content: string) {
  return content.trimStart().startsWith("这次没有成功：");
}

function parseSkillQuery(value: string) {
  const match = value.match(/(?:^|\s)@([\u4e00-\u9fa5A-Za-z0-9_-]*)$/);
  return match ? match[1] : null;
}

function filterConversations(items: ConversationHistoryItem[], query: string) {
  const normalized = query.trim().toLowerCase();
  if (!normalized) return items;
  return items.filter((item) => `${item.title} ${item.group}`.toLowerCase().includes(normalized));
}

function groupConversations(items: ConversationHistoryItem[]) {
  const groups = new Map<string, ConversationHistoryItem[]>();
  for (const item of items) {
    const label = item.pinned ? "置顶" : item.group;
    const group = groups.get(label) ?? [];
    group.push(item);
    groups.set(label, group);
  }
  return Array.from(groups, ([label, groupItems]) => ({ label, items: groupItems }));
}

function orderConversationHistory(items: ConversationHistoryItem[]) {
  return [...items].sort((a, b) => {
    if (Boolean(a.pinned) === Boolean(b.pinned)) return 0;
    return a.pinned ? -1 : 1;
  });
}

function upsertConversation(items: ConversationHistoryItem[], next: ConversationHistoryItem) {
  return orderConversationHistory([next, ...items.filter((item) => item.id !== next.id)]);
}

function mergeConversationHistories(
  archivedItems: ConversationHistoryItem[],
  localItems: ConversationHistoryItem[]
) {
  const byId = new Map<string, ConversationHistoryItem>();
  for (const item of archivedItems) {
    byId.set(item.id, sanitizeConversationHistoryItem(item));
  }
  for (const item of localItems) {
    byId.set(item.id, sanitizeConversationHistoryItem(item));
  }
  return orderConversationHistory(Array.from(byId.values()));
}

function lastActivityRecordIndex(records: ActivityRecordMap) {
  const indexes = Object.keys(records)
    .map((key) => Number(key))
    .filter((index) => Number.isInteger(index))
    .sort((a, b) => b - a);
  return indexes[0] ?? null;
}

function buildActivityDisplayItems(events: AgentActivityEvent[]): ActivityDisplayItem[] {
  const displayItems: ActivityDisplayItem[] = [];
  const normalizedEvents = events.map(normalizeActivityEvent);
  const completedToolKeys = new Set(
    normalizedEvents
      .filter((event) => event.phase === "observation" && event.tool_name)
      .map((event) => `${event.step ?? "meta"}:${event.tool_name}`)
  );
  let group:
    | {
        kind: ActivityDisplayGroupKind;
        startIndex: number;
        events: AgentActivityEvent[];
      }
    | null = null;

  const flushGroup = () => {
    if (!group) return;
    displayItems.push(makeActivityDisplayGroup(group.kind, group.events, group.startIndex));
    group = null;
  };

  normalizedEvents.forEach((normalized, index) => {
    if (
      normalized.phase === "action" &&
      normalized.tool_name &&
      completedToolKeys.has(`${normalized.step ?? "meta"}:${normalized.tool_name}`)
    ) {
      return;
    }
    if (shouldHideActivityEvent(normalized)) return;
    const groupKind = groupKindForActivityEvent(normalized);
    if (groupKind) {
      if (!group || group.kind !== groupKind) {
        flushGroup();
        group = { kind: groupKind, startIndex: index, events: [] };
      }
      group.events.push(normalized);
      return;
    }
    flushGroup();
    displayItems.push({
      kind: "event",
      key: normalized.id ?? `${normalized.phase}-${normalized.step ?? "meta"}-${index}`,
      event: normalized
    });
  });
  flushGroup();
  return displayItems;
}

function makeActivityDisplayGroup(
  groupKind: ActivityDisplayGroupKind,
  events: AgentActivityEvent[],
  startIndex: number
): Extract<ActivityDisplayItem, { kind: "group" }> {
  const status = mergedActivityStatus(events);
  return {
    kind: "group",
    key: `activity-group-${groupKind}-${startIndex}`,
    groupKind,
    status,
    title: activityGroupTitle(groupKind, events),
    detail: activityGroupDetail(groupKind, events, status),
    icon: groupKind === "files" ? "±" : groupKind === "tools" ? "·" : "$",
    open: status === "running" || status === "approval_required" || status === "error",
    events
  };
}

function normalizeActivityEvent(event: AgentActivityEvent): AgentActivityEvent {
  const keepDebugText = event.phase === "error" || event.command_status === "error";
  if (isModelRequestCommand(event)) {
    return {
      ...event,
      title: event.step ? `第 ${event.step} 轮 · 模型思考` : "模型思考",
      detail: "正在决定下一步工具调用或最终回复",
      content: event.content ? sanitizeLegacyModelActivity(event.content, keepDebugText) : event.content,
      activity_type: undefined,
      command: undefined,
      command_status: undefined
    };
  }
  return {
    ...event,
    title: humanActivityTitle(event),
    detail: event.detail ? sanitizeActivityText(humanActivityDetail(event), keepDebugText) : event.detail,
    content: event.content ? sanitizeActivityText(event.content, keepDebugText) : event.content
  };
}

function sanitizeLegacyModelActivity(text: string, keepDebugText = false) {
  if (keepDebugText) return sanitizeActivityText(text, true);
  const cleaned = text
    .split("\n")
    .filter((line) => {
      const value = line.trim();
      return !(
        value.startsWith("已接收：") ||
        value.startsWith("说明：这里") ||
        value.startsWith("等待模型返回 tool calling") ||
        value.startsWith("当前收到的只有内部规划流") ||
        value.startsWith("因此暂时没有内容可预览")
      );
    })
    .join("\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
  return sanitizeActivityText(cleaned || "正在等待模型返回…");
}

function shouldHideActivityEvent(event: AgentActivityEvent) {
  if (event.title === "运行状态" && event.command_status !== "running") return true;
  if (event.phase === "error" || event.command_status === "error" || event.command_status === "approval_required") return false;
  if (event.title === "载入上下文" || event.title === "准备处理") return true;
  if (/^第 \d+ 轮模型规划$/.test(event.title) || event.title === "请求模型") return true;
  if (event.title === "准备工具") return true;
  return false;
}

function groupKindForActivityEvent(event: AgentActivityEvent): ActivityDisplayGroupKind | null {
  if (event.phase === "error" || event.command_status === "error" || event.command_status === "approval_required") return null;
  if (event.activity_type === "file_edit") return "files";
  if (event.activity_type === "command") {
    if (isModelRequestCommand(event)) return null;
    return "commands";
  }
  if (isToolActivityEvent(event)) return "tools";
  return null;
}

function mergedActivityStatus(events: AgentActivityEvent[]): NonNullable<AgentActivityEvent["command_status"]> {
  if (events.some((event) => event.command_status === "error" || event.phase === "error")) return "error";
  if (events.some((event) => event.command_status === "approval_required" || event.approval_required)) {
    return "approval_required";
  }
  if (events.some((event) => event.command_status === "running")) return "running";
  return "success";
}

function activityGroupTitle(groupKind: ActivityDisplayGroupKind, events: AgentActivityEvent[]) {
  if (groupKind === "files") {
    return `已编辑 ${numberFormatter.format(events.length)} 个文件`;
  }
  if (groupKind === "tools") {
    const toolNames = uniqueActivityToolNames(events);
    if (toolNames.length === 1) return `已调用 ${toolNames[0]}`;
    return `已调用 ${numberFormatter.format(events.length)} 个工具`;
  }

  const readCount = events.filter(isReadCommand).length;
  const searchCount = events.filter(isSearchCommand).length;
  const listCount = events.filter(isListCommand).length;
  const otherCount = events.length - readCount - searchCount - listCount;
  const parts: string[] = [];
  if (readCount) parts.push(`已读取 ${numberFormatter.format(readCount)} 个文件`);
  if (searchCount) parts.push(searchCount > 1 ? `已搜索代码 ${numberFormatter.format(searchCount)} 次` : "已搜索代码");
  if (listCount) parts.push(listCount > 1 ? `已列出文件 ${numberFormatter.format(listCount)} 次` : "已列出文件");
  if (otherCount) parts.push(`已运行 ${numberFormatter.format(otherCount)} 条命令`);
  return joinChineseParts(parts.length ? parts : [`已运行 ${numberFormatter.format(events.length)} 条命令`]);
}

function activityGroupDetail(
  groupKind: ActivityDisplayGroupKind,
  events: AgentActivityEvent[],
  status: NonNullable<AgentActivityEvent["command_status"]>
) {
  const statusText = commandStatusText(status);
  if (groupKind === "files") {
    const names = events.map((event) => fileNameFromPath(event.file_path ?? event.detail ?? "文件")).slice(0, 2);
    return `${statusText} · ${names.join("、")}${events.length > names.length ? " 等" : ""}`;
  }
  if (groupKind === "tools") {
    const names = uniqueActivityToolNames(events).slice(0, 2);
    return `${statusText} · ${names.length ? names.join("、") : "展开查看工具参数和结果"}`;
  }
  return `${statusText} · 展开查看命令、输出和复制按钮`;
}

function activityRowTitle(event: AgentActivityEvent) {
  if (event.activity_type === "file_edit") {
    return `已编辑 ${fileNameFromPath(event.file_path ?? event.detail ?? "文件")}`;
  }
  if (event.activity_type === "command") {
    if (event.title === "运行状态") return "运行状态";
    if (isReadCommand(event)) return "读取文件";
    if (isSearchCommand(event)) return "搜索代码";
    if (isListCommand(event)) return "列出文件";
    if (isModelRequestCommand(event)) return "模型请求";
    return "运行命令";
  }
  return event.title;
}

function activityRowDetail(event: AgentActivityEvent) {
  if (event.activity_type === "file_edit") {
    return event.file_path ?? event.detail ?? "";
  }
  if (event.activity_type === "command") {
    const command = event.command ?? "";
    const firstLine = command.split("\n")[0] ?? "";
    return firstLine.length > 160 ? `${firstLine.slice(0, 157)}...` : firstLine;
  }
  return event.detail ?? event.tool_name ?? "";
}

function activityRowCommandLine(event: AgentActivityEvent) {
  const command = event.command ?? event.tool_name ?? "";
  if (!command) return event.title;
  const label = commandActivityLabel(event);
  if (label === "LLM" && command.startsWith("LLM ")) return command;
  if (command.startsWith(`${label} `)) return command;
  return `${label} ${command}`;
}

function activityRowCopyText(event: AgentActivityEvent) {
  const output = (event.approval_preview || event.content || "").trimEnd();
  const commandLine = activityRowCommandLine(event);
  return output ? `${commandLine}\n\n${output}` : commandLine;
}

function compactActivityOutput(text: string, limit = 5000) {
  if (text.length <= limit) return text;
  const head = Math.floor(limit * 0.42);
  const tail = limit - head - 48;
  return `${text.slice(0, head).trimEnd()}\n\n... 已省略 ${numberFormatter.format(text.length - head - tail)} 字 ...\n\n${text
    .slice(-tail)
    .trimStart()}`;
}

function commandActivityLabel(event: AgentActivityEvent) {
  const command = event.command ?? "";
  if (isModelRequestCommand(event)) return "LLM";
  if (
    event.title === "写入文件" ||
    event.title === "生成DOCX" ||
    command.startsWith("write_text_file") ||
    command.startsWith("create_docx_from_markdown")
  ) {
    return "Write";
  }
  return "Shell";
}

function humanActivityTitle(event: AgentActivityEvent) {
  if (event.title === "理解请求") return "载入上下文";
  if (event.title === "分析任务") return "准备处理";
  if (/^第 \d+ 轮模型规划$/.test(event.title)) return "请求模型";
  if (/^第 \d+ 轮模型规划失败$/.test(event.title)) return "模型请求失败";
  if (event.title === "模型行动说明") return "执行计划";
  if (event.title.startsWith("准备调用 ")) return "准备工具";
  if (event.title.startsWith("执行工具：")) return `执行 ${event.tool_name ?? event.title.replace("执行工具：", "")}`;
  if (event.title.endsWith(" 返回结果")) return event.title.replace(" 返回结果", " 结果");
  return event.title;
}

function humanActivityDetail(event: AgentActivityEvent) {
  const detail = event.detail ?? "";
  if (event.title === "理解请求") return "已载入当前会话上下文。";
  if (event.title === "分析任务") return "正在判断下一步。";
  if (/^第 \d+ 轮模型规划$/.test(event.title)) return "正在请求模型决定下一步。";
  if (event.title.startsWith("准备调用 ")) return "模型选择了这个工具。";
  return detail;
}

function sanitizeActivityText(text: string, keepDebugText = false) {
  if (keepDebugText) return text;
  return text
    .replace(/working memory/gi, "会话上下文")
    .replace(/原生 tool calling/gi, "工具调用")
    .replace(/tool calling/gi, "工具调用")
    .replace(/tool_calls?/gi, "工具调用")
    .replace(/ReAct/g, "执行流程")
    .replace(/reasoning\s+\d+\s*字；?/gi, "内部规划已更新；");
}

function isToolActivityEvent(event: AgentActivityEvent) {
  return Boolean(event.tool_name && (event.title.startsWith("执行 ") || event.title.endsWith(" 结果")));
}

function isModelRequestCommand(event: AgentActivityEvent) {
  return Boolean((event.command ?? "").startsWith("LLM tool planning"));
}

function isReadCommand(event: AgentActivityEvent) {
  const command = commandExecutableName(event.command);
  return ["cat", "sed", "nl", "head", "tail"].includes(command);
}

function isSearchCommand(event: AgentActivityEvent) {
  const command = commandExecutableName(event.command);
  return ["rg", "grep", "find"].includes(command);
}

function isListCommand(event: AgentActivityEvent) {
  const command = commandExecutableName(event.command);
  return ["ls", "wc"].includes(command);
}

function commandExecutableName(command?: string) {
  if (!command) return "";
  const parts = command.trim().split(/\s+/).filter(Boolean);
  const executable = parts.find((part) => !/^[A-Za-z_][A-Za-z0-9_]*=/.test(part)) ?? "";
  return executable.split(/[\\/]/).pop() ?? executable;
}

function uniqueActivityToolNames(events: AgentActivityEvent[]) {
  return Array.from(new Set(events.map((event) => event.tool_name).filter(Boolean) as string[]));
}

function fileNameFromPath(path: string) {
  return path.split(/[\\/]/).filter(Boolean).pop() ?? path;
}

function joinChineseParts(parts: string[]) {
  if (parts.length <= 1) return parts[0] ?? "";
  return `${parts.slice(0, -1).join("、")}和${parts[parts.length - 1]}`;
}

function commandStatusText(status: NonNullable<AgentActivityEvent["command_status"]>) {
  if (status === "success") return "成功";
  if (status === "error") return "失败";
  if (status === "approval_required") return "等待确认";
  return "运行中";
}

function pendingApprovalEvent(record: ActivityRecord) {
  for (let index = record.events.length - 1; index >= 0; index -= 1) {
    const event = record.events[index];
    if (event.approval_resolved) {
      return null;
    }
    if (
      event.activity_type === "command" &&
      (event.command_status === "approval_required" || event.approval_required)
    ) {
      return event;
    }
  }
  return null;
}

function appendActivityDelta(
  items: AgentActivityEvent[],
  delta: Extract<AgentStreamEvent, { event: "activity_delta" }>
) {
  const existingIndex = items.findIndex((item) => item.id === delta.id);
  if (existingIndex === -1) {
    return [
      ...items,
      {
        event: "activity" as const,
        id: delta.id,
        phase: delta.phase,
        title: delta.title,
        content: delta.content,
        detail: delta.detail,
        activity_type: delta.activity_type,
        command: delta.command,
        command_status: delta.command_status,
        risk_category: delta.risk_category,
        approval_required: delta.approval_required,
        approval_preview: delta.approval_preview,
        approval_resolved: delta.approval_resolved,
        approval_batch_count: delta.approval_batch_count,
        approval_batch_remaining: delta.approval_batch_remaining,
        approval_batch_commands: delta.approval_batch_commands,
        file_path: delta.file_path,
        additions: delta.additions,
        deletions: delta.deletions,
        step: delta.step,
        tool_name: delta.tool_name,
        selected_skill: delta.selected_skill,
        elapsed_ms: delta.elapsed_ms
      }
    ];
  }
  return items.map((item, index) =>
    index === existingIndex
      ? {
          ...item,
          phase: delta.phase ?? item.phase,
          title: delta.title ?? item.title,
          content:
            delta.append_mode === "replace"
              ? delta.content
              : `${item.content ?? ""}${delta.content}`,
          detail: delta.detail ?? item.detail,
          activity_type: delta.activity_type ?? item.activity_type,
          command: delta.command ?? item.command,
          command_status: delta.command_status ?? item.command_status,
          risk_category: delta.risk_category ?? item.risk_category,
          approval_required: delta.approval_required ?? item.approval_required,
          approval_preview: delta.approval_preview ?? item.approval_preview,
          approval_resolved: delta.approval_resolved ?? item.approval_resolved,
          approval_batch_count: delta.approval_batch_count ?? item.approval_batch_count,
          approval_batch_remaining: delta.approval_batch_remaining ?? item.approval_batch_remaining,
          approval_batch_commands: delta.approval_batch_commands ?? item.approval_batch_commands,
          file_path: delta.file_path ?? item.file_path,
          additions: delta.additions ?? item.additions,
          deletions: delta.deletions ?? item.deletions,
          step: delta.step ?? item.step,
          tool_name: delta.tool_name ?? item.tool_name,
          selected_skill: delta.selected_skill ?? item.selected_skill,
          elapsed_ms: delta.elapsed_ms ?? item.elapsed_ms
        }
      : item
  );
}

function accountConversationStorageKey(username?: string) {
  const account = (username || "anonymous").trim().toLowerCase();
  return `${conversationStorageKey}:${account}`;
}

function loadConversationHistory(username?: string) {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(accountConversationStorageKey(username));
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return orderConversationHistory(
      parsed.filter(isConversationHistoryItem).map(sanitizeConversationHistoryItem)
    );
  } catch {
    return [];
  }
}

function createConversationId() {
  return `local-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function saveConversationHistory(items: ConversationHistoryItem[], username?: string) {
  if (!username) return;
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(
      accountConversationStorageKey(username),
      JSON.stringify(items.map(sanitizeConversationHistoryItem))
    );
  } catch {
    // Local storage can be unavailable in private or restricted browser modes.
  }
}

function sanitizeConversationHistoryItem(item: ConversationHistoryItem): ConversationHistoryItem {
  return {
    ...item,
    messages: item.messages.map(sanitizeChatMessage),
    activities: item.activities ? sanitizeActivityRecords(item.activities) : item.activities
  };
}

function sanitizeChatMessage(message: ChatMessage): ChatMessage {
  if (message.role !== "assistant" || !containsToolCallMarkup(message.content)) return message;
  const content = stripToolCallMarkup(message.content).trim();
  return {
    ...message,
    content: content || "工具调用过程已隐藏。请重新发送上一条请求继续。"
  };
}

function sanitizeActivityRecords(records: ActivityRecordMap): ActivityRecordMap {
  return Object.fromEntries(
    Object.entries(records).map(([key, record]) => [
      key,
      {
        ...record,
        events: record.events.map(sanitizeActivityEvent)
      }
    ])
  );
}

function sanitizeActivityEvent(event: AgentActivityEvent): AgentActivityEvent {
  const next = { ...event };
  if (typeof next.detail === "string" && containsToolCallMarkup(next.detail)) {
    next.detail = stripToolCallMarkup(next.detail).trim() || "工具调用过程已隐藏。";
  }
  if (typeof next.content === "string" && containsToolCallMarkup(next.content)) {
    next.content = stripToolCallMarkup(next.content).trim() || "工具调用过程已隐藏。";
  }
  return next;
}

function containsToolCallMarkup(text: string) {
  return toolCallMarkupPattern.test(text);
}

function stripToolCallMarkup(text: string) {
  return text
    .replace(
      /<\s*(?:tool_calls|工具调用列表)(?=[\s>/])[^>]*>[\s\S]*?<\/\s*(?:tool_calls|工具调用列表)\s*>/gi,
      ""
    )
    .replace(
      /<\s*(?:tool_call|工具调用)(?=[\s>/])[^>]*>[\s\S]*?<\/\s*(?:tool_calls?|工具调用(?:列表)?)\s*>/gi,
      ""
    )
    .replace(/<\s*(?:tool_call|工具调用)(?=[\s>/])[^>]*\/\s*>/gi, "")
    .replace(/<\s*(?:tool_calls?|工具调用(?:列表)?)(?=[\s>/])[^>]*>[\s\S]*$/gi, "")
    .replace(/<\/?\s*(?:tool_calls?|工具调用(?:列表)?)(?=[\s>/])[^>]*>/gi, "");
}

function isConversationHistoryItem(value: unknown): value is ConversationHistoryItem {
  if (!value || typeof value !== "object") return false;
  const item = value as ConversationHistoryItem;
  const activities = item.activities;
  return (
    typeof item.id === "string" &&
    typeof item.title === "string" &&
    typeof item.group === "string" &&
    Array.isArray(item.messages) &&
    item.messages.every(
      (message) =>
        message &&
        (message.role === "user" || message.role === "assistant") &&
        typeof message.content === "string"
    ) &&
    (item.contextSummary === undefined || typeof item.contextSummary === "string") &&
    (item.contextSummaryMessageCount === undefined ||
      typeof item.contextSummaryMessageCount === "number") &&
    (activities === undefined || isActivityRecordMap(activities)) &&
    (item.activeActivityIndex === undefined ||
      item.activeActivityIndex === null ||
      typeof item.activeActivityIndex === "number") &&
    (item.pinned === undefined || typeof item.pinned === "boolean") &&
    (item.projectId === undefined || typeof item.projectId === "string")
  );
}

function isActivityRecordMap(value: unknown): value is ActivityRecordMap {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  return Object.entries(value).every(
    ([key, record]) => Number.isInteger(Number(key)) && isActivityRecord(record)
  );
}

function isActivityRecord(value: unknown): value is ActivityRecord {
  if (!value || typeof value !== "object") return false;
  const record = value as ActivityRecord;
  return (
    Array.isArray(record.events) &&
    record.events.every(isAgentActivityEvent) &&
    typeof record.elapsedMs === "number" &&
    typeof record.completed === "boolean"
  );
}

function isAgentActivityEvent(value: unknown): value is AgentActivityEvent {
  if (!value || typeof value !== "object") return false;
  const event = value as AgentActivityEvent;
  return (
    event.event === "activity" &&
    (event.phase === "thinking" ||
      event.phase === "action" ||
      event.phase === "observation" ||
      event.phase === "complete" ||
      event.phase === "error") &&
    typeof event.title === "string" &&
    (event.id === undefined || typeof event.id === "string") &&
    (event.detail === undefined || typeof event.detail === "string") &&
    (event.content === undefined || typeof event.content === "string") &&
    (event.activity_type === undefined ||
      event.activity_type === "command" ||
      event.activity_type === "file_edit") &&
    (event.command === undefined || typeof event.command === "string") &&
    (event.command_status === undefined ||
      event.command_status === "running" ||
      event.command_status === "success" ||
      event.command_status === "error" ||
      event.command_status === "approval_required") &&
    (event.risk_category === undefined || typeof event.risk_category === "string") &&
    (event.approval_required === undefined || typeof event.approval_required === "boolean") &&
    (event.approval_preview === undefined || typeof event.approval_preview === "string") &&
    (event.approval_resolved === undefined || typeof event.approval_resolved === "boolean") &&
    (event.approval_batch_count === undefined || typeof event.approval_batch_count === "number") &&
    (event.approval_batch_remaining === undefined ||
      typeof event.approval_batch_remaining === "number") &&
    (event.approval_batch_commands === undefined || Array.isArray(event.approval_batch_commands)) &&
    (event.file_path === undefined || typeof event.file_path === "string") &&
    (event.additions === undefined || typeof event.additions === "number") &&
    (event.deletions === undefined || typeof event.deletions === "number") &&
    (event.step === undefined || typeof event.step === "number") &&
    (event.tool_name === undefined || typeof event.tool_name === "string") &&
    (event.selected_skill === undefined ||
      event.selected_skill === null ||
      typeof event.selected_skill === "string") &&
    (event.elapsed_ms === undefined || typeof event.elapsed_ms === "number") &&
    (event.trace_id === undefined || typeof event.trace_id === "string") &&
    (event.debug_trace_path === undefined || typeof event.debug_trace_path === "string")
  );
}

function updateConversationMessages(
  items: ConversationHistoryItem[],
  id: string,
  messages: ChatMessage[],
  activities: ActivityRecordMap,
  activeActivityIndex: number,
  fallbackTitle = titleFromMessages(messages),
  contextSummary = "",
  contextSummaryMessageCount = 0,
  projectId?: string
) {
  const current = items.find((item) => item.id === id);
  if (!current) {
    return upsertConversation(items, {
      id,
      title: fallbackTitle,
      group: "最近",
      messages,
      contextSummary,
      contextSummaryMessageCount,
      activities,
      activeActivityIndex,
      projectId
    });
  }
  return upsertConversation(items, {
    ...current,
    messages,
    contextSummary,
    contextSummaryMessageCount,
    activities,
    activeActivityIndex
  });
}

function shouldGenerateModelTitle(item: ConversationHistoryItem) {
  if (item.title !== pendingConversationTitle) return false;
  const hasUser = item.messages.some((message) => message.role === "user" && message.content.trim());
  const hasAssistant = item.messages.some(
    (message) => message.role === "assistant" && message.content.trim()
  );
  return hasUser && hasAssistant;
}

function titleFromMessages(messages: ChatMessage[]) {
  const transcript = messages.map((message) => message.content).join("\n");
  const meetingName = inferMeetingNameForTitle(messages);
  if (meetingName && /会议|纪要|会议记录|会议计较|录音|音频|\.m4a|\.mp3|\.wav|asr|转写/i.test(transcript)) {
    return cleanConversationTitle(`${meetingName}会议纪要`, untitledConversationTitle);
  }

  const attachment = inferFirstAttachmentFromMessages(messages);
  if (attachment) {
    const stem = attachment.name.replace(/\.[^.]+$/, "").trim();
    if (stem) {
      const suffix = attachment.kind === "audio" ? "录音处理" : "文件处理";
      return cleanConversationTitle(`${stem}${suffix}`, untitledConversationTitle);
    }
  }

  const firstUser = messages.find((message) => message.role === "user")?.content ?? "";
  const normalized = stripAttachmentBlock(firstUser).replace(/\s+/g, " ").trim();
  if (normalized) return normalized.length > 24 ? `${normalized.slice(0, 24).trim()}…` : normalized;
  return untitledConversationTitle;
}

function cleanConversationTitle(value: string, fallback: string) {
  const title = value
    .replace(/^(对话)?标题\s*[:：]\s*/, "")
    .replace(/["'“”‘’`]/g, "")
    .replace(/\s+/g, " ")
    .trim();
  if (!title || title === pendingConversationTitle || title === untitledConversationTitle) return fallback;
  return title.length > 28 ? `${title.slice(0, 28).trim()}…` : title;
}

function titleFromChatInput(content: string, attachments: AttachmentItem[]) {
  const normalized = content.replace(/\s+/g, " ").trim();
  if (normalized) return normalized.length > 24 ? `${normalized.slice(0, 24)}…` : normalized;
  if (attachments.length > 0) return `处理 ${attachments[0].name}`;
  return untitledConversationTitle;
}

function titleFromAssistantReply(content: string) {
  const normalized = content
    .replace(/^[\s"'“”‘’`]+|[\s"'“”‘’`]+$/g, "")
    .replace(/[#*_>`-]/g, "")
    .replace(/\s+/g, " ")
    .trim();
  if (!normalized) return untitledConversationTitle;
  const firstClause = normalized.split(/[。！？!?；;]/, 1)[0]?.trim() || normalized;
  const compact = firstClause
    .replace(/^抱歉[，,]?\s*/, "")
    .replace(/^你好[，,]?\s*/, "")
    .trim();
  const title = compact || normalized;
  return title.length > 24 ? `${title.slice(0, 24).trim()}…` : title;
}

function inferMeetingNameForTitle(messages: ChatMessage[]) {
  for (const message of [...messages].reverse()) {
    const explicit = message.content.match(/会议名称\s*[:：]\s*([^\n，,。；;]{2,24})/);
    if (explicit?.[1]) return cleanupTitlePhrase(explicit[1]);
    if (message.role === "user") {
      const cleaned = cleanupTitlePhrase(stripAttachmentBlock(message.content));
      if (cleaned.length >= 2 && cleaned.length <= 16 && /[\u4e00-\u9fff]/.test(cleaned)) {
        return cleaned;
      }
    }
  }
  return "";
}

function inferFirstAttachmentFromMessages(messages: ChatMessage[]) {
  const transcript = messages.map((message) => message.content).join("\n");
  const match = transcript.match(/- \[([^\]]+)\]\s*([^:\n]+):\s*(meet_files\/attachments\/[^\s\n]+)/);
  if (!match) return null;
  return {
    label: match[1],
    name: match[2].trim(),
    path: match[3],
    kind: match[1] === "音频" ? "audio" : "file"
  };
}

function stripAttachmentBlock(value: string) {
  return value.replace(/\n*参考附件：[\s\S]*$/, "").trim();
}

function cleanupTitlePhrase(value: string) {
  return value
    .replace(/\s+/g, "")
    .replace(/^[：:，,。；;\s]+|[：:，,。；;\s]+$/g, "")
    .slice(0, 18);
}

function inferSkillFromText(value: string, skills: SkillInfo[]) {
  return skills.find((skill) => value.includes(skill.mention) || value.includes(`@${skill.label}`)) ?? null;
}

function hasDraggedFiles(event: DragEvent<HTMLElement>) {
  return Array.from(event.dataTransfer.types).includes("Files");
}

function formatMessageWithAttachments(content: string, items: AttachmentItem[]) {
  if (items.length === 0) return content;
  const body = content || "请根据以下参考附件继续处理。";
  const attachmentLines = items.map(
    (item) => `- [${labelForAttachment(item.kind)}] ${item.name}: ${item.path}`
  );
  return `${body}\n\n参考附件：\n${attachmentLines.join("\n")}`;
}

function collectConversationFileReferences(
  messages: ChatMessage[],
  attachments: AttachmentItem[],
  activities: ActivityRecordMap
) {
  const seen = new Set<string>();
  const paths: string[] = [];
  const pushPath = (path: string) => {
    const normalized = normalizeLocalFileReference(path);
    if (!normalized || seen.has(normalized)) return;
    seen.add(normalized);
    paths.push(normalized);
  };

  for (const item of attachments) {
    pushPath(item.path);
  }
  for (const message of messages) {
    for (const path of extractLocalFileReferences(message.content)) {
      pushPath(path);
    }
  }
  for (const record of Object.values(activities)) {
    for (const event of record.events) {
      const text = [event.title, event.detail, event.content, event.tool_name]
        .filter(Boolean)
        .join("\n")
        .slice(0, 50000);
      for (const path of extractLocalFileReferences(text)) {
        pushPath(path);
      }
    }
  }

  return paths.slice(0, 80);
}

function extractLocalFileReferences(text: string) {
  const pattern =
    /(?:file:\/\/)?(?:\/[^\s`'"<>|]*\/)?(?:meet_files|meeting_audio_minutes|work_agent_skills|web_frontend|work_agent_core|config|schemas|tmp|产出材料|分析材料|学习笔记)\/[^\s`'"<>|\\\u0000-\u001f]+/giu;
  return Array.from(text.matchAll(pattern), (match) => match[0]);
}

function mergeAttachmentsByPath(existing: AttachmentItem[], next: AttachmentItem[]) {
  const seen = new Set(existing.map((item) => item.path));
  const merged = [...existing];
  for (const item of next) {
    if (seen.has(item.path)) continue;
    seen.add(item.path);
    merged.push(item);
  }
  return merged;
}

function labelForAttachment(kind: AttachmentItem["kind"]) {
  if (kind === "audio") return "音频";
  if (kind === "image") return "图片";
  if (kind === "document") return "文档";
  return "文件";
}

function iconForAttachment(kind: AttachmentItem["kind"]) {
  if (kind === "audio") return <Music2 aria-hidden="true" />;
  if (kind === "image") return <ImageIcon aria-hidden="true" />;
  if (kind === "document") return <FileText aria-hidden="true" />;
  return <Paperclip aria-hidden="true" />;
}

function activityPhaseLabel(phase: AgentActivityEvent["phase"]) {
  if (phase === "thinking") return "准备";
  if (phase === "action") return "执行";
  if (phase === "observation") return "结果";
  if (phase === "complete") return "完成";
  return "异常";
}


function formatStreamErrorDetail(event: Extract<AgentStreamEvent, { event: "error" }>) {
  const lines: string[] = [];
  if (event.type) lines.push(`错误类型：${event.type}`);
  if (event.detail && event.detail !== event.message) lines.push(`原始错误：${event.detail}`);
  if (event.trace?.length) {
    lines.push("Traceback：");
    lines.push(event.trace.join("\n"));
  }
  return lines.join("\n");
}

function streamErrorMessage(event: Extract<AgentStreamEvent, { event: "error" }>) {
  const rawText = `${event.message} ${event.detail ?? ""}`.toLowerCase();
  if (rawText.includes("http 402") || rawText.includes("insufficient balance")) {
    return "模型服务余额不足，本轮已停止。充值或切换到可用模型后重试。";
  }
  return friendlyRuntimeError(event.message || event.detail || "模型请求失败。");
}

function mergeStreamErrorActivity(
  items: AgentActivityEvent[],
  event: Extract<AgentStreamEvent, { event: "error" }>,
  errorMessage: string
) {
  let errorIndex = -1;
  for (let index = items.length - 1; index >= 0; index -= 1) {
    if (items[index].phase === "error" || items[index].command_status === "error") {
      errorIndex = index;
      break;
    }
  }

  const errorDetail = formatStreamErrorDetail(event);
  if (errorIndex === -1) {
    return [
      ...items,
      {
        event: "activity" as const,
        phase: "error" as const,
        title: "处理失败",
        detail: errorMessage,
        content: errorDetail
      }
    ];
  }

  return items.map((item, index) => {
    if (index !== errorIndex) return item;
    const existingContent = (item.content ?? "").trimEnd();
    const content =
      errorDetail && !existingContent.includes(errorDetail)
        ? [existingContent, errorDetail].filter(Boolean).join("\n\n")
        : existingContent;
    return {
      ...item,
      phase: "error" as const,
      command_status: item.activity_type === "command" ? "error" as const : item.command_status,
      detail: errorMessage,
      content
    };
  });
}

function iconForActivity(phase: AgentActivityEvent["phase"]) {
  if (phase === "thinking") return <Sparkles aria-hidden="true" />;
  if (phase === "action") return <Wrench aria-hidden="true" />;
  if (phase === "observation") return <Search aria-hidden="true" />;
  if (phase === "complete") return <CheckCircle2 aria-hidden="true" />;
  return <AlertCircle aria-hidden="true" />;
}

function formatActivityDuration(ms: number) {
  if (!Number.isFinite(ms) || ms <= 0) return "0s";
  const seconds = Math.max(1, Math.round(ms / 1000));
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const rest = seconds % 60;
  return rest ? `${minutes}m ${rest}s` : `${minutes}m`;
}

function formatClockTime(timestamp: number) {
  if (!timestamp) return "";
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit"
  }).format(new Date(timestamp));
}

function iconForLibraryFile(file: Pick<FileItem, "kind" | "extension">) {
  return iconForAttachment(getLibraryKind(file));
}

function filterLibraryFiles(files: FileItem[], query: string, filter: FileFilter) {
  const normalizedQuery = query.trim().toLowerCase();
  return files.filter((file) => {
    const matchesFilter =
      filter === "all" ||
      (filter === "output" ? isGeneratedOutput(file) : getLibraryKind(file) === filter);
    if (!matchesFilter) return false;
    if (!normalizedQuery) return true;
    return `${file.name} ${file.path} ${file.extension}`.toLowerCase().includes(normalizedQuery);
  });
}

function countLibraryFiles(files: FileItem[]): Record<FileFilter, number> {
  return {
    all: files.length,
    audio: files.filter((file) => getLibraryKind(file) === "audio").length,
    image: files.filter((file) => getLibraryKind(file) === "image").length,
    document: files.filter((file) => getLibraryKind(file) === "document").length,
    output: files.filter(isGeneratedOutput).length
  };
}

function getLibraryKind(file: Pick<FileItem, "kind" | "extension">): AttachmentItem["kind"] {
  if (file.kind) return file.kind;
  const extension = file.extension.toLowerCase();
  if ([".m4a", ".mp3", ".wav", ".aac", ".flac", ".ogg", ".opus", ".wma", ".amr", ".aiff", ".aif", ".caf"].includes(extension)) {
    return "audio";
  }
  if ([".png", ".jpg", ".jpeg", ".webp", ".gif", ".heic", ".tif", ".tiff"].includes(extension)) {
    return "image";
  }
  if ([".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".md", ".txt", ".json", ".yml", ".yaml"].includes(extension)) {
    return "document";
  }
  return "file";
}

function isLibraryPreviewable(file: Pick<FileItem, "extension" | "previewable">) {
  if (typeof file.previewable === "boolean") return file.previewable;
  return [".md", ".txt", ".json", ".yml", ".yaml", ".csv", ".log", ".srt", ".vtt", ".py", ".ts", ".tsx", ".js", ".jsx", ".css", ".html"].includes(
    file.extension.toLowerCase()
  );
}

function isMarkdownFile(file: Pick<FileItem, "extension">) {
  return file.extension.toLowerCase() === ".md";
}

function fileTitleForReader(name: string) {
  return name.replace(/\.[^.]+$/, "").replace(/^\d{8}-\d{6}-/, "").trim() || name;
}

function isGeneratedOutput(file: FileItem) {
  if (file.path.includes("/attachments/")) return false;
  const name = file.name;
  const extension = file.extension.toLowerCase();
  const outputExtensions = new Set([".md", ".txt", ".docx", ".pdf", ".xlsx", ".pptx"]);
  if (!outputExtensions.has(extension)) return false;
  return (
    name.includes("内部留档版") ||
    name.includes("工作提交版") ||
    name.includes("完善版") ||
    name.includes("ASR转写稿") ||
    name.includes("会议沟通内容整理") ||
    name.includes("会议纪要")
  );
}

function buildMeetingGroups(archives: MeetingArchive[], files: FileItem[]): MeetingGroup[] {
  const archivedPaths = new Set<string>();
  const archiveGroups: MeetingGroup[] = archives.map((archive) => {
    const asr = archive.outputs.asr?.exists ? archive.outputs.asr : undefined;
    const internal = archive.outputs.internal?.exists ? archive.outputs.internal : undefined;
    const work = archive.outputs.work_md?.exists ? archive.outputs.work_md : undefined;
    const workDocx = archive.outputs.work_docx?.exists ? archive.outputs.work_docx : undefined;
    for (const output of [asr, internal, work, workDocx]) {
      if (output?.path) archivedPaths.add(output.path);
    }
    return {
      key: `archive:${archive.manifest_path}`,
      manifestPath: archive.manifest_path,
      title: archive.title || archive.meeting_id || "未命名会议",
      meetingTime: archive.meeting_time,
      modified:
        archive.updated_at ||
        Math.max(asr?.modified || 0, internal?.modified || 0, work?.modified || 0, workDocx?.modified || 0),
      asr,
      internal,
      work,
      workDocx
    };
  });
  if (archiveGroups.length > 0) {
    return archiveGroups.sort(compareMeetingGroupsByMeetingTime);
  }
  const archivedTitleKeys = archiveGroups.map((group) => meetingGroupTitleKey(group.title)).filter(Boolean);
  const legacyGroups = groupMeetingOutputs(files.filter((file) => !archivedPaths.has(file.path))).filter(
    (group) => !archivedTitleKeys.some((key) => meetingTitleKeysOverlap(meetingGroupTitleKey(group.title), key))
  );
  return [...archiveGroups, ...legacyGroups].sort(compareMeetingGroupsByMeetingTime);
}

function groupMeetingOutputs(files: FileItem[]): MeetingGroup[] {
  const groups = new Map<string, MeetingGroup>();

  for (const file of files) {
    const name = file.name;
    const extension = file.extension.toLowerCase();
    const isMeetingFile = name.includes("会议") || name.includes("纪要") || name.includes("转写稿");
    const isAsr = isMeetingFile && extension === ".md" && name.includes("ASR转写稿");
    const isInternal =
      isMeetingFile &&
      extension === ".md" &&
      (name.includes("内部留档版") || name.includes("会议沟通内容整理"));
    const isWork =
      isMeetingFile &&
      extension === ".md" &&
      name.includes("工作提交版") &&
      !name.includes("内部留档版");
    const isWorkDocx = isMeetingFile && extension === ".docx" && name.includes("会议纪要");
    if (!isAsr && !isInternal && !isWork && !isWorkDocx) continue;
    const title = name
      .replace(/\.(md|docx)$/i, "")
      .replace(/_?会议沟通内容整理_ASR转写稿_Qwen3$/u, "")
      .replace(/_?会议沟通内容整理_Qwen3内部留档版$/u, "")
      .replace(/_?会议沟通内容整理_内部留档版$/u, "")
      .replace(/_?会议纪要_Qwen3工作提交版$/u, "")
      .replace(/_?会议纪要_工作提交版$/u, "")
      .replace(/会议纪要_工作提交版$/u, "")
      .replace(/_?工作提交版$/u, "")
      .replace(/会议纪要$/u, "");
    const normalizedTitle = normalizeMeetingGroupTitle(title || file.name);
    const key = normalizedTitle || file.name;
    const group = groups.get(key) ?? { key, title: normalizedTitle || "未命名会议", modified: 0 };
    if (isAsr) group.asr = file;
    if (isInternal) group.internal = file;
    if (isWork) group.work = file;
    if (isWorkDocx && (!group.workDocx || file.modified > group.workDocx.modified)) group.workDocx = file;
    group.modified = Math.max(group.modified, file.modified);
    groups.set(key, group);
  }

  return Array.from(groups.values()).sort((left, right) => right.modified - left.modified);
}

function normalizeMeetingGroupTitle(title: string) {
  return title.replace(/^\d{4,8}/, "").replace(/[_-]+$/, "").trim();
}

function compareMeetingGroupsByMeetingTime(left: MeetingGroup, right: MeetingGroup) {
  const leftTime = meetingGroupSortValue(left);
  const rightTime = meetingGroupSortValue(right);
  const leftHasMeetingTime = Boolean(left.meetingTime?.display?.trim());
  const rightHasMeetingTime = Boolean(right.meetingTime?.display?.trim());
  if (leftHasMeetingTime !== rightHasMeetingTime) return leftHasMeetingTime ? -1 : 1;
  return rightTime - leftTime;
}

function meetingGroupSortValue(group: MeetingGroup) {
  const explicitStart = group.meetingTime?.start?.trim();
  if (explicitStart) {
    const parsed = Date.parse(explicitStart);
    if (Number.isFinite(parsed)) return parsed;
  }
  const display = group.meetingTime?.display?.trim() || "";
  const dateMatch = display.match(/(\d{4})年(\d{1,2})月(\d{1,2})日/);
  if (dateMatch) {
    const [, year, month, day] = dateMatch;
    const timeMatch = display.match(/(\d{1,2}):(\d{2})/);
    return new Date(
      Number(year),
      Number(month) - 1,
      Number(day),
      timeMatch ? Number(timeMatch[1]) : 0,
      timeMatch ? Number(timeMatch[2]) : 0
    ).getTime();
  }
  return group.modified * 1000;
}

function meetingGroupTitleKey(title: string) {
  return normalizeMeetingGroupTitle(title)
    .replace(/座谈沟通/g, "")
    .replace(/会议纪要/g, "")
    .replace(/会议/g, "")
    .replace(/人才培养方案论证会/g, "")
    .replace(/论证会/g, "")
    .replace(/[_\-\s　]+/g, "")
    .trim()
    .toLowerCase();
}

function meetingTitleKeysOverlap(left: string, right: string) {
  if (!left || !right) return false;
  if (left === right) return true;
  if (left.length < 3 || right.length < 3) return false;
  return left.includes(right) || right.includes(left);
}

function chatTurnLabel(content: string) {
  const normalized = String(content || "").replace(/\s+/g, " ").trim();
  if (!normalized) return "未命名提问";
  const firstSentence = normalized.split(/(?<=[。！？!?])\s*/)[0] || normalized;
  return firstSentence.length > 54 ? `${firstSentence.slice(0, 54)}…` : firstSentence;
}

async function fileToBase64(file: File) {
  const dataUrl = await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(reader.error || new Error("文件读取失败"));
    reader.readAsDataURL(file);
  });
  return dataUrl.includes(",") ? dataUrl.split(",", 2)[1] : dataUrl;
}

async function blobToBase64(blob: Blob) {
  const dataUrl = await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(reader.error || new Error("录音读取失败"));
    reader.readAsDataURL(blob);
  });
  return dataUrl.includes(",") ? dataUrl.split(",", 2)[1] : dataUrl;
}

function int16FrameToBase64(frame: Int16Array) {
  const bytes = new Uint8Array(frame.buffer, frame.byteOffset, frame.byteLength);
  let binary = "";
  const chunkSize = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    const chunk = bytes.subarray(offset, offset + chunkSize);
    binary += String.fromCharCode(...chunk);
  }
  return window.btoa(binary);
}

function compactPath(path: string) {
  const parts = path.split("/");
  return parts.length > 3 ? `…/${parts.slice(-2).join("/")}` : path;
}

function formatRecordingStartedAt(value: string) {
  const match = value.match(/^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})/);
  if (!match) return value;
  return `${match[1]}-${match[2]}-${match[3]} ${match[4]}:${match[5]}:${match[6]}`;
}

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${numberFormatter.format(Math.round(bytes / 1024))} KB`;
  return `${numberFormatter.format(Math.round(bytes / 1024 / 1024))} MB`;
}

function formatTemporarySyncRemaining(expiresAt: number, clockMs: number) {
  const seconds = Math.max(0, expiresAt - Math.floor(clockMs / 1000));
  if (seconds <= 0) return "即将清理";
  const minutes = Math.ceil(seconds / 60);
  if (minutes >= 60) return "剩余约 1 小时";
  return `剩余 ${minutes} 分钟`;
}

function formatFileDate(timestamp: number) {
  const date = new Date(timestamp * 1000);
  const now = new Date();
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const fileTime = date.getTime();
  if (fileTime >= startOfToday) return "今天";
  if (fileTime >= startOfToday - 24 * 60 * 60 * 1000) return "昨天";
  return dateFormatter.format(date);
}

function formatMeetingDate(meetingTime: MeetingTime | null | undefined) {
  const display = meetingTime?.display?.trim();
  return display || "会议时间未记录";
}
