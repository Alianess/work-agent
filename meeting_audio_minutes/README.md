# Meeting Audio Minutes

这个目录现在只保留一条本地会议录音 ASR 主链路：

```text
会议录音
  -> FFmpeg afftdn 降噪/标准化到 16 kHz mono WAV
  -> FSMN VAD 找说话边界并合并成约 90–120 秒块
  -> Qwen3-ASR MLX 8bit 转写（Mac Apple Silicon 推荐）
  -> 会议沟通内容整理 / 工作提交版纪要
```

## 当前保留的模型

- 主 ASR：`mlx-community/Qwen3-ASR-1.7B-8bit`（本地路径：`meeting_audio_minutes/model_cache/mlx-community/Qwen3-ASR-1.7B-8bit`）
- 分块辅助：FSMN VAD 小模型 `speech_fsmn_vad_zh-cn-16k-common-pytorch`

旧的本地 ASR 路径不再作为项目链路维护。

## 推荐命令

整段会议录音转写：

```bash
.venv/bin/python meeting_audio_minutes/scripts/transcribe_qwen3_asr_chunked.py \
  path/to/meeting.wav \
  --output-dir meet_files/asr_outputs/qwen3 \
  --backend mlx \
  --model-id meeting_audio_minutes/model_cache/mlx-community/Qwen3-ASR-1.7B-8bit \
  --cache-dir meeting_audio_minutes/model_cache \
  --device mlx-metal \
  --language Chinese \
  --chunk-mode vad \
  --chunk-seconds 120 \
  --workers 1
```

MLX 后端使用单个 Metal worker，`--workers` 固定为 1；项目只保留这条 MLX 8bit ASR 链路。

只生成 VAD 分块计划、不跑推理：

```bash
.venv/bin/python meeting_audio_minutes/scripts/transcribe_qwen3_asr_chunked.py \
  path/to/meeting.wav \
  --chunk-mode vad \
  --chunk-seconds 120 \
  --plan-only
```

## 输出

`transcribe_qwen3_asr_chunked.py` 会写出：

- `chunk_plan.json`：VAD 分块计划
- `summary.json`：转写汇总和每块元数据
- `transcript.md`：带时间段的 Markdown 转写
- `transcript.txt`：纯文本转写
- `items/chunk_*/`：每块单独结果

## 质量默认值

当前会议录音测试里，推荐默认：

- 降噪：FFmpeg `afftdn`
- 分块：VAD 边界合并，不机械硬切
- Qwen3 块长：`90–120s`，默认 `120s`
- 边界 padding：`300ms`

不要把每个 VAD 小段单独送进 Qwen3；那会丢上下文并造成文本碎片化。
