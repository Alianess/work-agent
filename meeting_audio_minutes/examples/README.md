# 示例

把会议录音放在任意路径后运行：

```bash
cd /path/to/work_agent/meeting_audio_minutes
python -m meeting_minutes /path/to/meeting.m4a --model medium --language zh --denoise-backend ffmpeg
```

如果会议里有固定项目名、人名或产品名：

```bash
python -m meeting_minutes /path/to/meeting.m4a \
  --model medium \
  --language zh \
  --denoise-backend ffmpeg \
  --initial-prompt "本次会议涉及 Work Agent、机器人产业数据源、客户线索、知识库建设。"
```
