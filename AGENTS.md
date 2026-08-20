# 本工作区的约定

> 这是项目上下文文件，由智能体在启动时读取。写这里而不是写进系统提示词，
> 是因为它约束的是**这个工作区**，换一个工作区就不该带着走。

## 运行环境

本项目只有一套受支持的 Python 环境：工作区根目录的 `.venv`。
Python、pip、Office 脚本、会议 ASR 和 VAD 都必须用它。

不要创建或调用 `.venv_agent`、`meeting_audio_minutes/.venv_project`、`.venv_deepfilter`、
Conda、系统 Python 或临时 venv，也不要用 `--user` 安装包。

- 环境检查：`scripts/runtime_env.sh check`
- 缺依赖时说明原因，请求用户批准后再 `scripts/runtime_env.sh bootstrap`
- Node/npm 使用该脚本的 `node` / `npm` 子命令

FFmpeg、LibreOffice、Poppler 是声明过的原生工具，不代表额外的 Python 环境。
DeepFilterNet 因 Python 版本冲突不进主运行环境，音频降噪默认用 FFmpeg。

## 测试

全量回归：

```
.venv/bin/python -m unittest discover -s tests -q
```
