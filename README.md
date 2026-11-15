# 分段看视频

让机器人“按段看完”一个本地视频：每一段做分镜抽帧、拼成九宫格，配上这一段的语音转写，再交给视觉模型写一段观后感，逐段推送，最后给整片总结。

支持的输入：

- 把视频丢进 `watch_dir`（默认丢到插件 `data/inbox`），在 QQ 里发 `/kv 文件名`；
- 直接把视频文件发给机器人（QQ 私聊）。

不限制时长——一小时、三小时都可以，按段滚动处理。

## 依赖

- `ffmpeg.exe` / `ffprobe.exe`：填 `ffmpeg_path` / `ffprobe_path`，或放进数据目录 `bin`、系统 PATH。
- Python 包：`pip install -r requirements.txt`（`funasr-onnx`、`onnxruntime`）。
- 语音转写（可选）：`asr_model_dir` 留空时，首次运行会自动把 SenseVoice ONNX 模型（约 240MB）下载到插件 `data/models/sensevoice`；也可以指向已有的模型目录。没配模型时只看画面，不影响主流程。
- 一个支持图片输入的对话 Provider（用于读九宫格）。

## 命令

| 命令 | 作用 |
| --- | --- |
| `/kv 文件名 [段数]` | 开始看视频，可选只看前 N 段 |
| `/kv zoom 段 格` | 把某一段的某一格再细分看（例如 `/kv zoom 3 5`） |
| `/kv status` | 看当前进度 |
| `/kv stop` | 停下当前任务 |
| `/kv list` | 列出可看的视频 |

`zoom` 依赖缓存里的分段记录，所以要先 `/kv` 看过那一段。

## 工作方式

1. `ffprobe` 取时长；
2. 每 `segment_seconds` 秒为一个窗口，窗口内用场景检测找切点；把窗口等分成 `frames_per_segment` 份，每份取「最长镜头的中间帧」，既有时间覆盖又避开转场糊帧；
3. 每格取一帧，拼成 3 列九宫格（带编号和时间轴）；
4. 抽这 5 分钟的音频，切块后交给 SenseVoice 转写；
5. 拼图 + 转写一起给视觉模型，写「这一段发生了什么」；
6. 逐段推送图和摘要，最后用所有分段摘要合成整片总结。

二级特写：每段结束后，如果开了 `auto_zoom`，会挑切点最多的一格再展开成一张更细的九宫格；也可以随时用 `/kv zoom` 手动展开。

## 配置

面板「已安装 → 分段看视频 → 参数设置」里可以直接改，保存后下一段生效：

- `segment_seconds`：每段时长（默认 300）
- `frames_per_segment`：每段格数（默认 9）
- scene_threshold：切镜灵敏度（0.2 更碎，0.5 更粗）
- double_frame：快剪双联帧（平均镜头 < 2 秒时，每格用前后两帧左右拼图）
- `zoom_enable` / `auto_zoom` / `zoom_frames`：特写开关与格数
- `asr_enable` / `asr_language`：语音转写开关与语种（`auto`/`zh`/`ja`/`en`/`ko`/`yue`）
- `asr_model_dir` / `asr_auto_download`：模型目录（留空为插件内目录）与是否自动下载
- `push_progress`：是否逐段推送
- `watch_dir`：视频文件夹（留空时用插件 `data/inbox`）
- `max_segments_per_run`：单次最多看几段，0 为不限

## 缓存与断点

分段结果存在 AstrBot 的 `data/plugin_data/astrbot_plugin_video_storyboard/cache/<视频指纹>/` 下：每格的帧、拼图、音频与 `log.json`（每段摘要、转写、特写记录）。

同一个视频再发 `/kv`，会复用已完成的分段，从断点继续；要重跑就先删掉对应缓存目录。

## 成本

按 5 分钟 9 格算：一小时约 12 段、12 张图；每段转写 + 一次视觉调用。实际耗时取决于 ASR 和出图速度，本机 29 分钟的视频大约 8–9 分钟跑完。

把 `frames_per_segment` 调大、`segment_seconds` 调小都会提高密度，也等比增加 token 和耗时。

## 注意

- 视觉模型看不到视频流，它看到的是拼图和转写文本；
- 内嵌字幕轨不会被解析；硬字幕靠视觉模型顺带读；
- 网页桥会话关闭后消息可能发不出去，进度推送都带了超时；
- 若同时跑了多个 AstrBot 实例，只有真正持有 6199 的那个会收到 QQ 消息。

## P3：章节索引与回看

看完后会自动生成章节目录（每段一个标题 + 关键词），存在 `log.json` 的 `film` 字段里。

- `/kv chapters` —— 列章节目录（时间 + 这一段讲什么）
- `/kv jump 09:00` —— 跳到 9 分钟那一段，回看它的摘要和九宫格
- `/kv jump 反转` —— 按关键词找相关段
- `/kv ask 这视频讲了什么` —— 基于已看的段落回答（会带上时间点，只依据视频内容，不瞎编）

老缓存（0.3.0 之前看的视频）没有标题字段，会退化成「第 N 段」；重新看一遍就会补上。
