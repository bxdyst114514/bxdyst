"""分段看视频：把本地视频切成段，分镜抽帧 + 拼图 + 台词转写，逐段交给视觉模型。"""

import asyncio
import base64
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.message_components import Image as AstrImage, Plain, Video
from astrbot.api.star import Context, Star, register

try:
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover
    Image = None

SYSTEM_PROMPT = (
    "你正和用户一起看视频，你只能看到按时间顺序拼好的画面格子（每格标了编号和时间）。"
    "用中文说人话，像在旁边一起看的人：先概括这一段在发生什么，再点出最值得注意的一个细节。"
    "不要罗列每一格，不要机械复述台词，不要说自己看不到声音以外的东西。"
)

FINAL_PROMPT = (
    "下面是一个视频按时间顺序的分段摘要。请把它合成一段整体总结：讲清这是什么内容、"
    "结构/剧情怎么走、最值得回味的点。像跟用户聊刚看完的片子，不要逐条复述。"
)


ASR_WORKER = r'''
import json, sys, contextlib
from funasr_onnx import SenseVoiceSmall
model = None
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
        if model is None:
            with contextlib.redirect_stdout(sys.stderr):
                model = SenseVoiceSmall(req["model_dir"], batch_size=1, device_id="-1",
                                        quantize=True, intra_op_num_threads=4)
        texts = []
        for wav in req["wavs"]:
            with contextlib.redirect_stdout(sys.stderr):
                res = model(wav, language=req.get("lang") or "auto", use_itn=True)
            if isinstance(res, dict):
                res = [res]
            for item in res or []:
                if isinstance(item, dict):
                    t = str(item.get("text") or "").strip()
                elif isinstance(item, str):
                    t = item.strip()
                else:
                    t = ""
                if t:
                    texts.append(t)
        sys.stdout.buffer.write((json.dumps({"text": " ".join(texts)}, ensure_ascii=False) + "\n").encode("utf-8"))
    except Exception as exc:
        sys.stdout.buffer.write((json.dumps({"text": "", "error": str(exc)}, ensure_ascii=False) + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()
'''



@register("astrbot_plugin_video_storyboard", "xiaoman", "分段看视频", "0.1.0")
class VideoStoryboard(Star):
    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context)
        self.config = config or {}
        # 持久化数据放 AstrBot 的 data/plugin_data，而不是插件自身目录（更新插件时才不会被覆盖）
        try:
            from astrbot.api.star import StarTools

            self.base = Path(StarTools.get_data_dir("astrbot_plugin_video_storyboard"))
        except Exception:
            self.base = Path(__file__).parent / "data"
        self.base.mkdir(parents=True, exist_ok=True)
        self.cache_root = self.base / "cache"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.job: dict | None = None
        self._asr_proc = None
        self._ffmpeg = None
        self._ffprobe = None
        self._register_page_api()

    # ---------- 扩展功能页面 ----------
    PAGE_KEYS = {
        "segment_seconds": int,
        "frames_per_segment": int,
        "scene_threshold": float,
        "zoom_enable": bool,
        "auto_zoom": bool,
        "zoom_frames": int,
        "asr_enable": bool,
        "asr_language": str,
        "push_progress": bool,
        "watch_dir": str,
        "max_segments_per_run": int,
    }

    def _register_page_api(self) -> None:
        try:
            register = getattr(self.context, "register_web_api", None)
            if register is None:
                logger.info("[video-storyboard] 当前 AstrBot 未提供 register_web_api，跳过扩展页面 API")
                return
            prefix = "/astrbot_plugin_video_storyboard/page"
            register(f"{prefix}/config", self._page_get_config, ["GET"], "获取分段看视频配置")
            register(f"{prefix}/config/save", self._page_set_config, ["POST"], "保存分段看视频配置")
            register(f"{prefix}/config/set", self._page_set_config_query, ["GET"], "保存分段看视频配置（GET）")
        except Exception as exc:  # pragma: no cover
            logger.warning(f"[video-storyboard] 扩展页面 API 注册失败: {exc}")

    def _page_payload(self) -> dict:
        values = {}
        for key, typ in self.PAGE_KEYS.items():
            val = self._c(key, None)
            if typ is bool:
                values[key] = bool(val)
            elif typ is int:
                try:
                    values[key] = int(val)
                except Exception:
                    values[key] = 0
            elif typ is float:
                try:
                    values[key] = float(val)
                except Exception:
                    values[key] = 0.0
            else:
                values[key] = "" if val is None else str(val)
        return {"ok": True, "config": values}

    async def _page_get_config(self) -> dict:
        return {"status": "ok", "message": "", "data": self._page_payload()}

    async def _page_set_config(self) -> dict:
        try:
            from quart import request as quart_request
            payload = await quart_request.get_json(silent=True) or {}
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            return {"status": "error", "message": "请求体格式不对", "data": {}}
        payload = payload.get("config") if isinstance(payload.get("config"), dict) else payload
        return self._apply_page_changes(payload)

    async def _page_set_config_query(self) -> dict:
        try:
            from quart import request as quart_request
            payload = dict(quart_request.args)
        except Exception:
            payload = {}
        return self._apply_page_changes(payload)

    def _apply_page_changes(self, payload: dict) -> dict:
        if not isinstance(payload, dict):
            return {"status": "error", "message": "请求体格式不对", "data": {}}
        changed = {}
        for key, typ in self.PAGE_KEYS.items():
            if key not in payload:
                continue
            raw = payload.get(key)
            try:
                if typ is bool:
                    val = raw if isinstance(raw, bool) else str(raw).strip().lower() in ("1", "true", "yes", "on")
                elif typ is int:
                    val = int(raw)
                elif typ is float:
                    val = float(raw)
                else:
                    val = str(raw).strip()
            except Exception:
                return {"status": "error", "message": f"参数 {key} 的值不合法", "data": {}}
            changed[key] = val
        if not changed:
            return {"status": "error", "message": "没有可保存的参数", "data": {}}
        try:
            self.config.update(changed)
            save = getattr(self.config, "save_config", None)
            if callable(save):
                save()
        except Exception as exc:
            return {"status": "error", "message": f"保存失败: {exc}", "data": {}}
        self._ffmpeg = None
        self._ffprobe = None
        logger.info(f"[video-storyboard] 扩展页面保存配置: {changed}")
        return {"status": "ok", "message": "已保存，下一段起生效", "data": self._page_payload()}


    def _watch_dir(self) -> Path:
        configured = str(self._c("watch_dir", "") or "").strip().strip('"')
        path = Path(configured).expanduser() if configured else (self.base / "data" / "inbox")
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return path

    # ---------- config ----------
    def _c(self, key: str, default=None):
        try:
            val = self.config.get(key, default)
        except Exception:
            return default
        return default if val is None else val

    def _find_bin(self, name: str) -> str:
        configured = str(self._c(f"{name}_path", "") or "").strip().strip('"')
        if configured and Path(configured).is_file():
            return configured
        found = shutil.which(name) or shutil.which(name + ".exe")
        if not found:
            raise RuntimeError(f"找不到 {name}，请安装或在配置里填写路径")
        return found

    @property
    def ffmpeg(self) -> str:
        if not self._ffmpeg:
            self._ffmpeg = self._find_bin("ffmpeg")
        return self._ffmpeg

    @property
    def ffprobe(self) -> str:
        if not self._ffprobe:
            self._ffprobe = self._find_bin("ffprobe")
        return self._ffprobe

    def watch_dir(self) -> Path:
        p = self._watch_dir()
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ---------- process helpers ----------
    async def _run(self, args: list[str], timeout: float = 1800) -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            raise RuntimeError(f"命令超时：{args[0]}")
        return proc.returncode, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")

    async def probe(self, path: Path) -> dict:
        rc, out, err = await self._run(
            [
                self.ffprobe, "-v", "error",
                "-show_entries", "format=duration,size",
                "-show_entries", "stream=codec_type,width,height",
                "-of", "json", str(path),
            ],
            timeout=120,
        )
        if rc != 0:
            raise RuntimeError(f"ffprobe 失败：{err[-300:]}")
        data = json.loads(out or "{}")
        duration = float((data.get("format") or {}).get("duration") or 0)
        if duration <= 0:
            raise RuntimeError("读不出视频时长，文件可能损坏")
        return {"duration": duration, "raw": data}

    # ---------- pipeline ----------
    async def scene_cuts(self, path: Path, start: float, dur: float) -> list[float]:
        th = float(self._c("scene_threshold", 0.3))
        rc, _, err = await self._run(
            [
                self.ffmpeg, "-hide_banner", "-nostats",
                "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(path),
                "-an", "-vf", f"select='gt(scene,{th})',showinfo",
                "-f", "null", "-",
            ],
            timeout=900,
        )
        if rc != 0:
            return []
        cuts = []
        for m in re.findall(r"pts_time:([0-9]+(?:\.[0-9]+)?)", err):
            t = float(m)
            if 1.0 < t < dur - 1.0 and (not cuts or t - cuts[-1] > 1.0):
                cuts.append(t)
        return cuts

    @staticmethod
    def _plan_times(cuts: list[float], dur: float, limit: int) -> list[tuple[float, float, float]]:
        """返回 (间隔起点, 间隔终点, 取帧时刻) 列表，按间隔长度合并到 limit 个。"""
        bounds = [0.0] + sorted(cuts) + [dur]
        intervals = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
        intervals = [(a, b) for a, b in intervals if b - a > 0.8]
        if not intervals:
            intervals = [(0.0, dur)]
        while len(intervals) > limit:
            idx = min(range(len(intervals)), key=lambda i: intervals[i][1] - intervals[i][0])
            a, b = intervals.pop(idx)
            if idx < len(intervals):
                intervals[idx] = (a, intervals[idx][1])
            elif idx > 0:
                intervals[idx - 1] = (intervals[idx - 1][0], b)
            else:
                break
        # 镜头太少的视频按时间均分补齐，保证横向覆盖
        while len(intervals) < limit:
            idx = max(range(len(intervals)), key=lambda i: intervals[i][1] - intervals[i][0])
            a, b = intervals[idx]
            if b - a < 2.0:
                break
            mid = (a + b) / 2.0
            intervals[idx:idx + 1] = [(a, mid), (mid, b)]
        return [(a, b, (a + b) / 2.0) for a, b in intervals]

    async def grab_frame(self, path: Path, at: float, out: Path) -> bool:
        rc, _, _ = await self._run(
            [
                self.ffmpeg, "-hide_banner", "-nostats", "-loglevel", "error",
                "-ss", f"{at:.3f}", "-i", str(path),
                "-frames:v", "1", "-vf", "scale=480:-2", "-q:v", "4", "-y", str(out),
            ],
            timeout=120,
        )
        return rc == 0 and out.exists() and out.stat().st_size > 0

    def make_sheet(self, frames: list[tuple[Path, str]], out: Path) -> None:
        cols = 3
        cw, ch, bar = 500, 282, 26
        rows = max(1, math.ceil(len(frames) / cols))
        sheet = Image.new("RGB", (cols * cw, rows * (ch + bar)), (14, 18, 26))
        draw = ImageDraw.Draw(sheet)
        for i, (fp, label) in enumerate(frames):
            col, row = i % cols, i // cols
            x0, y0 = col * cw, row * (ch + bar)
            try:
                im = Image.open(fp).convert("RGB")
            except Exception:
                continue
            im.thumbnail((cw - 4, ch - 4))
            sheet.paste(im, (x0 + (cw - im.width) // 2, y0 + (ch - im.height) // 2))
            draw.text((x0 + 6, y0 + ch + 6), label, fill=(225, 232, 240))
        sheet.save(out, "JPEG", quality=82)


    ASR_FILES = (
        "model_quant.onnx",
        "am.mvn",
        "config.yaml",
        "tokens.json",
        "chn_jpn_yue_eng_ko_spectok.bpe.model",
    )
    ASR_BASE = "https://hf-mirror.com/DennisHuang648/SenseVoiceSmall-onnx/resolve/main/"

    def _asr_model_dir(self) -> Path:
        configured = str(self._c("asr_model_dir", "") or "").strip().strip('"')
        if configured:
            return Path(configured).expanduser()
        return self.base / "data" / "models" / "sensevoice"

    def _download_file(self, url: str, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".part")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as fh:
            while True:
                block = resp.read(1 << 20)
                if not block:
                    break
                fh.write(block)
        tmp.replace(dst)

    def _ensure_asr_model(self) -> Path:
        folder = self._asr_model_dir()
        missing = [n for n in self.ASR_FILES if not (folder / n).is_file()]
        if not missing:
            return folder
        auto = bool(self._c("asr_auto_download", True))
        if not auto:
            raise RuntimeError(
                "语音模型缺失：" + str(folder) + "（缺 " + ", ".join(missing) + "）"
                "，请把 asr_model_dir 指向已有模型，或在配置里开启 asr_auto_download"
            )
        logger.info("下载 SenseVoice 模型到 %s（缺 %d 个文件，约 240MB）", folder, len(missing))
        for name in missing:
            if name.endswith(".bpe.model"):
                url = (
                    "https://modelscope.cn/api/v1/models/iic/SenseVoiceSmall/repo"
                    "?Revision=master&FilePath=" + name
                )
            else:
                url = self.ASR_BASE + name
            logger.info("下载 %s ...", name)
            self._download_file(url, folder / name)
            logger.info("完成 %s", name)
        return folder

    def transcribe(self, wav: Path, lang: str) -> str:
        py = str(self._c("asr_python", "") or "").strip()
        if not py or not Path(py).is_file():
            py = sys.executable
        if not Path(py).is_file():
            import sys as _sys

            py = _sys.executable
        # SenseVoice 单次只适合 30 秒内，先切块
        for c in wav.parent.glob("chunk_*.wav"):
            c.unlink(missing_ok=True)
        flags = 0x08000000 if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        subprocess.run(
            [
                self.ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(wav),
                "-f", "segment", "-segment_time", "25",
                "-ac", "1", "-ar", "16000", str(wav.parent / "chunk_%03d.wav"),
            ],
            capture_output=True,
            creationflags=flags,
        )
        chunks = sorted(wav.parent.glob("chunk_*.wav"))
        if not chunks:
            chunks = [wav]
        req = json.dumps(
            {
                "wavs": [str(c) for c in chunks],
                "lang": lang or "auto",
                "model_dir": str(self._ensure_asr_model()),
            },
            ensure_ascii=False,
        )
        proc = subprocess.run(
            [py, "-u", "-c", ASR_WORKER],
            input=req + "\n",
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=1800,
            creationflags=flags,
        )
        out = (proc.stdout or "").strip()
        line = out.splitlines()[-1] if out else ""
        try:
            data = json.loads(line or "{}")
        except Exception:
            raise RuntimeError((proc.stderr or out or "ASR 无输出")[-400:])
        finally:
            for c in chunks:
                if c != wav:
                    c.unlink(missing_ok=True)
        if data.get("error"):
            raise RuntimeError(data["error"])
        text = str(data.get("text") or "")
        text = re.sub(r"<\|[^|]+\|>", "", text).strip()
        return text

    async def extract_audio(self, path: Path, start: float, dur: float, out: Path) -> bool:
        rc, _, _ = await self._run(
            [
                self.ffmpeg, "-hide_banner", "-loglevel", "error",
                "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(path),
                "-vn", "-ac", "1", "-ar", "16000", "-y", str(out),
            ],
            timeout=600,
        )
        return rc == 0 and out.exists() and out.stat().st_size > 2000

    async def vision(self, sheet: Path, timeline: str, transcript: str, idx: int, total: int) -> str:
        provider = self.context.get_using_provider()
        if provider is None:
            raise RuntimeError("没有可用的模型 provider")
        data = base64.b64encode(sheet.read_bytes()).decode()
        prompt = (
            f"这是视频第 {idx}/{total} 段的分镜图，共 {timeline.count(chr(10)) + 1} 格，按时间从上到下、从左到右。\n"
            f"时间轴：\n{timeline}\n"
        )
        if transcript:
            prompt += f"\n这一段能听到的台词（转写，可能有错字）：\n{transcript[:1500]}\n"
        else:
            prompt += "\n这一段没有可用的台词转写，只能看画面。\n"
        prompt += "\n请用 2~3 句话概括这一段，再补一句你最在意的细节。"
        resp = await provider.text_chat(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            image_urls=[f"data:image/jpeg;base64,{data}"],
        )
        return str(getattr(resp, "completion_text", "") or "").strip()

    async def clean_transcript(self, text: str) -> str:
        """给 ASR 原文补标点、纠同音错字；失败就原样返回。"""
        try:
            provider = self.context.get_using_provider()
            if provider is None:
                return text
            resp = await provider.text_chat(
                prompt=(
                    "下面是一段语音识别原文，没有标点、可能有同音错字。"
                    "请补上标点并修正明显的同音错误，保持原意和顺序，不要总结、不要增删内容。"
                    "只输出整理后的正文。\n\n" + text[:4000]
                ),
                system_prompt="你是中文语音转写校对员，只做加标点和纠错。",
            )
            out = str(getattr(resp, "completion_text", "") or "").strip()
            return out or text
        except Exception as exc:
            logger.warning(f"转写整理失败：{exc}")
            return text

    async def final_summary(self, segments: list[dict]) -> str:
        provider = self.context.get_using_provider()
        if provider is None:
            return ""
        lines = []
        for s in segments:
            lines.append(f"[{self._ts(s['start'])}-{self._ts(s['end'])}] {s.get('summary', '')}")
        resp = await provider.text_chat(
            prompt=FINAL_PROMPT + "\n\n" + "\n".join(lines),
        )
        return str(getattr(resp, "completion_text", "") or "").strip()

    async def build_zoom(self, path, seg_dir: Path, seg_start: float, a: float, b: float,
                         seg_idx: int, cell_idx: int, total: int, lang: str, asr_on: bool) -> dict:
        zdir = seg_dir / f"cell{cell_idx:03d}"
        zdir.mkdir(exist_ok=True)
        dur = max(1.0, b - a)
        cuts = await self.scene_cuts(path, seg_start + a, dur)
        per = int(self._c("zoom_frames", 9) or 9)
        plan = self._plan_times(cuts, dur, per)
        frames = []
        for k, (x, y, mid) in enumerate(plan):
            fp = zdir / f"z{k:02d}.jpg"
            if await self.grab_frame(path, seg_start + a + mid, fp):
                frames.append((fp, f"{k + 1}  {self._ts(seg_start + a + x)}"))
        sheet = zdir / "sheet.jpg"
        transcript = ""
        if asr_on:
            wav = zdir / "z.wav"
            if await self.extract_audio(path, seg_start + a, dur, wav):
                try:
                    transcript = await asyncio.to_thread(self.transcribe, wav, lang)
                    if transcript:
                        transcript = await self.clean_transcript(transcript)
                except Exception as exc:
                    logger.warning(f"细分转写失败：{exc}")
                finally:
                    wav.unlink(missing_ok=True)
        summary = ""
        if frames:
            self.make_sheet(frames, sheet)
            provider = self.context.get_using_provider()
            if provider:
                data = base64.b64encode(sheet.read_bytes()).decode()
                timeline = "\n".join(
                    f"格{j + 1}：{self._ts(seg_start + a + plan[j][0])} ~ {self._ts(seg_start + a + plan[j][1])}"
                    for j in range(len(frames))
                )
                prompt = (
                    f"这是视频第 {seg_idx + 1}/{total} 段的第 {cell_idx + 1} 格特写，"
                    f"时间 {self._ts(seg_start + a)}-{self._ts(seg_start + b)}，共 {len(frames)} 格。\n"
                    f"时间轴：\n{timeline}\n"
                )
                if transcript:
                    prompt += f"\n这几十秒的台词（转写，可能有错字）：\n{transcript[:900]}\n"
                prompt += "\n用 1~2 句话说明这几十秒画面里发生了什么，注意人物动作、表情和字幕。"
                try:
                    resp = await provider.text_chat(
                        prompt=prompt,
                        system_prompt=SYSTEM_PROMPT,
                        image_urls=[f"data:image/jpeg;base64,{data}"],
                    )
                    summary = str(getattr(resp, "completion_text", "") or "").strip()
                except Exception as exc:
                    logger.warning(f"细分分析失败：{exc}")
        return {"a": a, "b": b, "summary": summary, "transcript": transcript[:1500],
                "sheet": str(sheet) if sheet.exists() else "", "frames": len(frames)}

    async def _auto_zoom(self, job, path, seg_dir, seg_start, plan, cuts, seg_idx, total, lang, asr_on):
        if not plan:
            return None
        counts = [sum(1 for c in cuts if a <= c < b) for (a, b, _m) in plan]
        k = max(range(len(plan)), key=lambda i: counts[i])
        a, b, _m = plan[k]
        info = await self.build_zoom(path, seg_dir, seg_start, a, b, seg_idx, k, total, lang, asr_on)
        return k, info

    @staticmethod
    def _ts(sec: float) -> str:
        sec = int(max(0, sec))
        h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    # ---------- job ----------
    def _resolve(self, name: str) -> Path:
        p = Path(name.strip().strip('"'))
        if p.is_file():
            return p
        cand = self.watch_dir() / name
        if cand.is_file():
            return cand
        raise FileNotFoundError(f"找不到视频：{name}（可以放进 {self.watch_dir()}）")

    def _cache_dir(self, path: Path) -> Path:
        h = hashlib.sha1()
        stat = path.stat()
        h.update(str(path).encode("utf-8", "ignore"))
        h.update(str(stat.st_size).encode())
        h.update(str(int(stat.st_mtime)).encode())
        d = self.cache_root / h.hexdigest()[:16]
        d.mkdir(parents=True, exist_ok=True)
        return d

    async def start_job(self, event: AstrMessageEvent, path: Path, limit_segments: int = 0):
        if self.job and not self.job["task"].done():
            return f"已经有一个任务在看了（{self.job['name']}），先 /kv stop 再开新的。"
        cd = self._cache_dir(path)
        log_path = cd / "log.json"
        done = []
        if log_path.exists():
            try:
                done = json.loads(log_path.read_text("utf-8")).get("segments", [])
            except Exception:
                done = []
        umo = event.unified_msg_origin
        self.job = {
            "name": path.name, "path": path, "umo": umo, "cache": cd, "log": log_path,
            "stop": False, "segments": done,
        }
        if done:
            self.job["start_index"] = len(done)
        task = asyncio.create_task(self._worker(limit_segments))
        self.job["task"] = task
        return f"开始看《{path.name}》，共 {self._ts(self.job.get('duration', 0)) or '?'}；每段看完我会说一句。"

    async def _worker(self, limit_segments: int):
        job = self.job
        path: Path = job["path"]
        try:
            info = await self.probe(path)
            job["duration"] = info["duration"]
            seg_len = float(self._c("segment_seconds", 300))
            seg_len = max(20.0, seg_len)
            total = max(1, math.ceil(info["duration"] / seg_len))
            if limit_segments and limit_segments > 0:
                total = min(total, limit_segments)
            if int(self._c("max_segments_per_run", 0) or 0) > 0:
                total = min(total, int(self._c("max_segments_per_run")))
            per = int(self._c("frames_per_segment", 9) or 9)
            per = max(2, min(per, 16))
            asr_on = bool(self._c("asr_enable", True))
            lang = str(self._c("asr_language", "auto") or "auto")
            push = bool(self._c("push_progress", True))
            zoom_on = bool(self._c("zoom_enable", True))
            auto_zoom = bool(self._c("auto_zoom", True))
            done = job["segments"]
            start_idx = len(done)
            if start_idx >= total:
                text = "这个视频之前已经看完了。"
                if done:
                    text += "\n\n整体印象：" + (await self.final_summary(done))
                self._bg_say(job["umo"], text)
                return

            self._bg_say(
                job["umo"],
                f"开始看了：《{path.name}》{self._ts(info['duration'])}，{total} 段"
                + (f"（从第 {start_idx + 1} 段接着看）" if start_idx else ""),
            )
            for i in range(start_idx, total):
                if job["stop"]:
                    self._bg_say(job["umo"], "好，先看到这。")
                    return
                start = i * seg_len
                end = min(info["duration"], start + seg_len)
                dur = end - start
                seg_dir = job["cache"] / f"seg{i:03d}"
                seg_dir.mkdir(exist_ok=True)
                steps = [f"分镜 {self._ts(start)}-{self._ts(end)}"]
                cuts = await self.scene_cuts(path, start, dur)
                plan = self._plan_times(cuts, dur, per)
                frames = []
                for k, (a, b, mid) in enumerate(plan):
                    fp = seg_dir / f"f{k:02d}.jpg"
                    ok = await self.grab_frame(path, start + mid, fp)
                    if ok:
                        frames.append((fp, f"{k + 1}  {self._ts(start + a)}"))
                if not frames:
                    frames = []
                sheet = seg_dir / "sheet.jpg"
                transcript = ""
                if asr_on:
                    wav = seg_dir / "a.wav"
                    if await self.extract_audio(path, start, dur, wav):
                        try:
                            transcript = await asyncio.to_thread(self.transcribe, wav, lang)
                            if transcript:
                                transcript = await self.clean_transcript(transcript)
                        except Exception as exc:
                            logger.warning(f"转写失败：{exc}")
                        finally:
                            wav.unlink(missing_ok=True)
                timeline = "\n".join(
                    f"格{j + 1}：{self._ts(start + (plan[j][0]))} ~ {self._ts(start + (plan[j][1]))}"
                    for j in range(len(frames))
                )
                summary = ""
                if frames:
                    self.make_sheet(frames, sheet)
                    try:
                        summary = await self.vision(sheet, timeline, transcript, i + 1, total)
                    except Exception as exc:
                        logger.error(f"视觉分析失败：{exc}")
                        summary = f"这一段没看成（{exc}）。"
                else:
                    summary = "这一段没有抽到可用画面。"
                item = {
                    "index": i, "start": start, "end": end,
                    "summary": summary, "transcript": transcript[:3000],
                    "sheet": str(sheet) if sheet.exists() else "",
                    "cells": [[round(a, 2), round(b, 2)] for a, b, _m in plan],
                }
                done.append(item)
                job["log"].write_text(
                    json.dumps({"file": str(path), "duration": info["duration"], "segments": done},
                               ensure_ascii=False, indent=1),
                    "utf-8",
                )
                if push:
                    text_i = f"第 {i + 1}/{total} 段（{self._ts(start)}-{self._ts(end)}）：{summary}"
                    parts = [Plain(text_i)]
                    if sheet.exists():
                        parts.append(AstrImage.fromFileSystem(str(sheet)))
                    try:
                        await asyncio.wait_for(self.context.send_message(job["umo"], MessageChain(parts)), timeout=60)
                    except Exception as exc:
                        logger.warning(f"发送分段失败：{exc}")
                        self._bg_say(job["umo"], text_i)
                if zoom_on and auto_zoom and item.get("cells"):
                    try:
                        z = await self._auto_zoom(job, path, seg_dir, start, plan, cuts,
                                                 i, total, lang, asr_on)
                        if z:
                            k, zinfo = z
                            item.setdefault("zooms", []).append({"cell": k, **zinfo})
                            job["log"].write_text(
                                json.dumps({"file": str(path), "duration": info["duration"],
                                            "segments": done}, ensure_ascii=False, indent=1),
                                "utf-8",
                            )
                            ztext = (
                                f"第 {i + 1} 段第 {k + 1} 格特写"
                                f"（{self._ts(start + zinfo['a'])}-{self._ts(start + zinfo['b'])}）："
                                f"{zinfo['summary']}"
                            )
                            zparts = [Plain(ztext)]
                            if zinfo.get("sheet"):
                                zparts.append(AstrImage.fromFileSystem(zinfo["sheet"]))
                            try:
                                await asyncio.wait_for(self.context.send_message(job["umo"], MessageChain(zparts)), timeout=60)
                            except Exception as exc:
                                logger.warning(f"发送细分失败：{exc}")
                    except Exception as exc:
                        logger.warning(f"细分失败：{exc}")
            overall = await self.final_summary(done)
            text = f"看完了，共 {total} 段。"
            if overall:
                text += "\n\n" + overall
            self._bg_say(job["umo"], text)
        except Exception as exc:
            logger.exception("视频任务失败")
            await self._say(job["umo"], f"看到一半出错了：{exc}")
        finally:
            if self.job:
                self.job["task"] = asyncio.create_task(asyncio.sleep(0))

    def _bg_say(self, umo: str, text: str):
        task = asyncio.create_task(self._say(umo, text))
        self.__dict__.setdefault("_bg_tasks", set()).add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def _say(self, umo: str, text: str):
        try:
            await asyncio.wait_for(
                self.context.send_message(umo, MessageChain([Plain(text)])), timeout=20
            )
        except Exception as exc:
            logger.warning(f"发送消息失败：{exc}")

    # ---------- commands ----------
    @filter.command("kv", priority=12000)
    async def kv(self, event: AstrMessageEvent):
        event.stop_event()
        raw = (event.message_str or "").strip()
        args = raw.split()
        if args and args[0].startswith("/"):
            args = args[1:]
        sub = args[1] if len(args) > 1 else ""
        if not sub or sub in ("help", "?"):
            yield event.plain_result(
                "用法：\n"
                "/kv 文件名.mp4 —— 开始看（放到视频文件夹里）\n"
                "/kv 文件 2 —— 只看前 2 段\n"
                "/kv zoom 3 5 —— 把第 3 段第 5 格再切成 9 小格细看\n"
                "/kv status —— 进度\n/kv stop —— 停\n/kv list —— 文件夹里有什么"
            )
            return
        if sub == "status":
            if not self.job:
                yield event.plain_result("现在没在看看视频。")
                return
            j = self.job
            done = len(j["segments"])
            dur = j.get("duration") or 0
            seg = float(self._c("segment_seconds", 300))
            total = max(1, math.ceil(dur / seg)) if dur else 0
            yield event.plain_result(f"正在看《{j['name']}》：{done}/{total} 段")
            return
        if sub == "stop":
            if self.job:
                self.job["stop"] = True
                yield event.plain_result("好，这一段看完就停。")
            else:
                yield event.plain_result("本来就没在看。")
            return
        if sub == "list":
            files = [p.name for p in sorted(self.watch_dir().iterdir()) if p.is_file()]
            yield event.plain_result("文件夹里：\n" + ("\n".join(files) if files else "（空）"))
            return
        if sub == "zoom":
            if not self.job:
                yield event.plain_result("现在没有正在看或刚看完的视频。")
                return
            segs = self.job.get("segments") or []
            seg_i = int(args[2]) - 1 if len(args) > 2 and args[2].isdigit() else len(segs) - 1
            cell_i = int(args[3]) - 1 if len(args) > 3 and args[3].isdigit() else 0
            if not (0 <= seg_i < len(segs)):
                yield event.plain_result("段号不对，先用 /kv status 看进度。")
                return
            seg = segs[seg_i]
            cells = seg.get("cells") or []
            if not cells:
                dur = float(seg["end"]) - float(seg["start"])
                cells = [[round(k * dur / 9.0, 2), round((k + 1) * dur / 9.0, 2)] for k in range(9)]
            if cell_i < 0 or cell_i >= len(cells):
                yield event.plain_result(f"格号不对，这一段是 {len(cells)} 格。")
                return
            hit = next((z for z in (seg.get("zooms") or []) if z.get("cell") == cell_i and z.get("sheet")), None)
            if hit:
                yield event.plain_result(f"第 {seg_i + 1} 段第 {cell_i + 1} 格：{hit.get('summary', '')}")
                try:
                    await self.context.send_message(
                        event.unified_msg_origin,
                        MessageChain([AstrImage.fromFileSystem(hit["sheet"])]),
                    )
                except Exception as exc:
                    logger.warning(f"发送细分图失败：{exc}")
                return
            yield event.plain_result(f"正在展开第 {seg_i + 1} 段第 {cell_i + 1} 格…")
            a, b = cells[cell_i]
            seg_dir = self.job["cache"] / f"seg{seg_i:03d}"
            seg_len = float(self._c("segment_seconds", 300))
            total = max(1, math.ceil((self.job.get("duration") or 0) / seg_len))
            lang = str(self._c("asr_language", "auto") or "auto")
            asr_on = bool(self._c("asr_enable", True))
            try:
                info = await self.build_zoom(self.job["path"], seg_dir, float(seg["start"]),
                                             a, b, seg_i, cell_i, total, lang, asr_on)
            except Exception as exc:
                yield event.plain_result(f"展开失败：{exc}")
                return
            seg.setdefault("zooms", []).append({"cell": cell_i, **info})
            try:
                self.job["log"].write_text(
                    json.dumps({"file": str(self.job["path"]),
                                "duration": self.job.get("duration") or 0,
                                "segments": segs}, ensure_ascii=False, indent=1),
                    "utf-8",
                )
            except Exception:
                pass
            yield event.plain_result(f"第 {seg_i + 1} 段第 {cell_i + 1} 格：{info['summary']}")
            if info.get("sheet"):
                try:
                    await self.context.send_message(
                        event.unified_msg_origin,
                        MessageChain([AstrImage.fromFileSystem(info["sheet"])]),
                    )
                except Exception as exc:
                    logger.warning(f"发送细分图失败：{exc}")
            return
        limit = 0
        if len(args) > 2 and args[2].isdigit():
            limit = int(args[2])
        try:
            path = self._resolve(sub)
        except Exception as exc:
            yield event.plain_result(str(exc))
            return
        msg = await self.start_job(event, path, limit)
        yield event.plain_result(msg)

    VIDEO_SUFFIXES = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".ts", ".m4v", ".wmv")

    def _search_name(self, name: str):
        if not name:
            return None
        dirs = [
            self._watch_dir(),
            self.base / "data" / "inbox",
        ]
        for d in dirs:
            try:
                if not d.is_dir():
                    continue
                hit = d / name
                if hit.is_file():
                    return hit
                for f in d.rglob(name):
                    if f.is_file():
                        return f
            except Exception:
                continue
        return None

    async def _materialize(self, comp):
        """把 QQ 视频/文件组件变成本地文件；拿不到就返回 None。"""
        inbox = self.base / "data" / "inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        cands = []
        for attr in ("file", "path", "url"):
            v = getattr(comp, attr, "")
            if v:
                cands.append(str(v))
        name = str(getattr(comp, "name", "") or "")
        for s in cands:
            if s.startswith("file://"):
                try:
                    p = Path(urllib.parse.unquote(urllib.parse.urlparse(s).path.lstrip("/")))
                    if p.is_file():
                        return p
                except Exception:
                    pass
            elif s.startswith("http://") or s.startswith("https://"):
                try:
                    dst = inbox / (name or f"qq_{int(time.time())}.mp4")
                    if not dst.suffix:
                        dst = dst.with_suffix(".mp4")
                    urllib.request.urlretrieve(s, dst)
                    if dst.is_file() and dst.stat().st_size > 1024:
                        return dst
                except Exception as exc:
                    logger.warning(f"下载视频失败：{exc}")
            elif s.startswith("base64://"):
                try:
                    import base64 as _b64

                    data = _b64.b64decode(s[len("base64://"):])
                    dst = inbox / (name or f"qq_{int(time.time())}.mp4")
                    dst.write_bytes(data)
                    return dst
                except Exception:
                    pass
            else:
                p = Path(s)
                if p.is_file():
                    return p
                hit = self._search_name(p.name)
                if hit:
                    return hit
        hit = self._search_name(name)
        if hit:
            return hit
        return None

    @filter.event_message_type(filter.EventMessageType.ALL, priority=12000)
    async def on_video(self, event: AstrMessageEvent):
        raw = (event.message_str or "").strip()
        try:
            _n = len(event.get_messages() or [])
        except Exception:
            _n = -1
        plain = raw.lstrip("/").lstrip()
        if plain.lower().startswith("kv"):
            event.stop_event()
            async for res in self.kv(event):
                yield res
            return
        try:
            comps = event.get_messages() or []
        except Exception:
            return
        if not comps:
            return
        self._debug_inbox(comps)
        for comp in comps:
            kind = type(comp).__name__
            if kind not in ("Video", "File"):
                continue
            name = str(getattr(comp, "name", "") or getattr(comp, "file", "") or "")
            if kind == "File" and name and not name.lower().endswith(self.VIDEO_SUFFIXES):
                continue
            path = await self._materialize(comp)
            if path and path.is_file():
                msg = await self.start_job(event, path, 0)
                await self._say(event.unified_msg_origin, msg)
                return
            await self._say(
                event.unified_msg_origin,
                "视频我这边只收到了文件名/链接，没拿到文件本身。"
                "把它丢进视频文件夹，再发 /kv 文件名 就行。",
            )
            return
