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
    import numpy as np
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover
    Image = None

SYSTEM_PROMPT = (
    "你是小满，主人的陪伴者。你正和主人一起看视频：画面被拆成一格一格的截图（每格有编号和时间），"
    "声音转成了文字。用第一人称、温柔口语说人话：先说你看到的这一段在发生什么，再点出你最在意的一个细节；"
    "不要罗列每一格，不要机械复述台词，也不要说自己看不到视频——你正在看。"
)

FINAL_PROMPT = (
    "下面是你和主人刚看完的片子，按时间顺序分了段。用第一人称给主人一个整体印象："
    "这是什么内容、结构或剧情怎么走、你最回味的地方。像刚一起看完在聊，不要逐条复述，三到五句。"
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
        sys.stdout.buffer.write((json.dumps({"text": " ".join(texts), "chunks": texts}, ensure_ascii=False) + "\n").encode("utf-8"))
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
        "max_gap_seconds": int,
        "scene_threshold": float,
        "zoom_enable": bool,
        "auto_zoom": bool,
        "zoom_frames": int,
        "asr_enable": bool,
        "asr_language": str,
        "push_progress": bool,
        "push_mode": str,
        "push_segment_detail": bool,
        "memory_write": bool,
        "memory_write_mode": str,
        "subtitle_mode": str,
        "subtitle_enable": bool,
        "ocr_enable": bool,
        "ocr_fps": float,
        "ocr_band": float,
        "double_frame": bool,
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

    PAGE_RANGES = {
        "segment_seconds": (15, 3600),
        "frames_per_segment": (2, 36),
        "max_gap_seconds": (3, 60),
        "zoom_frames": (2, 36),
        "scene_threshold": (0.05, 1.0),
        "ocr_fps": (0.05, 5.0),
        "ocr_band": (0.05, 0.6),
        "max_segments_per_run": (0, 9999),
    }

    def _apply_page_changes(self, payload: dict) -> dict:
        if not isinstance(payload, dict):
            return {"ok": False, "message": "请求体格式不对"}
        changed = {}
        skipped = []
        for key, typ in self.PAGE_KEYS.items():
            if key not in payload:
                continue
            raw = payload.get(key)
            if raw is None or (isinstance(raw, str) and not raw.strip()):
                skipped.append(key)
                continue
            try:
                if typ is bool:
                    val = raw if isinstance(raw, bool) else str(raw).strip().lower() in ("1", "true", "yes", "on")
                elif typ is int:
                    val = int(float(raw))
                elif typ is float:
                    val = float(raw)
                else:
                    val = str(raw).strip()
            except Exception:
                return {"ok": False, "message": f"参数 {key} 的值不合法"}
            rng = self.PAGE_RANGES.get(key)
            if rng and isinstance(val, (int, float)) and not isinstance(val, bool):
                lo, hi = rng
                val = max(lo, min(hi, val))
            changed[key] = val
        if "push_mode" in changed:
            pm = str(changed["push_mode"]).strip().lower()
            if pm in ("off", "summary", "segment"):
                changed["push_progress"] = pm != "off"
                changed["push_segment_detail"] = pm == "segment"
        if not changed:
            return {"ok": False, "message": "没有可保存的参数（可能页面还没读到配置）"}
        try:
            self.config.update(changed)
            save = getattr(self.config, "save_config", None)
            if callable(save):
                save()
        except Exception as exc:
            return {"ok": False, "message": f"保存失败: {exc}"}
        tail = f"（已跳过空值: {', '.join(skipped)}）" if skipped else ""
        return {"ok": True, "message": "已保存，下一段起生效" + tail, "config": self._page_payload()}

    def _watch_dir(self) -> Path:
        configured = str(self._c("watch_dir", "") or "").strip().strip('"')
        path = Path(configured).expanduser() if configured else (self.base / "inbox")
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
        """均匀窗口保证覆盖，窗内取最长镜头的中点（避开转场/模糊）。"""
        n = max(1, int(limit))
        win = dur / n
        cs = sorted(t for t in cuts if 0.6 < t < dur - 0.6)
        bounds = [0.0] + cs + [dur]
        shots = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
        shots = [(a, b) for a, b in shots if b - a >= 0.4]
        out = []
        for i in range(n):
            ws, we = i * win, min(dur, (i + 1) * win)
            cand = []
            for a, b in shots:
                lo, hi = max(a, ws), min(b, we)
                if hi - lo > 0.3:
                    cand.append((hi - lo, lo, hi))
            if cand:
                _, lo, hi = max(cand)
                mid = (lo + hi) / 2.0
            else:
                mid = (ws + we) / 2.0
            out.append((ws, we, mid))
        return out

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

    @staticmethod
    def _pair(left: Path, right: Path, out: Path) -> None:
        """前后两帧左右拼成一张 16:9，铺满格子不留黑边。"""
        try:
            a = Image.open(left).convert("RGB")
            b = Image.open(right).convert("RGB")
            h, gap = 540, 6
            half = (960 - gap) // 2
            def cover(im):
                r = max(half / im.width, h / im.height)
                im = im.resize((max(half, int(im.width * r)), max(h, int(im.height * r))))
                x = (im.width - half) // 2
                y = (im.height - h) // 2
                return im.crop((x, y, x + half, y + h))
            canvas = Image.new("RGB", (half * 2 + gap, h), (14, 18, 26))
            canvas.paste(cover(a), (0, 0))
            canvas.paste(cover(b), (half + gap, 0))
            canvas.save(out, "JPEG", quality=85)
        except Exception:
            try:
                left.replace(out)
            except Exception:
                pass

    @staticmethod
    def _grid_cols(n: int) -> int:
        """按格数选列数：9→3、12→4、16→4，尽量方。"""
        if n <= 4:
            return 2
        if n <= 9:
            return 3
        if n <= 16:
            return 4
        if n <= 25:
            return 5
        if n <= 36:
            return 6
        return 7

    def make_sheet(self, frames: list[tuple[Path, str]], out: Path) -> None:
        cols = self._grid_cols(len(frames))
        cw = min(500, int(1600 / cols))
        ch, bar = int(cw * 0.564), 26
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
        return self.base / "models" / "sensevoice"

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

    def transcribe(self, wav: Path, lang: str) -> tuple[str, list[str]]:
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
        chunks = [re.sub(r"<\|[^|]+\|>", "", str(c or "")).strip() for c in (data.get("chunks") or [])]
        chunks = [c for c in chunks if c]
        return text, chunks

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

    def _ocr(self):
        engine = getattr(self, "_ocr_engine", None)
        if engine is None:
            from rapidocr_onnxruntime import RapidOCR
            engine = RapidOCR()
            self._ocr_engine = engine
        return engine

    async def _has_subs(self, path: Path) -> bool:
        rc, out, _ = await self._run(
            [self.ffprobe, "-v", "error", "-select_streams", "s", "-show_entries",
             "stream=index", "-of", "csv=p=0", str(path)], timeout=120)
        return rc == 0 and bool(out.strip())

    async def _subs_window(self, path: Path, start: float, end: float, work: Path) -> str:
        srt = work / "subs.srt"
        if not srt.exists():
            rc, _, _ = await self._run(
                [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                 "-i", str(path), "-map", "0:s:0?", "-c:s", "srt", str(srt)], timeout=600)
            if rc != 0 or not srt.exists():
                srt.write_text("", "utf-8")
        try:
            raw = srt.read_text("utf-8", "ignore")
        except Exception:
            return ""
        if not raw.strip():
            return ""
        out = []
        for block in raw.replace("\r", "").split("\n\n"):
            lines = [x.strip() for x in block.split("\n") if x.strip()]
            if len(lines) < 3 or "-->" not in lines[1]:
                continue
            a, b = [x.strip() for x in lines[1].split("-->")[:2]]
            def sec(x):
                hms, _, ms = x.replace(",", ".").partition(".")
                h, m, s = [float(v) for v in hms.split(":")]
                return h * 3600 + m * 60 + s + float("0." + (ms or "0"))
            try:
                s0, s1 = sec(a), sec(b)
            except Exception:
                continue
            if s1 < start or s0 > end:
                continue
            out.append(" ".join(lines[2:]))
        return "\n".join(out)[:3000]


    async def _audio_profile(self, path: Path, start: float, dur: float) -> dict:
        """声学预检：静音占比 / 平均音量（不依赖 ASR 模型，独立证据）。"""
        rc, _, err = await self._run(
            [
                self.ffmpeg, "-hide_banner", "-nostats", "-loglevel", "info",
                "-ss", f"{start:.3f}", "-t", f"{max(1.0, dur):.3f}", "-i", str(path),
                "-vn", "-af", "silencedetect=noise=-35dB:d=0.6,volumedetect",
                "-f", "null", "-",
            ],
            timeout=300,
        )
        info = {"ok": rc == 0, "mean_db": None, "speech_ratio": 1.0}
        if rc != 0 or not err:
            return info
        m = re.search(r"mean_volume:\s*(-?[0-9.]+)\s*dB", err)
        if m:
            info["mean_db"] = float(m.group(1))
        sil, cur = 0.0, None
        for line in err.splitlines():
            m = re.search(r"silence_start:\s*(-?[0-9.]+)", line)
            if m:
                cur = max(0.0, float(m.group(1)))
                continue
            m = re.search(r"silence_end:\s*(-?[0-9.]+)", line)
            if m and cur is not None:
                sil += max(0.0, float(m.group(1)) - cur)
                cur = None
        if cur is not None:
            sil += max(0.0, dur - cur)
        info["speech_ratio"] = max(0.0, min(1.0, 1.0 - sil / max(1.0, dur)))
        return info

    async def _ocr_window(self, path: Path, start: float, end: float, work: Path) -> str:
        band = float(self._c("ocr_band", 0.24) or 0.24)
        fps = float(self._c("ocr_fps", 0.4) or 0.4)
        tmp = work / "ocr"
        tmp.mkdir(exist_ok=True)
        for old in tmp.glob("*.jpg"):
            old.unlink(missing_ok=True)
        rc, _, _ = await self._run(
            [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
             "-ss", f"{start:.3f}", "-t", f"{max(1.0, end - start):.3f}", "-i", str(path),
             "-vf", f"fps={fps},crop=iw:trunc(ih*{band:.2f}/2)*2:0:trunc(ih*{max(0.0, 0.96 - band):.2f}/2)*2,scale=iw:ih",
             "-q:v", "4", str(tmp / "c%04d.jpg")], timeout=900)
        if rc != 0:
            return ""

        def kana(x):
            return sum(1 for c in x if 0x3040 <= ord(c) <= 0x30FF)

        def han(x):
            return sum(1 for c in x if 0x4E00 <= ord(c) <= 0x9FFF)

        def run_ocr():
            engine = self._ocr()
            frames = []
            for img in sorted(tmp.glob("*.jpg")):
                try:
                    im = Image.open(img).convert("RGB")
                    if im.height < 180 and im.width < 1600:
                        scale = min(2.0, 2400.0 / max(1, im.width))
                        im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
                    arr = np.asarray(im)
                    res, _ = engine(arr)
                except Exception:
                    continue
                lines = []
                for item in (res or []):
                    try:
                        txt = str(item[1]).strip()
                        score = float(item[2])
                    except Exception:
                        continue
                    if score >= 0.5 and txt:
                        lines.append(txt)
                frames.append(lines)
            return frames

        frames = await asyncio.to_thread(run_ocr)
        import difflib

        def norm(x):
            x = "".join(ch for ch in x if ch.isprintable())
            return x.strip(" \t\"'\u201c\u201d\u2018\u2019\u300c\u300d\u300e\u300f-\u2014\u2026")

        def keep(line):
            if len(line) < 3:
                return False
            han = sum(1 for c in line if 0x4E00 <= ord(c) <= 0x9FFF)
            kana = sum(1 for c in line if 0x3040 <= ord(c) <= 0x30FF)
            if han or kana:
                return True
            # 纯拉丁：过滤 UI/水印类全大写短词
            return len(line) >= 6 and not line.isupper()

        merged = []
        for lines in frames:
            if not lines:
                continue
            cand = [norm(x) for x in lines]
            cand = [x for x in cand if keep(x)]
            if not cand:
                continue
            zh = [x for x in cand if sum(1 for c in x if 0x4E00 <= ord(c) <= 0x9FFF) >= 2]
            pick = max(zh, key=lambda x: sum(1 for c in x if 0x4E00 <= ord(c) <= 0x9FFF)) if zh else max(cand, key=len)
            if any(difflib.SequenceMatcher(None, pick, prev).ratio() >= 0.72 for prev in merged[-3:]):
                continue
            merged.append(pick)
        return "\n".join(merged)[:3000]

    async def collect_text(self, path: Path, start: float, end: float, work: Path,
                           lang: str, asr_on: bool):
        mode = str(self._c("subtitle_mode", "auto") or "auto")
        use_subs = mode != "off" and bool(self._c("subtitle_enable", True))
        use_ocr = mode != "off" and bool(self._c("ocr_enable", True))
        if use_subs:
            if await self._has_subs(path):
                txt = await self._subs_window(path, start, end, work)
                if len(txt.strip()) >= 6:
                    return txt, "subtitle"
            if use_ocr:
                try:
                    txt = await self._ocr_window(path, start, end, work)
                    if len(txt.strip()) >= 6:
                        return txt, "ocr"
                except Exception as exc:
                    logger.warning(f"OCR 失败：{exc}")
        if not asr_on:
            return "", ""
        dur = max(1.0, end - start)
        prof = await self._audio_profile(path, start, dur)
        if prof["speech_ratio"] < 0.12 or (prof["mean_db"] is not None and prof["mean_db"] < -45):
            logger.info(f"音频预检：无有效人声 (mean={prof['mean_db']}dB ratio={prof['speech_ratio']:.2f})，跳过 ASR")
            return "", "none"
        wav = work / "a.wav"
        if await self.extract_audio(path, start, dur, wav):
            try:
                txt, chunks = await asyncio.to_thread(self.transcribe, wav, lang)
                if txt:
                    norm = ["".join(ch for ch in c if ch.isalnum()) for c in chunks]
                    norm = [x for x in norm if x]
                    repeated = len(norm) >= 3 and len(set(norm)) == 1 and len(norm[0]) <= 4
                    cps = len(txt) / dur
                    if repeated or (len(txt) <= 2 and dur > 8):
                        logger.warning(f"ASR 幻觉判定：文本过短或重复（{len(txt)} 字 / {dur:.0f}s，chunks={len(norm)}），丢弃")
                        return "", "none"
                    if cps > 9 or cps < 0.15:
                        logger.warning(f"ASR 可疑：语速 {cps:.2f} 字/秒（时长 {dur:.0f}s），保留但标记")
                    txt = await self.clean_transcript(txt)
                    return txt, "asr"
            except Exception as exc:
                logger.warning(f"转写失败：{exc}")
            finally:
                wav.unlink(missing_ok=True)
        return "", "none"

    @staticmethod
    def _parse_vision(text: str) -> tuple[str, str, list[str], str]:
        title, summary, keywords, detail = "", "", [], ""
        for line in (text or "").splitlines():
            s = line.strip().lstrip("-*# ").strip()
            for label, sep in (("标题", "："), ("摘要", "："), ("关键词", "："), ("细节", "：")):
                if s.startswith(label):
                    body = s.split(sep, 1)[-1].strip() if sep in s else s[len(label):].strip(" ：:")
                    if label == "标题":
                        title = body[:30]
                    elif label == "摘要":
                        summary = body
                    elif label == "关键词":
                        keywords = [k for k in re.split(r"[、,，;；/\s]+", body) if k]
                    else:
                        detail = body
                    break
        if not summary:
            summary = (text or "").strip()
        if not title:
            title = summary.split("。")[0][:14] or "这一段"
        return title, summary, keywords[:6], detail

    def _chapters(self, segments: list[dict]) -> list[dict]:
        out = []
        for seg in segments:
            out.append({
                "i": int(seg.get("index", 0)) + 1,
                "start": float(seg.get("start", 0)),
                "end": float(seg.get("end", 0)),
                "title": seg.get("title") or f"第 {int(seg.get('index', 0)) + 1} 段",
                "keywords": seg.get("keywords") or [],
                "summary": seg.get("summary", ""),
            })
        return out

    # ---------- current (上次看过的片子) ----------
    def _state_path(self) -> Path:
        return self.base / "state.json"

    def _load_state(self) -> dict:
        try:
            return json.loads(self._state_path().read_text("utf-8"))
        except Exception:
            return {}

    def _save_state(self, **kw):
        data = self._load_state()
        data.update(kw)
        try:
            self._state_path().write_text(json.dumps(data, ensure_ascii=False, indent=1), "utf-8")
        except Exception:
            pass

    def _current_log(self):
        if self.job and self.job.get("logdata"):
            return self.job["logdata"]
        cur = str(self._load_state().get("current") or "")
        if cur:
            log = self._log_by_name(cur)
            if log:
                return log
        return self._latest_log()

    def _use_name(self, name: str):
        """切到某个已看过的片子，返回 log 或 None。"""
        log = self._log_by_name(name)
        if log:
            try:
                self._save_state(current=str(self._resolve(name)))
            except Exception:
                pass
        return log

    def _log_by_name(self, name: str):
        """按文件名找已缓存的 log（支持 /kv chapters full4.mp4）。"""
        try:
            path = self._resolve(name)
        except Exception:
            return None
        cd = self._cache_dir(path)
        lp = cd / "log.json"
        if not lp.is_file():
            return None
        try:
            data = json.loads(lp.read_text("utf-8"))
            data["_path"] = str(lp)
            return data
        except Exception:
            return None

    def _latest_log(self):
        """默认取“看得最完整”的那份：段数最多，其次时长最长。"""
        best, best_key = None, None
        try:
            for p in self.cache_root.glob("*/log.json"):
                try:
                    data = json.loads(p.read_text("utf-8"))
                except Exception:
                    continue
                key = (len(data.get("segments") or []), float(data.get("duration") or 0))
                if best_key is None or key > best_key:
                    best, best_key = data, key
                    data["_path"] = str(p)
        except Exception:
            return None
        return best

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
        prompt += (
            "\n请严格按下面四行输出，不要额外解释：\n"
            "标题：10 字以内，这一段在讲什么\n"
            "摘要：2~3 句话概括这一段\n"
            "关键词：3~6 个，用、分隔\n"
            "细节：一句你最在意的细节"
        )
        resp = await provider.text_chat(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            image_urls=[f"data:image/jpeg;base64,{data}"],
        )
        return str(getattr(resp, "completion_text", "") or "").strip()

    async def clean_transcript(self, text: str) -> str:
        """给 ASR 原文补标点、纠同音错字；失败就原样返回。"""
        if not isinstance(text, str):
            logger.warning(f"clean_transcript 收到 {type(text)}: {text!r}"[:300])
            text = str(text)
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
            head = s.get("title") or ""
            lines.append(
                f"[{self._ts(s['start'])}-{self._ts(s['end'])}] {head}：{s.get('summary', '')}"
            )
        resp = await provider.text_chat(
            prompt=FINAL_PROMPT + "\n\n" + "\n".join(lines),
        )
        return str(getattr(resp, "completion_text", "") or "").strip()

    async def _zoom_job(self, job, event, seg, seg_i: int, cell_i: int, seg_dir: Path,
                        a: float, b: float, total: int, lang: str, asr_on: bool) -> None:
        try:
            info = await self.build_zoom(job["path"], seg_dir, float(seg["start"]),
                                         a, b, seg_i, cell_i, total, lang, asr_on)
        except Exception as exc:
            logger.warning(f"细分展开失败：{exc}")
            try:
                await self._send_event(event, MessageChain([Plain(f"展开失败：{exc}")]))
            except Exception:
                pass
            return
        seg.setdefault("zooms", []).append({"cell": cell_i, **info})
        try:
            job["log"].write_text(
                json.dumps({"file": str(job["path"]),
                            "duration": job.get("duration") or 0,
                            "segments": job.get("segments") or []},
                           ensure_ascii=False, indent=1),
                "utf-8",
            )
        except Exception:
            pass
        try:
            await self._send_event(
                event, MessageChain([Plain(f"第 {seg_i + 1} 段第 {cell_i + 1} 格：{info.get('summary', '')}")])
            )
            if info.get("sheet"):
                await self._send_event(event, MessageChain([AstrImage.fromFileSystem(info["sheet"])]))
        except Exception as exc:
            logger.warning(f"发送细分结果失败：{exc}")

    async def build_zoom(self, path, seg_dir: Path, seg_start: float, a: float, b: float,
                         seg_idx: int, cell_idx: int, total: int, lang: str, asr_on: bool) -> dict:
        zdir = seg_dir / f"cell{cell_idx:03d}"
        zdir.mkdir(exist_ok=True)
        dur = max(1.0, b - a)
        cuts = await self.scene_cuts(path, seg_start + a, dur)
        per = int(self._c("zoom_frames", 9) or 9)
        # 短格子不必抽满：最多每 0.5 秒一帧
        per = max(2, min(per, int(dur / 0.5) or 2))
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
                    transcript, _chunks = await asyncio.to_thread(self.transcribe, wav, lang)
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
        cand = self._watch_dir() / name
        if cand.is_file():
            return cand
        raise FileNotFoundError(f"找不到视频：{name}（可以放进 {self._watch_dir()}）")

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
            "stop": False, "segments": done, "event": event,
        }
        self._save_state(current=str(path))
        if done:
            self.job["start_index"] = len(done)
        task = asyncio.create_task(self._worker(limit_segments))
        self.job["task"] = task
        name = path.name
        openers = [
            f"《{name}》是吧，我这就去看——先摸一遍镜头，看完一段跟你说一句。",
            f"好，《{name}》交给我。先扫一遍画面节奏，再一段一段跟你唠。",
            f"《{name}》收到～我先看一眼它多长、镜头怎么切，然后边看边跟你说。",
        ]
        return openers[sum(map(ord, name)) % len(openers)]

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
            per = max(2, min(per, 36))
            gap = int(self._c("max_gap_seconds", 8) or 8)
            gap = max(3, min(60, gap))
            if seg_len / per > gap:
                per = min(36, max(per, int(math.ceil(seg_len / gap))))
            asr_on = bool(self._c("asr_enable", True))
            lang = str(self._c("asr_language", "auto") or "auto")
            push = bool(self._c("push_progress", True))
            mode = str(self._c("push_mode", "") or "").strip().lower()
            if mode not in ("off", "summary", "segment"):
                mode = "segment" if push else ("summary" if bool(self._c("push_segment_detail", True)) else "off")
            push_seg = mode == "segment"
            push_final = mode != "off"
            zoom_on = bool(self._c("zoom_enable", True))
            auto_zoom = bool(self._c("auto_zoom", True))
            done = job["segments"]
            try:
                prev_seg = float(json.loads(job["log"].read_text("utf-8")).get("segment_seconds", 0) or 0)
            except Exception:
                prev_seg = 0.0
            if done and prev_seg > 0 and abs(prev_seg - seg_len) > 0.5:
                done.clear()
                job["segments"] = done
                self._bg_job(job, "分段时长改过了，之前那几段对不上，我重新从头看。")
            start_idx = len(done)
            if start_idx >= total:
                if not push_final:
                    return
                text = "这个我先前已经看过啦，直接跟你说说印象？"
                if done:
                    text += "\n\n" + (await self.final_summary(done))
                self._bg_job(job, text)
                return

            for i in range(start_idx, total):
                if job["stop"]:
                    self._bg_job(job, "好，先看到这。")
                    return
                start = i * seg_len
                end = min(info["duration"], start + seg_len)
                dur = end - start
                seg_dir = job["cache"] / f"seg{i:03d}"
                seg_dir.mkdir(exist_ok=True)
                steps = [f"分镜 {self._ts(start)}-{self._ts(end)}"]
                cuts = await self.scene_cuts(path, start, dur)
                plan = self._plan_times(cuts, dur, per)
                dense = (len(cuts) / max(dur, 1.0)) >= 0.5 and bool(self._c("double_frame", True))
                frames = []
                for k, (a, b, mid) in enumerate(plan):
                    fp = seg_dir / f"f{k:02d}.jpg"
                    if dense:
                        lo = max(a + 0.1, mid - 0.3)
                        hi = min(b - 0.1, mid + 0.3)
                        f1 = seg_dir / f"f{k:02d}a.jpg"
                        f2 = seg_dir / f"f{k:02d}b.jpg"
                        ok1 = await self.grab_frame(path, start + lo, f1)
                        ok2 = await self.grab_frame(path, start + hi, f2)
                        if ok1 and ok2:
                            self._pair(f1, f2, fp)
                            frames.append((fp, f"{k + 1}  {self._ts(start + a)}"))
                            continue
                    ok = await self.grab_frame(path, start + mid, fp)
                    if ok:
                        frames.append((fp, f"{k + 1}  {self._ts(start + a)}"))
                if not frames:
                    frames = []
                sheet = seg_dir / "sheet.jpg"
                transcript, tsrc = await self.collect_text(path, start, end, seg_dir, lang, asr_on)
                timeline = "\n".join(
                    "\u7b2c{0}\u683c\uff1a{1} ~ {2}".format(k + 1, self._ts(start + plan[k][0]), self._ts(start + plan[k][1]))
                    for k in range(len(plan))
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
                title, summary, keywords, detail = self._parse_vision(summary)
                item = {
                    "index": i, "start": start, "end": end,
                    "title": title, "keywords": keywords, "detail": detail,
                    "summary": summary, "transcript": transcript[:3000], "text_source": tsrc,
                    "sheet": str(sheet) if sheet.exists() else "",
                    "cells": [[round(a, 2), round(b, 2)] for a, b, _m in plan],
                }
                done.append(item)
                job["log"].write_text(
                    json.dumps({"file": str(path), "duration": info["duration"], "segment_seconds": seg_len, "segments": done},
                               ensure_ascii=False, indent=1),
                    "utf-8",
                )
                if push_seg:
                    text_i = f"第 {i + 1}/{total} 段（{self._ts(start)}-{self._ts(end)}）：{summary}"
                    parts = [Plain(text_i)]
                    if sheet.exists():
                        parts.append(AstrImage.fromFileSystem(str(sheet)))
                    chain = MessageChain(parts)
                    sent = False
                    umo_i = str(job.get("umo") or "")
                    if umo_i and not umo_i.startswith("webchat"):
                        sent = await self._qq_send(
                            umo_i.split(":")[-1], text_i, str(sheet) if sheet.exists() else "",
                            self_id=getattr(job.get("event"), "self_id", None),
                        )
                    if not sent:
                        try:
                            ev = job.get("event")
                            if ev is not None and hasattr(ev, "send"):
                                await asyncio.wait_for(ev.send(chain), timeout=60)
                                self._log("push_event_ok", f"seg={i + 1}")
                                sent = True
                        except Exception as exc:
                            self._log("push_event_fail", f"{type(exc).__name__}: {exc}")
                    if not sent:
                        try:
                            await asyncio.wait_for(self.context.send_message(job["umo"], chain), timeout=60)
                            self._log("push_umo_ok", f"seg={i + 1}")
                        except Exception as exc:
                            self._log("push_umo_fail", f"{type(exc).__name__}: {exc}")
                            logger.warning(f"发送分段失败：{exc}")
                            self._bg_job(job, text_i)
                if str(self._c("memory_write_mode", "both")) in ("both", "segment"):
                    vid = self._vid(path)
                    await self._remember_experience(
                        job,
                        f"和主人一起看了《{path.stem}》第 {i + 1}/{total} 段（{self._ts(start)}-{self._ts(end)}）：{summary}",
                        f"watch:{vid}:{i}",
                        0.55,
                        {"video_title": path.stem, "segment": i + 1,
                         "start": start, "end": end, "keywords": keywords},
                    )
                if push_seg and zoom_on and auto_zoom and item.get("cells"):
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
                            zchain = MessageChain(zparts)
                            ev = job.get("event")
                            if ev is not None and hasattr(ev, "send"):
                                if not await self._send_event(ev, zchain):
                                    self._bg_job(job, ztext)
                            else:
                                try:
                                    await asyncio.wait_for(self.context.send_message(job["umo"], zchain), timeout=60)
                                except Exception as exc:
                                    logger.warning(f"发送细分失败：{exc}")
                    except Exception as exc:
                        logger.warning(f"细分失败：{exc}")
            overall = await self.final_summary(done)
            if str(self._c("memory_write_mode", "both")) in ("both", "final"):
                vid = self._vid(path)
                await self._remember_experience(
                    job,
                    f"和主人一起看完了《{path.stem}》（{self._ts(info['duration'])}，{total} 段）：{overall}",
                    f"watch:{vid}:final",
                    0.8,
                    {"video_title": path.stem, "segments": total},
                )
            chapters = self._chapters(done)
            try:
                job["log"].write_text(
                    json.dumps({"file": str(path), "duration": info["duration"], "segment_seconds": seg_len,
                                "segments": done,
                                "film": {"overview": overall, "chapters": chapters}},
                               ensure_ascii=False, indent=1),
                    "utf-8",
                )
            except Exception:
                pass
            if not push_final:
                return
            text = f"看完啦，一共 {total} 段。"
            if overall:
                text += "\n\n" + overall
            self._bg_job(job, text)
        except Exception as exc:
            self._log("worker_crash", f"{type(exc).__name__}: {exc}")
            logger.exception("视频任务失败")
            await self._say(job["umo"], f"看到一半出错了：{exc}")
        finally:
            if self.job:
                self.job["task"] = asyncio.create_task(asyncio.sleep(0))

    async def _qq_send(self, uid: str, text: str, image: str = "", self_id=None) -> bool:
        """直接用 aiocqhttp 的 CQHttp 客户端发私聊，绕开 umo/事件对象。"""
        try:
            plat = self.context.get_platform("aiocqhttp")
        except Exception as exc:
            self._log("qq_send_plat_fail", f"{type(exc).__name__}: {exc}")
            return False
        if plat is None:
            self._log("qq_send_plat_none", "aiocqhttp 平台未加载")
            return False
        try:
            if not str(uid).isdigit():
                self._log("qq_send_bad_uid", str(uid)[:40])
                return False
            cli = plat.get_client()
            msg = []
            if text:
                msg.append({"type": "text", "data": {"text": text}})
            if image:
                msg.append({"type": "image", "data": {"file": Path(image).resolve().as_uri()}})
            if not msg:
                return False
            kwargs = {"user_id": int(uid), "message": msg}
            if self_id:
                kwargs["self_id"] = int(self_id)
            await asyncio.wait_for(cli.send_private_msg(**kwargs), timeout=60)
            self._log("qq_send_ok", f"uid={uid} img={'y' if image else 'n'}")
            return True
        except Exception as exc:
            self._log("qq_send_fail", f"{type(exc).__name__}: {exc}")
            logger.warning(f"CQHttp 直发失败：{exc}")
            return False

    async def _send_event(self, event, chain):
        try:
            await asyncio.wait_for(event.send(chain), timeout=60)
            return True
        except Exception as exc:
            self._log("event_send_fail", f"{type(exc).__name__}: {exc}")
            logger.warning(f"event.send 失败：{exc}")
            return False

    def _bg_job(self, job, text: str, chain=None):
        async def run():
            umo = str(job.get("umo") or "")
            if umo and not umo.startswith("webchat"):
                if await self._qq_send(umo.split(":")[-1], text or "",
                                        self_id=getattr(job.get("event"), "self_id", None)):
                    return
            try:
                ev = job.get("event")
                if ev is not None and hasattr(ev, "send"):
                    await asyncio.wait_for(ev.send(chain or MessageChain([Plain(text)])), timeout=60)
                    self._log("push_event_ok", (text or "")[:20])
                    return
            except Exception as exc:
                self._log("push_event_fail", f"{type(exc).__name__}: {exc}")
            try:
                await asyncio.wait_for(
                    self.context.send_message(job["umo"], chain or MessageChain([Plain(text)])), timeout=60
                )
                self._log("push_umo_ok", (text or "")[:20])
            except Exception as exc:
                self._log("push_umo_fail", f"{type(exc).__name__}: {exc}")

        task = asyncio.create_task(run())
        self.__dict__.setdefault("_bg_tasks", set()).add(task)

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
        self._dbg("kv", event)
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
                "/kv chapters —— 列上次看的那部片的章节目录\n"
                "/kv chapters 文件名.mp4 —— 换成另一部已看过的片子\n"
                "/kv use 文件名.mp4 —— 切换当前片子\n"
                "/kv jump 90:00 或 /kv jump 反转 —— 跳到某一时刻/某个话题，回看那一段\n"
                "/kv ask 这电影讲了什么 —— 基于已看的段落回答问题\n"
                "/kv status —— 进度\n/kv stop —— 停\n/kv list —— 文件夹里有什么"
            )
            return
        if sub in {"ocr", "subs"}:
            val = args[2].lower() if len(args) > 2 else ""
            key = "ocr_enable" if sub == "ocr" else "subtitle_enable"
            if val in {"on", "1", "true"}:
                self._apply_page_changes({key: True})
            elif val in {"off", "0", "false"}:
                self._apply_page_changes({key: False})
            state = "\u5f00" if bool(self._c(key, True)) else "\u5173"
            label = "OCR \u753b\u9762\u5b57\u5e55" if sub == "ocr" else "\u5185\u5d4c\u5b57\u5e55"
            yield event.plain_result(f"{label}\uff1a{state}\uff08/kv {sub} on|off \u5207\u6362\uff0c\u4e0b\u4e00\u6bb5\u751f\u6548\uff09")
            return

        if sub in ("chapters", "chapter", "ch", "toc"):
            name = next((a for a in args[2:] if "." in a), "")
            log = self._use_name(name) if name else self._current_log()
            if not log:
                yield event.plain_result("还没有看完的视频。先用 /kv 文件名 看一个。")
                return
            chapters = self._chapters(log.get("segments") or [])
            if not chapters:
                yield event.plain_result("这个视频还没有分段记录。")
                return
            lines = [f"{c['i']}. {self._ts(c['start'])}-{self._ts(c['end'])}  {c['title']}"
                     for c in chapters]
            title = Path(str(log.get("file") or "")).name or "当前片子"
            yield event.plain_result(f"章节目录（{title}）：\n" + "\n".join(lines))
            return
        if sub == "jump":
            arg = " ".join(a for a in args[2:] if "." not in a).strip()
            if not arg:
                yield event.plain_result("用法：/kv jump 90:00 或 /kv jump 反转")
                return
            name = next((a for a in args[2:] if "." in a), "")
            log = self._use_name(name) if name else self._current_log()
            if not log:
                yield event.plain_result("还没有看完的视频。")
                return
            seg = self._pick_segment(log, arg)
            if not seg:
                yield event.plain_result("没找到那一段，试试 /kv chapters 看目录。")
                return
            yield event.plain_result(
                f"第 {int(seg.get('index', 0)) + 1} 段（{self._ts(float(seg['start']))}-{self._ts(float(seg['end']))}）"
                f" {seg.get('title', '')}：{seg.get('summary', '')}"
            )
            sheet = seg.get("sheet")
            if sheet and Path(sheet).is_file():
                try:
                    await self._send_event(event, MessageChain([AstrImage.fromFileSystem(sheet)]))
                except Exception as exc:
                    logger.warning(f"发送段落图失败：{exc}")
            return
        if sub == "ask":
            question = " ".join(a for a in args[2:] if "." not in a).strip()
            if not question:
                yield event.plain_result("用法：/kv ask 这电影讲了什么")
                return
            name = next((a for a in args[2:] if "." in a), "")
            log = self._use_name(name) if name else self._current_log()
            if not log:
                yield event.plain_result("还没有看完的视频。先用 /kv 看一个。")
                return
            segs = log.get("segments") or []
            if not segs:
                yield event.plain_result("这个视频还没有分段记录。")
                return
            picked = self._retrieve(segs, question)
            provider = self.context.get_using_provider()
            if provider is None:
                yield event.plain_result("没有可用的模型。")
                return
            blocks = []
            for s in picked:
                blocks.append(
                    f"[{self._ts(float(s['start']))}-{self._ts(float(s['end']))}] "
                    f"{s.get('title', '')}：{s.get('summary', '')}\n"
                    f"台词节选：{(s.get('transcript') or '')[:600]}"
                )
            prompt = (
                "下面是机器人之前分段看过的视频内容（带时间）。请只根据这些内容回答用户的问题，"
                "回答里尽量带上时间点；不确定就直说没看到。\n\n"
                + "\n\n".join(blocks)
                + f"\n\n用户的问题：{question}"
            )
            try:
                resp = await provider.text_chat(prompt=prompt)
                answer = str(getattr(resp, "completion_text", "") or "").strip()
            except Exception as exc:
                yield event.plain_result(f"回答失败：{exc}")
                return
            yield event.plain_result(answer or "没整理出答案。")
            return
        if sub in ("ping", "test"):
            uid = ((args[2] if len(args) > 2 else "").strip()
                   or str(self._c("admin_qq", "") or "")
                   or (event.unified_msg_origin or "").split(":")[-1])
            if not uid:
                yield event.plain_result("用法：/kv ping 342202362")
                return
            ok = await self._qq_send(uid, "推送自测：这条能收到，进度和图就能收到。",
                                     self_id=getattr(event, "self_id", None))
            yield event.plain_result("已发送，看看 QQ 收到没有。" if ok else "发送失败，看日志 qq_send_fail。")
            return
        if sub in ("use", "switch"):
            name = " ".join(args[2:]).strip()
            if not name:
                yield event.plain_result("用法：/kv use 文件名.mp4")
                return
            log = self._use_name(name)
            if not log:
                yield event.plain_result(f"没找到《{name}》的观看记录，先用 /kv {name} 看一遍。")
                return
            yield event.plain_result(f"好，当前片子切成《{name}》了。")
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
            files = [p.name for p in sorted(self._watch_dir().iterdir()) if p.is_file()]
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
                    await self._send_event(event, MessageChain([AstrImage.fromFileSystem(hit["sheet"])]))
                except Exception as exc:
                    logger.warning(f"发送细分图失败：{exc}")
                return
            yield event.plain_result(
                f"我凑近把第 {seg_i + 1} 段第 {cell_i + 1} 格再看仔细一点，等一下下。"
            )
            a, b = cells[cell_i]
            seg_dir = self.job["cache"] / f"seg{seg_i:03d}"
            seg_len = float(self._c("segment_seconds", 300))
            total = max(1, math.ceil((self.job.get("duration") or 0) / seg_len))
            lang = str(self._c("asr_language", "auto") or "auto")
            asr_on = bool(self._c("asr_enable", True))
            asyncio.create_task(
                self._zoom_job(self.job, event, seg, seg_i, cell_i, seg_dir, a, b, total, lang, asr_on)
            )
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

    def _pick_segment(self, log: dict, arg: str):
        segs = log.get("segments") or []
        seconds = None
        m = re.search(r"(\d{1,3}):(\d{2})(?::(\d{2}))?", arg)
        if m:
            if m.group(3) is not None:
                seconds = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
            else:
                seconds = int(m.group(1)) * 60 + int(m.group(2))
        else:
            m2 = re.search(r"(\d+)\s*(?:分|分钟)", arg)
            if m2:
                seconds = int(m2.group(1)) * 60
        if seconds is not None:
            for s in segs:
                if float(s.get("start", 0)) <= seconds < float(s.get("end", 0)):
                    return s
            return None
        return self._retrieve(segs, arg, limit=1)[0] if self._retrieve(segs, arg, limit=1) else None

    def _retrieve(self, segs: list[dict], query: str, limit: int = 3) -> list[dict]:
        q = (query or "").lower()
        tokens = [w for w in re.split(r"[\s，。？?！!、,;；:：]+", q) if len(w) >= 2]
        scored = []
        for s in segs:
            text = " ".join([
                str(s.get("title", "")), str(s.get("summary", "")),
                " ".join(s.get("keywords") or []), str(s.get("transcript", ""))[:1500],
            ]).lower()
            score = sum(text.count(w) for w in tokens) if tokens else 0
            if q and q in text:
                score += 3
            scored.append((score, s))
        scored.sort(key=lambda x: (-x[0], float(x[1].get("start", 0))))
        picked = [s for sc, s in scored[:limit] if sc > 0]
        if not picked:
            picked = [s for _sc, s in scored[:min(2, len(scored))]]
        return picked

    VIDEO_SUFFIXES = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".ts", ".m4v", ".wmv")

    def _search_name(self, name: str):
        if not name:
            return None
        dirs = [
            self._watch_dir(),
            self.base / "inbox",
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
        inbox = self.base / "inbox"
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

    def _log(self, tag: str, msg: str) -> None:
        try:
            with open(self.base / "debug_events.log", "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%H:%M:%S')} {tag} {msg}\n")
        except Exception:
            pass

    def _dbg(self, tag: str, event) -> None:
        try:
            comps = [type(c).__name__ for c in (event.message_obj.message or [])]
            line = f"{time.strftime('%H:%M:%S')} {tag} str={event.message_str!r} comps={comps}\n"
            with open(self.base / "debug_events.log", "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass

    @staticmethod
    def _vid(path) -> str:
        try:
            path = Path(path)
            st = path.stat()
            return hashlib.sha1(f"{path.name}:{st.st_size}".encode("utf-8")).hexdigest()[:16]
        except Exception:
            return str(path)

    def _memory_bridge(self):
        try:
            import sys
            for name, mod in list(sys.modules.items()):
                if name.endswith("astrbot_plugin_memory_companion.main"):
                    getter = getattr(mod, "get_memory_companion_bridge", None)
                    if callable(getter):
                        return getter()
        except Exception:
            return None
        return None

    async def _remember_experience(self, job, content: str, memory_id: str,
                                   importance: float, meta: dict) -> None:
        if not bool(self._c("memory_write", True)):
            return
        bridge = self._memory_bridge()
        if bridge is None:
            self._log("memory_skip", "bridge_missing")
            return
        ev = job.get("event")
        try:
            uid = str(ev.get_sender_id()) if ev is not None and hasattr(ev, "get_sender_id") else ""
            uname = str(ev.get_sender_name()) if ev is not None and hasattr(ev, "get_sender_name") else ""
            await bridge.record_shared_experience(
                content=content,
                experience_type="watch",
                bot_id=str(getattr(ev, "self_id", "") or ""),
                bot_name=str(self._c("bot_name", "小满")),
                user_id=uid,
                user_name=uname,
                scope="private",
                session_id=str(job.get("umo") or ""),
                platform="aiocqhttp",
                source_plugin="astrbot_plugin_video_storyboard",
                memory_id=memory_id,
                confidence=0.88,
                importance=importance,
                metadata=meta,
            )
            self._log("memory_ok", memory_id)
        except Exception as exc:
            self._log("memory_fail", f"{type(exc).__name__}: {exc}")

    @filter.llm_tool(name="watch_video")
    async def tool_watch_video(self, event: AstrMessageEvent, filename: str, segments: int = 0):
        """让机器人自己看一个本地视频：分段抽帧、读字幕/听声音，然后逐段讲给主人听。

        Args:
            filename(string): 视频文件名，放在视频文件夹里的那个名字，也可以给完整路径
            segments(number): 最多看几段，0 表示整部都看
        """
        try:
            path = self._resolve(filename)
        except Exception as exc:
            return f"没找到视频 {filename}：{exc}"
        if not path or not path.is_file():
            return f"没找到视频 {filename}，让他先把文件放进视频文件夹。"
        return await self.start_job(event, path, int(segments or 0))

    @filter.llm_tool(name="video_chapters")
    async def tool_video_chapters(self, event: AstrMessageEvent):
        """列出一部已经看过的视频的章节（每段的时间范围和小标题）。

        不带参数时用最近看过的那部。
        """
        log = self._latest_log()
        if not log:
            return "还没有看完过的视频，先让我看一部吧。"
        segs = log.get("segments") or []
        lines = [f"《{log.get('file') or '视频'}》"]
        for i, s in enumerate(segs):
            lines.append(f"{i + 1}. {s.get('start_str', '')}-{s.get('end_str', '')} {s.get('title') or ''}")
        return "\n".join(lines)

    @filter.llm_tool(name="video_ask")
    async def tool_video_ask(self, event: AstrMessageEvent, question: str):
        """就一部已经看过的视频回答问题（比如“刚才那集讲了什么”“那个反转在第几分钟”）。

        Args:
            question(string): 想问关于这部视频的问题
        """
        log = self._latest_log()
        if not log:
            return "还没有看完过的视频，先让我看一部吧。"
        segs = log.get("segments") or []
        picks = self._retrieve(segs, question, limit=3)
        if not picks:
            return "这部片子里我没找到相关的内容。"
        parts = []
        for s in picks:
            parts.append(
                f"[{s.get('start_str', '')}-{s.get('end_str', '')}] {s.get('title') or ''}："
                f"{s.get('summary') or ''}\n台词节选：{(s.get('transcript') or '')[:200]}"
            )
        return "\n\n".join(parts)

    async def terminate(self):
        if self.job:
            self.job["stop"] = True
            logger.info("video storyboard: job marked stop before unload")

    @filter.event_message_type(filter.EventMessageType.ALL, priority=12000)
    async def on_video(self, event: AstrMessageEvent):
        self._dbg("on_video", event)
        raw = (event.message_str or "").strip()
        try:
            _n = len(event.get_messages() or [])
        except Exception:
            _n = -1
        try:
            comps = event.get_messages() or []
        except Exception:
            return
        if not comps:
            return
        for comp in comps:
            kind = type(comp).__name__
            if kind not in ("Video", "File"):
                continue
            name = str(getattr(comp, "name", "") or getattr(comp, "file", "") or "")
            if kind == "File" and name and not name.lower().endswith(self.VIDEO_SUFFIXES):
                continue
            path = await self._materialize(comp)
            if path and path.is_file():
                event.stop_event()
                msg = await self.start_job(event, path, 0)
                await self._say(event.unified_msg_origin, msg)
                return
            await self._say(
                event.unified_msg_origin,
                "视频我这边只收到了文件名/链接，没拿到文件本身。"
                "把它丢进视频文件夹，再发 /kv 文件名 就行。",
            )
            return
