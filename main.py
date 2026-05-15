"""
VideoKit Backend — FastAPI + yt-dlp
支援 YouTube、Instagram Reels、一般影片 URL
"""

import os
import uuid
import tempfile
import asyncio
from pathlib import Path
from functools import partial

import yt_dlp
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl


# ── 設定 ──────────────────────────────────────────────────────────────────────

TEMP_DIR = Path(tempfile.gettempdir()) / "videokit"
TEMP_DIR.mkdir(exist_ok=True)

# CORS：開放所有來源（Google Sites / 靜態頁面皆可呼叫）
app = FastAPI(title="VideoKit API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Schema ────────────────────────────────────────────────────────────────────

class VideoInfoRequest(BaseModel):
    url: str


class DownloadRequest(BaseModel):
    url: str
    format: str = "mp4"      # "mp4" | "mp4_720" | "mp4_1080" | "audio"
    quality: str = "best"


# ── 工具函式 ──────────────────────────────────────────────────────────────────

def build_ydl_opts(output_path: str, fmt: str) -> dict:
    """根據格式選擇 yt-dlp 參數"""

    common = {
        "outtmpl": output_path,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }

    format_map = {
        "mp4": {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "merge_output_format": "mp4",
        },
        "mp4_720": {
            "format": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
            "merge_output_format": "mp4",
        },
        "mp4_1080": {
            "format": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
            "merge_output_format": "mp4",
        },
        "audio": {
            "format": "bestaudio/best",
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }],
        },
    }

    return {**common, **format_map.get(fmt, format_map["mp4"])}


def _fetch_info(url: str) -> dict:
    """同步執行，放進 thread pool 避免 block event loop"""
    opts = {"quiet": True, "no_warnings": True, "noplaylist": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return {
            "title": info.get("title", "Untitled"),
            "duration": info.get("duration"),
            "thumbnail": info.get("thumbnail"),
            "uploader": info.get("uploader"),
            "platform": info.get("extractor_key", "").lower(),
            "formats": [
                {"id": f["format_id"], "ext": f.get("ext"), "height": f.get("height")}
                for f in info.get("formats", [])
                if f.get("vcodec") != "none"
            ][-10:],  # 只回傳最後 10 個格式避免 payload 過大
        }


def _download_video(url: str, output_path: str, fmt: str) -> str:
    """同步下載，回傳實際檔案路徑（yt-dlp 可能修改副檔名）"""
    opts = build_ydl_opts(output_path, fmt)
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        # 取得 yt-dlp 實際寫入的路徑
        return ydl.prepare_filename(info)


def cleanup_file(path: str):
    """BackgroundTask：下載完成後刪除暫存檔"""
    try:
        Path(path).unlink(missing_ok=True)
    except Exception:
        pass


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/info")
async def get_video_info(req: VideoInfoRequest):
    """
    取得影片資訊（標題、縮圖、時長）不下載
    前端可用來預覽後再決定是否下載
    """
    try:
        loop = asyncio.get_event_loop()
        info = await loop.run_in_executor(None, partial(_fetch_info, req.url))
        return info
    except yt_dlp.utils.DownloadError as e:
        raise HTTPException(status_code=422, detail=f"無法解析 URL：{str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/download")
async def download_video(req: DownloadRequest, background_tasks: BackgroundTasks):
    """
    下載影片並以 FileResponse 串流回前端
    暫存檔在回應完成後自動清除
    """
    job_id = uuid.uuid4().hex
    ext = "mp3" if req.format == "audio" else "mp4"
    output_template = str(TEMP_DIR / f"{job_id}.%(ext)s")
    output_path = str(TEMP_DIR / f"{job_id}.{ext}")

    try:
        loop = asyncio.get_event_loop()
        actual_path = await loop.run_in_executor(
            None,
            partial(_download_video, req.url, output_template, req.format),
        )
    except yt_dlp.utils.DownloadError as e:
        raise HTTPException(status_code=422, detail=f"下載失敗：{str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # yt-dlp 有時會用 actual_path，有時副檔名不同，找到實際檔案
    if not Path(actual_path).exists():
        # fallback：掃描 temp dir 找 job_id 開頭的檔
        candidates = list(TEMP_DIR.glob(f"{job_id}.*"))
        if not candidates:
            raise HTTPException(status_code=500, detail="下載完成但找不到檔案")
        actual_path = str(candidates[0])

    media_type = "audio/mpeg" if ext == "mp3" else "video/mp4"
    filename = Path(actual_path).name

    # 回應完成後刪除暫存檔
    background_tasks.add_task(cleanup_file, actual_path)

    return FileResponse(
        path=actual_path,
        media_type=media_type,
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
