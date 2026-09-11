from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
import unicodedata
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.asr import FunASRTranscriber
from src.formatter import build_document, write_json, write_markdown
from src.utils import SUPPORTED_AUDIO_EXTENSIONS


ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = ROOT / "input" / "web"
OUTPUT_DIR = ROOT / "output" / "web"
FRONTEND_DIST = ROOT / "frontend" / "dist"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class JobView(BaseModel):
    id: str
    status: Literal["queued", "loading_models", "transcribing", "formatting", "completed", "failed"]
    filename: str
    created_at: float
    elapsed_seconds: float
    phase: str
    progress_current: int = 0
    progress_total: int = 0
    progress_percent: float | None = None
    estimated_remaining_seconds: float | None = None
    device: str | None = None
    error: str | None = None
    segments: list[dict[str, Any]] | None = None
    original_duration_seconds: float | None = None
    trim_start_seconds: float | None = None
    trim_end_seconds: float | None = None


class JobSummary(BaseModel):
    id: str
    status: Literal["queued", "loading_models", "transcribing", "formatting", "completed", "failed"]
    filename: str
    created_at: float
    segment_count: int = 0
    original_duration_seconds: float | None = None
    trim_start_seconds: float | None = None
    trim_end_seconds: float | None = None


class RenameJobRequest(BaseModel):
    name: str


app = FastAPI(title="Interview Transcriber", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_model_lock = threading.Lock()
_transcriber: FunASRTranscriber | None = None
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="funasr")


def _metadata_path(job_id: str) -> Path:
    return OUTPUT_DIR / job_id / "job.json"


def _normalize_record_name(requested_name: str, current_filename: str) -> tuple[str, str]:
    name = unicodedata.normalize("NFC", requested_name).strip()
    current_extension = Path(current_filename).suffix
    if current_extension and name.lower().endswith(current_extension.lower()):
        name = name[: -len(current_extension)].rstrip()
    invalid_characters = '<>:"/\\|?*\0'
    if not name or name in {".", ".."}:
        raise ValueError("记录名称不能为空")
    if len(name) > 120:
        raise ValueError("记录名称不能超过 120 个字符")
    if any(character in invalid_characters or ord(character) < 32 for character in name):
        raise ValueError("记录名称不能包含路径符号或特殊字符")
    if name.endswith((".", " ")):
        raise ValueError("记录名称不能以句点或空格结尾")
    return name, f"{name}{current_extension}"


def _persist_job_locked(job_id: str) -> None:
    job = _jobs[job_id]
    metadata_path = _metadata_path(job_id)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in job.items()
        if key != "processing_started_at"
    }
    temporary_path = metadata_path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_path.replace(metadata_path)


def _update_job(job_id: str, **values: Any) -> None:
    with _jobs_lock:
        job = _jobs[job_id]
        job.update(values, updated_at=time.time())
        if values.get("status") in {"completed", "failed"}:
            job["elapsed_seconds"] = max(0.0, time.time() - job["created_at"])
        _persist_job_locked(job_id)


def _update_progress(job_id: str, phase: str, current: int, total: int) -> None:
    now = time.monotonic()
    with _jobs_lock:
        job = _jobs[job_id]
        if phase == "transcribing" and total > 0 and job.get("processing_started_at") is None:
            job["processing_started_at"] = now

        percent = round(current / total * 100, 1) if total > 0 else None
        estimated_remaining = None
        started_at = job.get("processing_started_at")
        if phase == "transcribing" and started_at is not None and current > 0 and current < total:
            seconds_per_segment = (now - started_at) / current
            estimated_remaining = round(seconds_per_segment * (total - current), 1)

        job.update(
            phase=phase,
            progress_current=current,
            progress_total=total,
            progress_percent=percent,
            estimated_remaining_seconds=estimated_remaining,
        )


def _get_transcriber() -> FunASRTranscriber:
    global _transcriber
    with _model_lock:
        if _transcriber is None:
            _transcriber = FunASRTranscriber(
                ROOT / "config" / "hotwords.txt",
                device="auto",
                speaker_mode="vad_segment",
                max_segment_seconds=6,
            )
            _transcriber._load_model()
        return _transcriber


def _probe_audio_duration(audio_path: Path) -> float:
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        duration = float(completed.stdout.strip())
    except (FileNotFoundError, subprocess.SubprocessError, ValueError) as error:
        raise ValueError("无法读取录音时长，请确认已安装 FFmpeg 且录音文件未损坏") from error
    if duration <= 0:
        raise ValueError("录音时长无效")
    return round(duration, 3)


def _validate_trim_range(
    duration: float,
    requested_start: float | None,
    requested_end: float | None,
) -> tuple[float, float]:
    start = 0.0 if requested_start is None else requested_start
    end = duration if requested_end is None else requested_end
    if start < 0:
        raise ValueError("裁剪开始时间不能小于 0")
    if end > duration + 0.25:
        raise ValueError("裁剪结束时间不能超过录音时长")
    end = min(end, duration)
    if end - start < 1:
        raise ValueError("裁剪后至少需要保留 1 秒录音")
    return round(start, 3), round(end, 3)


def _trim_audio(job_id: str, source: Path, start: float, end: float) -> Path:
    destination = UPLOAD_DIR / f"{job_id}-trimmed.wav"
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-ss",
                str(start),
                "-t",
                str(end - start),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(destination),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except FileNotFoundError as error:
        destination.unlink(missing_ok=True)
        raise RuntimeError("找不到 FFmpeg，请先运行 brew install ffmpeg") from error
    except (subprocess.SubprocessError, OSError) as error:
        destination.unlink(missing_ok=True)
        detail = getattr(error, "stderr", "") or ""
        raise RuntimeError(f"裁剪录音失败：{detail[-300:].strip() or '请检查音频文件'}") from error
    return destination


def _run_transcription(job_id: str) -> None:
    job = _jobs[job_id]
    try:
        if job.get("trim_requested"):
            _update_job(job_id, status="loading_models", phase="preparing_audio")
            source_path = Path(job["audio_path"])
            trimmed_path = _trim_audio(
                job_id,
                source_path,
                job["trim_start_seconds"],
                job["trim_end_seconds"],
            )
            source_path.unlink(missing_ok=True)
            _update_job(
                job_id,
                audio_path=trimmed_path,
                audio_download_name=f"{Path(job['filename']).stem}-trimmed.wav",
            )
            job = _jobs[job_id]
        _update_job(job_id, status="loading_models", phase="loading_models")
        transcriber = _get_transcriber()
        _update_job(job_id, status="transcribing", phase="vad", device=transcriber.device)
        segments = transcriber.transcribe(
            job["audio_path"],
            progress_callback=lambda phase, current, total: _update_progress(
                job_id, phase, current, total
            ),
        )
        _update_job(
            job_id,
            status="formatting",
            phase="formatting",
            estimated_remaining_seconds=None,
        )

        job_output = OUTPUT_DIR / job_id
        job_output.mkdir(parents=True, exist_ok=True)
        # Keep the user's original filename in exported documents rather than
        # exposing the UUID used for safe local storage.
        document = build_document(Path(job["filename"]), segments)
        json_path = job_output / f"{Path(job['filename']).stem}.json"
        markdown_path = job_output / f"{Path(job['filename']).stem}.md"
        write_json(document, json_path)
        write_markdown(document, markdown_path)
        _update_job(
            job_id,
            status="completed",
            phase="completed",
            progress_percent=100.0,
            estimated_remaining_seconds=0.0,
            segments=segments,
            json_path=json_path,
            markdown_path=markdown_path,
        )
    except Exception as error:
        _update_job(
            job_id,
            status="failed",
            phase="failed",
            estimated_remaining_seconds=None,
            error=str(error) or type(error).__name__,
        )


def _job_view(job_id: str) -> JobView:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="转写任务不存在")
        elapsed = job.get("elapsed_seconds")
        if elapsed is None:
            elapsed = time.time() - job["created_at"]
        return JobView(
            id=job_id,
            status=job["status"],
            filename=job["filename"],
            created_at=job["created_at"],
            elapsed_seconds=round(elapsed, 1),
            phase=job.get("phase", job["status"]),
            progress_current=job.get("progress_current", 0),
            progress_total=job.get("progress_total", 0),
            progress_percent=job.get("progress_percent"),
            estimated_remaining_seconds=job.get("estimated_remaining_seconds"),
            device=job.get("device"),
            error=job.get("error"),
            segments=job.get("segments"),
            original_duration_seconds=job.get("original_duration_seconds"),
            trim_start_seconds=job.get("trim_start_seconds"),
            trim_end_seconds=job.get("trim_end_seconds"),
        )


def _job_summary_locked(job_id: str) -> JobSummary:
    job = _jobs[job_id]
    return JobSummary(
        id=job_id,
        status=job["status"],
        filename=job["filename"],
        created_at=job["created_at"],
        segment_count=len(job.get("segments") or []),
        original_duration_seconds=job.get("original_duration_seconds"),
        trim_start_seconds=job.get("trim_start_seconds"),
        trim_end_seconds=job.get("trim_end_seconds"),
    )


def _rename_completed_job_locked(job_id: str, requested_name: str) -> JobSummary:
    job = _jobs[job_id]
    base_name, new_filename = _normalize_record_name(requested_name, job["filename"])
    if new_filename == job["filename"]:
        return _job_summary_locked(job_id)

    old_audio_path = Path(job["audio_path"])
    job_output = OUTPUT_DIR / job_id
    old_json_path = Path(job.get("json_path") or job_output / f"{Path(job['filename']).stem}.json")
    old_markdown_path = Path(
        job.get("markdown_path") or job_output / f"{Path(job['filename']).stem}.md"
    )
    if old_audio_path.parent.resolve() != UPLOAD_DIR.resolve():
        raise OSError("录音文件不在任务存储目录中")
    if old_json_path.parent.resolve() != job_output.resolve():
        raise OSError("JSON 文件不在任务存储目录中")
    if old_markdown_path.parent.resolve() != job_output.resolve():
        raise OSError("Markdown 文件不在任务存储目录中")
    audio_target = UPLOAD_DIR / f"{job_id}-{base_name}{old_audio_path.suffix.lower()}"
    json_target = old_json_path.parent / f"{base_name}.json"
    markdown_target = old_markdown_path.parent / f"{base_name}.md"

    for target, current in (
        (audio_target, old_audio_path),
        (json_target, old_json_path),
        (markdown_target, old_markdown_path),
    ):
        if target != current and target.exists():
            raise FileExistsError(f"目标文件已存在：{target.name}")

    previous_job = dict(job)
    audio_moved = False
    try:
        document = build_document(Path(new_filename), job.get("segments") or [])
        write_json(document, json_target)
        write_markdown(document, markdown_target)
        old_audio_path.replace(audio_target)
        audio_moved = True
        job.update(
            filename=new_filename,
            audio_path=audio_target,
            audio_download_name=f"{base_name}{audio_target.suffix.lower()}",
            json_path=json_target,
            markdown_path=markdown_target,
            updated_at=time.time(),
        )
        _persist_job_locked(job_id)
    except Exception:
        job.clear()
        job.update(previous_job)
        if audio_moved and audio_target.exists() and not old_audio_path.exists():
            audio_target.replace(old_audio_path)
        if json_target != old_json_path:
            json_target.unlink(missing_ok=True)
        if markdown_target != old_markdown_path:
            markdown_target.unlink(missing_ok=True)
        raise

    if old_json_path != json_target:
        old_json_path.unlink(missing_ok=True)
    if old_markdown_path != markdown_target:
        old_markdown_path.unlink(missing_ok=True)
    return _job_summary_locked(job_id)


def _load_persisted_jobs() -> None:
    """Restore saved jobs and migrate result folders created before history existed."""
    for job_output in OUTPUT_DIR.iterdir():
        if not job_output.is_dir() or len(job_output.name) != 32:
            continue
        job_id = job_output.name
        metadata_path = _metadata_path(job_id)
        try:
            if metadata_path.exists():
                job = json.loads(metadata_path.read_text(encoding="utf-8"))
                for key in ("audio_path", "json_path", "markdown_path"):
                    if job.get(key):
                        job[key] = Path(job[key])
            else:
                json_paths = [path for path in job_output.glob("*.json") if path.name != "job.json"]
                if not json_paths:
                    continue
                json_path = json_paths[0]
                document = json.loads(json_path.read_text(encoding="utf-8"))
                audio_paths = list(UPLOAD_DIR.glob(f"{job_id}.*"))
                if not audio_paths:
                    continue
                audio_path = audio_paths[0]
                markdown_paths = list(job_output.glob("*.md"))
                created_at = min(audio_path.stat().st_mtime, json_path.stat().st_mtime)
                job = {
                    "id": job_id,
                    "status": "completed",
                    "phase": "completed",
                    "filename": document.get("source") or audio_path.name,
                    "audio_path": audio_path,
                    "created_at": created_at,
                    "updated_at": json_path.stat().st_mtime,
                    "elapsed_seconds": 0.0,
                    "progress_percent": 100.0,
                    "estimated_remaining_seconds": 0.0,
                    "segments": document.get("segments") or [],
                    "json_path": json_path,
                    "markdown_path": markdown_paths[0] if markdown_paths else None,
                }

            if job.get("status") not in {"completed", "failed"}:
                job.update(
                    status="failed",
                    phase="failed",
                    error="服务曾在任务完成前退出，请重新提交录音。",
                    elapsed_seconds=job.get("elapsed_seconds", 0.0),
                )
            _jobs[job_id] = job
            _persist_job_locked(job_id)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue


_load_persisted_jobs()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/transcriptions", response_model=JobView, status_code=202)
def create_transcription(
    file: UploadFile = File(...),
    trim_start_seconds: float | None = Form(None),
    trim_end_seconds: float | None = Form(None),
) -> JobView:
    filename = Path(file.filename or "recording").name
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_AUDIO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise HTTPException(status_code=400, detail=f"不支持的音频格式，支持：{supported}")

    job_id = uuid.uuid4().hex
    audio_path = UPLOAD_DIR / f"{job_id}{extension}"
    with audio_path.open("wb") as target:
        shutil.copyfileobj(file.file, target)

    try:
        duration = _probe_audio_duration(audio_path)
        trim_start, trim_end = _validate_trim_range(
            duration, trim_start_seconds, trim_end_seconds
        )
    except ValueError as error:
        audio_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(error)) from error
    trim_requested = trim_start > 0.05 or trim_end < duration - 0.05

    now = time.time()
    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "phase": "queued",
            "filename": filename,
            "audio_path": audio_path,
            "audio_download_name": filename,
            "created_at": now,
            "updated_at": now,
            "original_duration_seconds": duration,
            "trim_start_seconds": trim_start,
            "trim_end_seconds": trim_end,
            "trim_requested": trim_requested,
        }
        _persist_job_locked(job_id)
    _executor.submit(_run_transcription, job_id)
    return _job_view(job_id)


@app.get("/api/transcriptions", response_model=list[JobSummary])
def list_transcriptions() -> list[JobSummary]:
    with _jobs_lock:
        job_ids = sorted(
            _jobs,
            key=lambda item: _jobs[item].get("created_at", 0),
            reverse=True,
        )
        return [_job_summary_locked(job_id) for job_id in job_ids]


@app.get("/api/transcriptions/{job_id}", response_model=JobView)
def get_transcription(job_id: str) -> JobView:
    return _job_view(job_id)


@app.patch("/api/transcriptions/{job_id}", response_model=JobSummary)
def rename_transcription(job_id: str, request: RenameJobRequest) -> JobSummary:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="转写任务不存在")
        if job["status"] != "completed":
            raise HTTPException(status_code=409, detail="只有处理完成的记录可以重命名")
        try:
            return _rename_completed_job_locked(job_id, request.name)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except FileExistsError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except OSError as error:
            raise HTTPException(status_code=500, detail="无法同步修改本地文件名") from error


@app.delete("/api/transcriptions/{job_id}", status_code=204)
def delete_transcription(job_id: str) -> Response:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="转写任务不存在")
        if job["status"] not in {"completed", "failed"}:
            raise HTTPException(status_code=409, detail="正在处理的任务不能删除")
        audio_path = Path(job["audio_path"])
        _jobs.pop(job_id)

    if audio_path.parent.resolve() == UPLOAD_DIR.resolve():
        audio_path.unlink(missing_ok=True)
    # An interrupted FFmpeg process may leave its deterministic partial output
    # beside the uploaded source even though audio_path still points at source.
    (UPLOAD_DIR / f"{job_id}-trimmed.wav").unlink(missing_ok=True)
    job_output = OUTPUT_DIR / job_id
    if job_output.parent.resolve() == OUTPUT_DIR.resolve():
        shutil.rmtree(job_output, ignore_errors=True)
    return Response(status_code=204)


@app.get("/api/transcriptions/{job_id}/audio")
def get_audio(job_id: str) -> FileResponse:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="转写任务不存在")
        path = job["audio_path"]
    return FileResponse(path, filename=job.get("audio_download_name", job["filename"]))


@app.get("/api/transcriptions/{job_id}/export/{kind}")
def export_transcription(job_id: str, kind: Literal["json", "markdown"]) -> FileResponse:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="转写任务不存在")
        if job["status"] != "completed":
            raise HTTPException(status_code=409, detail="转写尚未完成")
        key = "json_path" if kind == "json" else "markdown_path"
        path = job[key]
        if path is None:
            raise HTTPException(status_code=404, detail="导出文件不存在")
    return FileResponse(path, filename=path.name)


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
