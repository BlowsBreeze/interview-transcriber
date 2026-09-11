from __future__ import annotations

import platform
import logging
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any, Callable

from .utils import read_hotwords


class TranscriptionError(RuntimeError):
    """An error raised while loading or running the local ASR pipeline."""


logger = logging.getLogger(__name__)
SPEAKER_MODES = ("vad_segment", "punc_segment")
ProgressCallback = Callable[[str, int, int], None]


class _ProgressTrackingMixin:
    """Observe FunASR's VAD and CAM++ calls without changing its pipeline.

    FunASR's public callback restarts for every nested inference call. This
    mixin instead counts the VAD segments returned by FSMN-VAD and advances
    once each segment's CAM++ inference has completed.
    """

    _progress_reporter: ProgressCallback | None = None
    _progress_total = 0
    _progress_completed = 0

    def set_progress_reporter(self, reporter: ProgressCallback | None) -> None:
        self._progress_reporter = reporter
        self._progress_total = 0
        self._progress_completed = 0

    def _report_progress(self, phase: str) -> None:
        if self._progress_reporter is None:
            return
        try:
            self._progress_reporter(phase, self._progress_completed, self._progress_total)
        except Exception:
            logger.exception("转写进度回调执行失败")

    def inference(self, input: Any, input_len: Any = None, model: Any = None, **cfg: Any) -> Any:
        actual_model = self.model if model is None else model
        results = super().inference(input, input_len=input_len, model=model, **cfg)

        if actual_model is getattr(self, "vad_model", None):
            self._progress_total = sum(
                len(item.get("value", []))
                for item in results
                if isinstance(item, Mapping) and isinstance(item.get("value"), Sequence)
            )
            self._progress_completed = 0
            self._report_progress("transcribing")
        elif actual_model is getattr(self, "spk_model", None) and self._progress_total:
            # FunASR invokes CAM++ once for each original VAD segment. The
            # speaker input itself may contain several overlapping voiceprint
            # chunks, but it still represents one completed transcript segment.
            self._progress_completed = min(self._progress_completed + 1, self._progress_total)
            self._report_progress("transcribing")
        return results


def normalize_text(value: Any) -> str:
    """Normalize FunASR sentence text without rendering token lists as Python reprs.

    FunASR's ``timestamp_sentence`` fallback returns ``text`` as the result of
    ``text_postprocessed.split()`` when punctuation IDs are unavailable.  Those
    list items are ASR tokens, not a display representation.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        # vad_segment may return a tokenized string such as
        # "这 个 React 项 目" rather than a list. Treat whitespace-separated
        # values as tokens and apply the same Chinese/English joining rules.
        tokens = value.strip().split()
        return _merge_text_tokens(tokens)
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return ""

    tokens: list[str] = []
    for part in value:
        normalized = normalize_text(part)
        if normalized:
            tokens.append(normalized)

    return _merge_text_tokens(tokens)


def _merge_text_tokens(tokens: Sequence[str]) -> str:
    merged = ""
    for token in tokens:
        if not token:
            continue
        # Match FunASR's timestamp_sentence behavior: English tokens are
        # separated from adjacent Chinese/English tokens, while Chinese tokens
        # and punctuation stay contiguous.
        starts_latin = token[0].isascii() and token[0].isalpha()
        ends_latin = bool(merged) and merged[-1].isascii() and merged[-1].isalpha()
        if merged and (starts_latin or ends_latin):
            merged += " "
        merged += token
    return merged.strip()


def _time_in_seconds(value: Any, fallback: float = 0.0) -> float:
    """Convert FunASR millisecond timestamps defensively to seconds."""
    if isinstance(value, Real) and not isinstance(value, bool):
        return round(float(value) / 1000, 3)
    if isinstance(value, str):
        try:
            return round(float(value) / 1000, 3)
        except ValueError:
            return fallback
    return fallback


def _timestamp_bounds(sentence: Mapping[str, Any]) -> tuple[Any, Any]:
    """Use start/end first, then recover bounds from a timestamp pair list."""
    start = sentence.get("start")
    end = sentence.get("end")
    timestamps = sentence.get("timestamp") or sentence.get("timestamps") or []
    if (start is None or end is None) and isinstance(timestamps, Sequence) and timestamps:
        first, last = timestamps[0], timestamps[-1]
        if isinstance(first, Sequence) and len(first) >= 2 and isinstance(last, Sequence) and len(last) >= 2:
            start = first[0] if start is None else start
            end = last[1] if end is None else end
    return start, end


def sentence_info_schema(sentence_info: Any) -> str:
    """Return a privacy-preserving description of actual FunASR result shapes."""
    if not isinstance(sentence_info, list):
        return f"{type(sentence_info).__name__} (expected list)"
    samples = [item for item in sentence_info if isinstance(item, Mapping)][:5]
    if not samples:
        return f"list[{len(sentence_info)}]"
    fields = ("text", "sentence", "start", "end", "timestamp", "spk")
    types = []
    for field in fields:
        observed = sorted({type(item.get(field)).__name__ for item in samples if field in item})
        if observed:
            types.append(f"{field}={'|'.join(observed)}")
    return f"list[{len(sentence_info)}]: " + ", ".join(types)


def resolve_device(requested: str) -> str:
    """Choose a practical default device, including Apple Silicon MPS when available."""
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        if platform.system() == "Darwin" and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class FunASRTranscriber:
    """Loads the complete FunASR pipeline once and transcribes audio files."""

    def __init__(
        self,
        hotwords_file: str | Path,
        device: str = "auto",
        ncpu: int = 4,
        speaker_mode: str = "vad_segment",
        max_segment_seconds: float = 6.0,
    ) -> None:
        if speaker_mode not in SPEAKER_MODES:
            allowed = ", ".join(SPEAKER_MODES)
            raise ValueError(f"不支持的 Speaker 切段模式：{speaker_mode}（可选：{allowed}）")
        if max_segment_seconds <= 0:
            raise ValueError("VAD 最大单段时长必须大于 0 秒")
        self.hotwords = read_hotwords(hotwords_file)
        self.device = resolve_device(device)
        self.ncpu = ncpu
        self.speaker_mode = speaker_mode
        self.max_segment_seconds = max_segment_seconds
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from funasr import AutoModel
        except ImportError as error:
            raise TranscriptionError(
                "未安装 FunASR。请先按 README 安装 PyTorch 和 requirements.txt。"
            ) from error

        try:
            class ObservableAutoModel(_ProgressTrackingMixin, AutoModel):
                pass

            self._model = ObservableAutoModel(
                model="paraformer-zh",
                vad_model="fsmn-vad",
                vad_kwargs={"max_single_segment_time": int(self.max_segment_seconds * 1000)},
                punc_model="ct-punc",
                spk_model="cam++",
                # VAD produces speech turns even if CT-Punc cannot split a
                # long recording into sentences. punc_segment remains
                # available for recordings whose punctuation is reliable.
                spk_mode=self.speaker_mode,
                device=self.device,
                ncpu=self.ncpu,
                disable_update=True,
            )
        except Exception as error:
            raise TranscriptionError(
                f"无法初始化 FunASR 模型（设备：{self.device}）。首次运行需要下载模型；"
                "请确认网络、Python/PyTorch 安装和可用磁盘空间。"
            ) from error
        return self._model

    def transcribe(
        self,
        audio_path: str | Path,
        progress_callback: ProgressCallback | None = None,
    ) -> list[dict[str, Any]]:
        """Return normalized transcript segments from one local recording."""
        model = self._load_model()
        if hasattr(model, "set_progress_reporter"):
            model.set_progress_reporter(progress_callback)
        if progress_callback:
            progress_callback("vad", 0, 0)
        try:
            results = model.generate(
                input=str(audio_path),
                hotword=self.hotwords or None,
                return_spk_res=True,
                sentence_timestamp=True,
            )
            if progress_callback:
                total = getattr(model, "_progress_total", 0)
                progress_callback("finalizing", total, total)
        except Exception as error:
            raise TranscriptionError(
                "转写失败。若输入为 m4a/mp3，请确认已安装 ffmpeg；也请检查录音是否损坏。"
            ) from error
        finally:
            if hasattr(model, "set_progress_reporter"):
                model.set_progress_reporter(None)

        if not results:
            return []
        result = results[0] if isinstance(results, list) else results
        sentence_info = result.get("sentence_info", []) if isinstance(result, dict) else []
        print(f"FunASR sentence_info 结构：{sentence_info_schema(sentence_info)}")
        segments: list[dict[str, Any]] = []
        if isinstance(sentence_info, list):
            for index, item in enumerate(sentence_info):
                if not isinstance(item, Mapping):
                    logger.warning("跳过第 %d 个异常 sentence：期望字典，实际为 %s", index, type(item).__name__)
                    continue
                try:
                    segment = self._normalize_sentence(item)
                except Exception:
                    logger.exception("跳过第 %d 个无法解析的 FunASR sentence", index)
                    continue
                if segment is not None:
                    segments.append(segment)
        if segments:
            return segments

        text = normalize_text(result.get("text")) if isinstance(result, dict) else ""
        return [{"start": 0.0, "end": 0.0, "speaker": "Speaker Unknown", "text": text}] if text else []

    @staticmethod
    def _normalize_sentence(sentence: Mapping[str, Any]) -> dict[str, Any] | None:
        # Current FunASR may supply `sentence` or `text`; in its no-punctuation
        # fallback `text` is a token list. Prefer a non-empty sentence value.
        text = normalize_text(sentence.get("sentence")) or normalize_text(sentence.get("text"))
        if not text:
            return None
        start_ms, end_ms = _timestamp_bounds(sentence)
        start = _time_in_seconds(start_ms)
        end = _time_in_seconds(end_ms, fallback=start)
        if end < start:
            logger.warning("FunASR sentence 的结束时间早于开始时间，已将结束时间修正为开始时间。")
            end = start
        speaker_id = sentence.get("spk")
        speaker = f"Speaker {speaker_id}" if speaker_id is not None else "Speaker Unknown"
        return {
            "start": start,
            "end": end,
            "speaker": speaker,
            "text": text,
        }
