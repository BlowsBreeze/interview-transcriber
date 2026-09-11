from __future__ import annotations

from pathlib import Path


SUPPORTED_AUDIO_EXTENSIONS = {".m4a", ".mp3", ".wav", ".flac", ".aac", ".ogg"}


def validate_audio_file(path: str | Path) -> Path:
    """Return a resolved input file path or raise a clear validation error."""
    audio_path = Path(path).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"找不到录音文件：{audio_path}")
    if not audio_path.is_file():
        raise ValueError(f"输入路径不是文件：{audio_path}")
    if audio_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise ValueError(f"不支持的音频格式：{audio_path.suffix or '无扩展名'}（支持：{supported}）")
    return audio_path


def read_hotwords(path: str | Path, default_weight: int = 20) -> str:
    """Read the editable hotword file in FunASR's 'word weight' syntax."""
    hotwords_path = Path(path)
    if not hotwords_path.exists():
        raise FileNotFoundError(f"找不到 hotwords 配置文件：{hotwords_path}")

    entries: list[str] = []
    for line_number, raw_line in enumerate(hotwords_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(maxsplit=1)
        if len(parts) == 2 and parts[1].isdigit():
            entries.append(f"{parts[0]} {parts[1]}")
        else:
            entries.append(f"{line} {default_weight}")
    return " ".join(entries)
