from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def format_timestamp(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}" if hours else f"{minutes:02}:{seconds:02}"


def build_document(source: Path, segments: list[dict[str, Any]]) -> dict[str, Any]:
    return {"source": source.name, "segments": segments}


def write_json(document: dict[str, Any], output_path: Path) -> None:
    output_path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _join_display_text(left: str, right: str) -> str:
    """Join two ASR chunks readably without producing duplicate punctuation."""
    left = left.rstrip()
    right = right.lstrip()
    if not left:
        return right
    if not right:
        return left
    if left[-1] in "，。！？；：、,.!?;:":
        return left + right
    return left + "，" + right


def merge_adjacent_speaker_segments(
    segments: list[dict[str, Any]], max_gap_seconds: float = 2.0
) -> list[dict[str, Any]]:
    """Group adjacent display segments without changing the source JSON data.

    Each original VAD segment remains available in JSON for precise seeking and
    later corrections. Markdown groups only consecutive segments with the same
    anonymous speaker and a small time gap.
    """
    merged: list[dict[str, Any]] = []
    for segment in segments:
        current = dict(segment)
        if not merged:
            merged.append(current)
            continue

        previous = merged[-1]
        gap = float(current["start"]) - float(previous["end"])
        same_speaker = current.get("speaker") == previous.get("speaker")
        close_enough = -0.1 <= gap <= max_gap_seconds
        if same_speaker and close_enough:
            previous["end"] = max(float(previous["end"]), float(current["end"]))
            # VAD boundaries are often short pauses rather than sentence ends.
            # A Chinese comma gives merged Speaker turns a readable visual flow.
            previous["text"] = _join_display_text(previous["text"], current["text"])
        else:
            merged.append(current)
    return merged


def write_markdown(
    document: dict[str, Any], output_path: Path, merge_speakers: bool = True
) -> None:
    lines = [f"# {Path(document['source']).stem}", "", f"原始文件：{document['source']}", ""]
    source_segments = document["segments"]
    display_segments = merge_adjacent_speaker_segments(source_segments) if merge_speakers else source_segments
    for segment in display_segments:
        start = format_timestamp(float(segment["start"]))
        end = format_timestamp(float(segment["end"]))
        lines.extend([f"## {start} - {end} · {segment['speaker']}", "", segment["text"], ""])
    if not source_segments:
        lines.extend(["_未检测到可转写的语音内容。_", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")
