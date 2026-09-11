#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from src.asr import SPEAKER_MODES, FunASRTranscriber, TranscriptionError
from src.formatter import build_document, write_json, write_markdown
from src.utils import validate_audio_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用本地 FunASR 转写面试录音，并导出 Markdown / JSON。")
    parser.add_argument("audio", help="录音文件路径（支持 m4a、mp3、wav 等）")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="输出目录，默认：output")
    parser.add_argument("--hotwords", type=Path, default=Path("config/hotwords.txt"), help="hotwords 配置路径")
    parser.add_argument("--device", default="auto", help="auto、mps、cpu 或 cuda:0，默认：auto")
    parser.add_argument("--ncpu", type=int, default=4, help="CPU 推理线程数，默认：4")
    parser.add_argument(
        "--speaker-mode",
        choices=SPEAKER_MODES,
        default="vad_segment",
        help="Speaker 切段策略：vad_segment（默认，适合长录音）或 punc_segment",
    )
    parser.add_argument(
        "--max-segment-seconds",
        type=float,
        default=6.0,
        help="VAD 最大单段秒数，默认：6；越短越不容易混入两位说话人",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        audio_path = validate_audio_file(args.audio)
        if args.ncpu < 1:
            raise ValueError("--ncpu 必须大于 0")
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"读取录音：{audio_path.name}")
        print("加载 FunASR 模型（首次运行会下载模型）…")
        transcriber = FunASRTranscriber(
            args.hotwords,
            device=args.device,
            ncpu=args.ncpu,
            speaker_mode=args.speaker_mode,
            max_segment_seconds=args.max_segment_seconds,
        )
        print(
            f"开始转写（设备：{transcriber.device}，Speaker 切段：{transcriber.speaker_mode}，"
            f"VAD 最大单段：{transcriber.max_segment_seconds:g} 秒）…"
        )
        segments = transcriber.transcribe(audio_path)
        document = build_document(audio_path, segments)
        json_path = output_dir / f"{audio_path.stem}.json"
        markdown_path = output_dir / f"{audio_path.stem}.md"
        write_json(document, json_path)
        write_markdown(document, markdown_path)
        speaker_counts = Counter(segment["speaker"] for segment in segments)
        print("完成。")
        print(f"有效片段：{len(segments)}；Speaker 分布：{dict(sorted(speaker_counts.items()))}")
        print(f"Markdown：{markdown_path}")
        print(f"JSON：{json_path}")
        return 0
    except (FileNotFoundError, ValueError, TranscriptionError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
