import tempfile
import unittest
from pathlib import Path

from src.formatter import merge_adjacent_speaker_segments, write_markdown


class SpeakerDisplayMergeTests(unittest.TestCase):
    def test_merges_adjacent_same_speaker_with_small_gap(self):
        source = [
            {"start": 0.0, "end": 5.8, "speaker": "Speaker 0", "text": "第一段"},
            {"start": 6.0, "end": 11.5, "speaker": "Speaker 0", "text": "第二段"},
            {"start": 11.7, "end": 14.0, "speaker": "Speaker 1", "text": "回答"},
        ]

        merged = merge_adjacent_speaker_segments(source)

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["start"], 0.0)
        self.assertEqual(merged[0]["end"], 11.5)
        self.assertEqual(merged[0]["text"], "第一段，第二段")
        self.assertEqual(len(source), 3, "display merging must not mutate JSON source segments")

    def test_does_not_merge_across_long_pause(self):
        source = [
            {"start": 0.0, "end": 2.0, "speaker": "Speaker 0", "text": "前段"},
            {"start": 5.0, "end": 7.0, "speaker": "Speaker 0", "text": "后段"},
        ]
        self.assertEqual(len(merge_adjacent_speaker_segments(source)), 2)

    def test_markdown_merges_headings_with_chinese_comma(self):
        document = {
            "source": "interview.m4a",
            "segments": [
                {"start": 0.0, "end": 5.0, "speaker": "Speaker 0", "text": "你好"},
                {"start": 5.2, "end": 8.0, "speaker": "Speaker 0", "text": "请介绍自己"},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "interview.md"
            write_markdown(document, output)
            markdown = output.read_text(encoding="utf-8")

        self.assertEqual(markdown.count("· Speaker 0"), 1)
        self.assertIn("你好，请介绍自己", markdown)

    def test_does_not_duplicate_existing_punctuation(self):
        source = [
            {"start": 0.0, "end": 2.0, "speaker": "Speaker 0", "text": "你好。"},
            {"start": 2.1, "end": 4.0, "speaker": "Speaker 0", "text": "请介绍自己"},
        ]
        merged = merge_adjacent_speaker_segments(source)
        self.assertEqual(merged[0]["text"], "你好。请介绍自己")


if __name__ == "__main__":
    unittest.main()
