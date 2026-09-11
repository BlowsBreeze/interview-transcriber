import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import backend.app as backend
from backend.app import _normalize_record_name, _validate_trim_range


class TrimRangeValidationTests(unittest.TestCase):
    def test_defaults_to_the_complete_recording(self):
        self.assertEqual(_validate_trim_range(120.0, None, None), (0.0, 120.0))

    def test_accepts_a_valid_contiguous_range(self):
        self.assertEqual(_validate_trim_range(120.0, 10.25, 80.75), (10.25, 80.75))

    def test_rejects_negative_start_and_out_of_bounds_end(self):
        with self.assertRaisesRegex(ValueError, "不能小于"):
            _validate_trim_range(120.0, -1.0, 80.0)
        with self.assertRaisesRegex(ValueError, "不能超过"):
            _validate_trim_range(120.0, 10.0, 121.0)

    def test_rejects_a_range_shorter_than_one_second(self):
        with self.assertRaisesRegex(ValueError, "至少需要保留 1 秒"):
            _validate_trim_range(120.0, 30.0, 30.5)


class RecordRenameTests(unittest.TestCase):
    def test_normalizes_name_and_preserves_original_extension(self):
        self.assertEqual(
            _normalize_record_name("  新的面试记录.m4a  ", "旧名称.m4a"),
            ("新的面试记录", "新的面试记录.m4a"),
        )

    def test_rejects_paths_and_empty_names(self):
        for name in ("", "../secret", "bad/name", "bad:name"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                _normalize_record_name(name, "旧名称.m4a")

    def test_renames_audio_exports_metadata_and_document_titles(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            upload_dir = root / "input"
            output_dir = root / "output"
            job_id = "a" * 32
            job_output = output_dir / job_id
            upload_dir.mkdir()
            job_output.mkdir(parents=True)
            audio_path = upload_dir / f"{job_id}.m4a"
            json_path = job_output / "旧名称.json"
            markdown_path = job_output / "旧名称.md"
            audio_path.write_bytes(b"audio")
            json_path.write_text("{}", encoding="utf-8")
            markdown_path.write_text("old", encoding="utf-8")
            job = {
                "id": job_id,
                "status": "completed",
                "phase": "completed",
                "filename": "旧名称.m4a",
                "audio_path": audio_path,
                "audio_download_name": "旧名称.m4a",
                "json_path": json_path,
                "markdown_path": markdown_path,
                "segments": [{"start": 0.0, "end": 1.0, "speaker": "Speaker 0", "text": "测试"}],
                "created_at": 1.0,
            }

            with (
                patch.object(backend, "UPLOAD_DIR", upload_dir),
                patch.object(backend, "OUTPUT_DIR", output_dir),
                patch.dict(backend._jobs, {job_id: job}),
            ):
                summary = backend._rename_completed_job_locked(job_id, "新的名称")

                new_audio = upload_dir / f"{job_id}-新的名称.m4a"
                new_json = job_output / "新的名称.json"
                new_markdown = job_output / "新的名称.md"
                self.assertEqual(summary.filename, "新的名称.m4a")
                self.assertTrue(new_audio.exists())
                self.assertTrue(new_json.exists())
                self.assertTrue(new_markdown.exists())
                self.assertFalse(audio_path.exists())
                self.assertFalse(json_path.exists())
                self.assertFalse(markdown_path.exists())
                self.assertEqual(json.loads(new_json.read_text(encoding="utf-8"))["source"], "新的名称.m4a")
                self.assertTrue(new_markdown.read_text(encoding="utf-8").startswith("# 新的名称"))
                metadata = json.loads((job_output / "job.json").read_text(encoding="utf-8"))
                self.assertEqual(metadata["filename"], "新的名称.m4a")


if __name__ == "__main__":
    unittest.main()
