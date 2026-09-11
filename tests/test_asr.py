import unittest

from src.asr import (
    FunASRTranscriber,
    _ProgressTrackingMixin,
    normalize_text,
    sentence_info_schema,
)


class FunASRResultNormalizationTests(unittest.TestCase):
    def test_merges_funasr_token_list_without_python_list_representation(self):
        sentence = {
            "text": ["我", "主要", "使用", "React", "开发", "。"],
            "start": 1200,
            "end": 3400,
            "spk": 1,
            "timestamp": [[1200, 1300], [1300, 1500]],
        }

        normalized = FunASRTranscriber._normalize_sentence(sentence)

        self.assertEqual(normalized["text"], "我主要使用 React 开发。")
        self.assertEqual(normalized["start"], 1.2)
        self.assertEqual(normalized["end"], 3.4)
        self.assertEqual(normalized["speaker"], "Speaker 1")

    def test_handles_none_missing_and_nested_text_parts(self):
        self.assertEqual(normalize_text(None), "")
        self.assertEqual(normalize_text(["Vue", ["组件"], "化"]), "Vue 组件化")
        self.assertIsNone(FunASRTranscriber._normalize_sentence({"text": None}))

    def test_merges_tokenized_vad_string_and_keeps_english_word_spacing(self):
        self.assertEqual(
            normalize_text("这 个 过 程 中 呢 就 是 React Fiber 的 治 理"),
            "这个过程中呢就是 React Fiber 的治理",
        )
        self.assertEqual(normalize_text("Micro Frontend"), "Micro Frontend")

    def test_recovers_time_from_timestamp_and_handles_unknown_speaker(self):
        normalized = FunASRTranscriber._normalize_sentence(
            {"text": ["测试"], "timestamp": [[250, 500], [500, 900]]}
        )
        self.assertEqual(normalized["start"], 0.25)
        self.assertEqual(normalized["end"], 0.9)
        self.assertEqual(normalized["speaker"], "Speaker Unknown")

    def test_schema_reports_actual_field_types_without_text_content(self):
        schema = sentence_info_schema([{"text": ["敏感内容"], "start": 0, "spk": 1}])
        self.assertIn("text=list", schema)
        self.assertNotIn("敏感内容", schema)

    def test_vad_segment_is_the_default_and_punc_segment_remains_available(self):
        transcriber = FunASRTranscriber("config/hotwords.txt")
        self.assertEqual(transcriber.speaker_mode, "vad_segment")
        self.assertEqual(transcriber.max_segment_seconds, 6.0)
        legacy = FunASRTranscriber("config/hotwords.txt", speaker_mode="punc_segment")
        self.assertEqual(legacy.speaker_mode, "punc_segment")

    def test_rejects_unknown_speaker_mode_before_model_loading(self):
        with self.assertRaises(ValueError):
            FunASRTranscriber("config/hotwords.txt", speaker_mode="unknown")

    def test_rejects_non_positive_vad_segment_duration(self):
        with self.assertRaises(ValueError):
            FunASRTranscriber("config/hotwords.txt", max_segment_seconds=0)


class ProgressTrackingTests(unittest.TestCase):
    class FakeAutoModel:
        def __init__(self):
            self.model = object()
            self.vad_model = object()
            self.spk_model = object()

        def inference(self, input, input_len=None, model=None, **cfg):
            if model is self.vad_model:
                return [{"value": [[0, 1000], [1200, 2000], [2200, 3000]]}]
            return [{"spk_embedding": "fake"}]

    class ObservableFakeModel(_ProgressTrackingMixin, FakeAutoModel):
        pass

    def test_reports_vad_total_and_each_completed_speaker_segment(self):
        model = self.ObservableFakeModel()
        events = []
        model.set_progress_reporter(lambda phase, current, total: events.append((phase, current, total)))

        model.inference("audio", model=model.vad_model)
        model.inference("segment-1", model=model.spk_model)
        model.inference("segment-2", model=model.spk_model)
        model.inference("segment-3", model=model.spk_model)

        self.assertEqual(
            events,
            [
                ("transcribing", 0, 3),
                ("transcribing", 1, 3),
                ("transcribing", 2, 3),
                ("transcribing", 3, 3),
            ],
        )


if __name__ == "__main__":
    unittest.main()
