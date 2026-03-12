import unittest
import importlib.util
from pathlib import Path


def _load_tpu_guardrails_module():
    repo_root = Path(__file__).resolve().parents[1]
    mod_path = repo_root / "es_finetuning" / "tpu_guardrails.py"
    spec = importlib.util.spec_from_file_location("tpu_guardrails", mod_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


tpu_guardrails = _load_tpu_guardrails_module()
is_tpu_device = tpu_guardrails.is_tpu_device
pad_partial_tpu_batch = tpu_guardrails.pad_partial_tpu_batch
validate_tpu_batch_settings = tpu_guardrails.validate_tpu_batch_settings


class TpuGuardrailsTests(unittest.TestCase):
    def test_is_tpu_device(self):
        self.assertTrue(is_tpu_device("xla:0"))
        self.assertFalse(is_tpu_device("cpu"))

    def test_validate_tpu_batch_settings_rejects_auto(self):
        with self.assertRaisesRegex(ValueError, "fixed integer batch size"):
            validate_tpu_batch_settings("auto")

    def test_validate_tpu_batch_settings_rejects_non_int(self):
        with self.assertRaisesRegex(ValueError, "integer batch size"):
            validate_tpu_batch_settings("18")

    def test_validate_tpu_batch_settings_rejects_non_positive(self):
        with self.assertRaisesRegex(ValueError, "batch_size > 0"):
            validate_tpu_batch_settings(0)

    def test_validate_tpu_batch_settings_rejects_mismatch(self):
        with self.assertRaisesRegex(
            ValueError,
            "mini_batch_size == batch_size",
        ):
            validate_tpu_batch_settings(18, mini_batch_size=16)

    def test_validate_tpu_batch_settings_accepts_valid(self):
        self.assertEqual(
            validate_tpu_batch_settings(18, mini_batch_size=18),
            18,
        )

    def test_pad_partial_tpu_batch_no_padding_on_cpu(self):
        contexts = ["a", "b"]
        pop_idxs = [10, 11]
        precache = [("k0", "v0"), ("k1", "v1")]
        out = pad_partial_tpu_batch(
            contexts,
            pop_idxs,
            precache,
            batch_size=4,
            device="cpu",
        )
        out_contexts, out_pop_idxs, out_precache, original_n = out
        self.assertEqual(original_n, 2)
        self.assertEqual(out_contexts, ["a", "b"])
        self.assertEqual(out_pop_idxs, [10, 11])
        self.assertEqual(out_precache, [("k0", "v0"), ("k1", "v1")])

    def test_pad_partial_tpu_batch_pads_on_xla(self):
        contexts = ["a", "b"]
        pop_idxs = [10, 11]
        precache = [("k0", "v0"), ("k1", "v1")]
        out = pad_partial_tpu_batch(
            contexts,
            pop_idxs,
            precache,
            batch_size=4,
            device="xla:0",
        )
        out_contexts, out_pop_idxs, out_precache, original_n = out

        self.assertEqual(original_n, 2)
        self.assertEqual(out_contexts, ["a", "b", "b", "b"])
        self.assertEqual(out_pop_idxs, [10, 11, 11, 11])
        self.assertEqual(
            out_precache,
            [("k0", "v0"), ("k1", "v1"), ("k1", "v1"), ("k1", "v1")],
        )

    def test_pad_partial_tpu_batch_no_padding_for_full_batch(self):
        contexts = ["a", "b", "c", "d"]
        out = pad_partial_tpu_batch(
            contexts,
            chunk_pop_idxs=None,
            chunk_precached_tensors=None,
            batch_size=4,
            device="xla:0",
        )
        out_contexts, out_pop_idxs, out_precache, original_n = out
        self.assertEqual(original_n, 4)
        self.assertEqual(out_contexts, contexts)
        self.assertIsNone(out_pop_idxs)
        self.assertIsNone(out_precache)


if __name__ == "__main__":
    unittest.main()
