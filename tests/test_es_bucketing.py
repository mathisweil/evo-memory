import importlib.util
import sys
import types
import unittest
from pathlib import Path


def _load_bucketing_module():
    repo_root = Path(__file__).resolve().parents[1]
    sys.modules["numpy"] = _make_fake_numpy()
    mod_path = repo_root / "es_finetuning" / "bucketing.py"
    spec = importlib.util.spec_from_file_location("es_bucketing", mod_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_fake_numpy():
    class _FakeArray(list):
        def __init__(self, values, dtype=None):
            super().__init__(values)
            self.dtype = dtype
            self.shape = (len(values),)

        def tolist(self):
            return list(self)

    class _FakeRandom:
        @staticmethod
        def seed(_seed):
            return None

        @staticmethod
        def choice(values, size=None, replace=False):
            seq = list(values)
            if size is None:
                return seq[0]
            if not replace and size > len(seq):
                raise ValueError("size exceeds available values")
            if replace:
                out = [seq[0] for _ in range(size)]
            else:
                out = seq[:size]
            dtype = getattr(values, "dtype", None)
            return _FakeArray(out, dtype=dtype)

    fake_numpy = types.SimpleNamespace()
    fake_numpy.ndarray = _FakeArray
    fake_numpy.int64 = "int64"
    fake_numpy.array = lambda values, dtype=None: _FakeArray(list(values), dtype=dtype)
    fake_numpy.random = _FakeRandom()
    return fake_numpy


bucketing = _load_bucketing_module()
bucket_sequence_length = bucketing.bucket_sequence_length
build_bucketed_request_pools = bucketing.build_bucketed_request_pools
make_bucketed_resample_fn = bucketing.make_bucketed_resample_fn
iter_compile_warmup_requests = bucketing.iter_compile_warmup_requests


class _FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        token_count = len(text.split())
        if add_special_tokens:
            token_count += 1
        return type("Tokenized", (), {"input_ids": list(range(token_count))})()


class _FakeTaskSampler:
    def __init__(self):
        self.latest = None

    def set_requests_per_task(self, requests_dict):
        self.latest = requests_dict


class EsBucketingTests(unittest.TestCase):
    def test_bucket_sequence_length_rounds_up(self):
        self.assertEqual(
            bucket_sequence_length(750, bucket_boundaries=[512, 1024, 2048]),
            1024,
        )
        self.assertEqual(
            bucket_sequence_length(4097, bucket_boundaries=[512, 1024, 2048, 4096]),
            5120,
        )

    def test_build_bucketed_request_pools_filters_ineligible_buckets(self):
        tokenizer = _FakeTokenizer()
        task_prompts = {
            "lb/qasper": [
                "one two",
                "one two three",
                "one two three four",
                "one two three four five",
            ],
            "lb/triviaqa": [
                "one two",
                "one two three",
                "one two three four",
                "short",
            ],
        }

        pools = build_bucketed_request_pools(
            task_prompts=task_prompts,
            tokenizer=tokenizer,
            sampled_requests_per_task=2,
            max_prompt_conditioning=None,
            add_special_tokens=False,
            bucket_boundaries=[2, 4, 8],
        )

        self.assertEqual(sorted(pools), [4])
        self.assertEqual(
            pools[4],
            {
                "lb/qasper": [1, 2],
                "lb/triviaqa": [1, 2],
            },
        )

    def test_make_bucketed_resample_fn_sets_requests_and_returns_dict(self):
        sampler = _FakeTaskSampler()
        bucket_pools = {
            4: {
                "lb/qasper": [10, 11, 12],
                "lb/triviaqa": [20, 21, 22],
            }
        }

        resample_fn = make_bucketed_resample_fn(
            sampler,
            bucket_pools,
            sampled_requests_per_task=2,
        )
        requests_dict = resample_fn()

        self.assertIsNotNone(sampler.latest)
        self.assertEqual(set(requests_dict), {"lb/qasper", "lb/triviaqa"})
        for task_name, indices in requests_dict.items():
            self.assertEqual(indices.shape, (2,))
            self.assertEqual(indices.dtype, "int64")
            self.assertTrue(set(indices.tolist()).issubset(set(bucket_pools[4][task_name])))

    def test_iter_compile_warmup_requests_uses_first_indices(self):
        bucket_pools = {
            4: {"lb/qasper": [9, 10, 11]},
            8: {"lb/qasper": [20, 21, 22]},
        }

        warmups = list(
            iter_compile_warmup_requests(
                bucket_pools,
                sampled_requests_per_task=2,
            )
        )

        self.assertEqual(len(warmups), 2)
        self.assertEqual(warmups[0][0], 4)
        self.assertEqual(warmups[0][1]["lb/qasper"].tolist(), [9, 10])
        self.assertEqual(warmups[1][0], 8)
        self.assertEqual(warmups[1][1]["lb/qasper"].tolist(), [20, 21])


if __name__ == "__main__":
    unittest.main()
