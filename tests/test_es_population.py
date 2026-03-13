import importlib.util
import sys
import types
import unittest
from pathlib import Path


def _load_population_module():
    repo_root = Path(__file__).resolve().parents[1]
    package_name = "es_finetuning"
    module_name = f"{package_name}.population"

    sys.modules["numpy"] = _make_fake_numpy()

    package = types.ModuleType(package_name)
    package.__path__ = [str(repo_root / package_name)]
    sys.modules.setdefault(package_name, package)

    noise_module = types.ModuleType(f"{package_name}.noise")
    noise_module.apply_es_update = lambda *args, **kwargs: None
    noise_module.perturb_weights = lambda *args, **kwargs: None
    noise_module.restore_weights = lambda *args, **kwargs: None
    sys.modules[f"{package_name}.noise"] = noise_module

    mod_path = repo_root / package_name / "population.py"
    spec = importlib.util.spec_from_file_location(module_name, mod_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
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

    def _array_split(values, sections):
        seq = list(values)
        length = len(seq)
        base = length // sections
        extra = length % sections
        out = []
        start = 0
        for idx in range(sections):
            stop = start + base + (1 if idx < extra else 0)
            out.append(_FakeArray(seq[start:stop], dtype=getattr(values, "dtype", None)))
            start = stop
        return out

    def _median(values):
        seq = sorted(values)
        n = len(seq)
        mid = n // 2
        if n % 2:
            return float(seq[mid])
        return float(seq[mid - 1] + seq[mid]) / 2.0

    def _mean(values):
        return float(sum(values)) / float(len(values))

    fake_numpy = types.SimpleNamespace()
    fake_numpy.ndarray = _FakeArray
    fake_numpy.int64 = "int64"
    fake_numpy.arange = lambda stop, dtype=None: _FakeArray(range(stop), dtype=dtype)
    fake_numpy.array_split = _array_split
    fake_numpy.median = _median
    fake_numpy.mean = _mean
    return fake_numpy


population = _load_population_module()
shard_population_indices = population.shard_population_indices
summarize_phase_history = population.summarize_phase_history
SingleProcessPopulationExecutor = population.SingleProcessPopulationExecutor


class EsPopulationTests(unittest.TestCase):
    def test_shard_population_indices_balances_members(self):
        self.assertEqual(shard_population_indices(8, 4, 0), [0, 1])
        self.assertEqual(shard_population_indices(8, 4, 3), [6, 7])
        self.assertEqual(shard_population_indices(3, 5, 4), [])

    def test_shard_population_indices_rejects_invalid_rank(self):
        with self.assertRaisesRegex(ValueError, "worker_rank"):
            shard_population_indices(4, 2, 2)

    def test_summarize_phase_history_respects_warmup(self):
        summary = summarize_phase_history(
            {
                "iteration_s": [10.0, 4.0, 6.0],
                "evaluate_s": [8.0, 3.0, 5.0],
            },
            startup_time_s=12.345678,
            warmup_iterations=1,
        )

        self.assertEqual(summary["startup_time_s"], 12.345678)
        self.assertEqual(summary["warmup_iterations"], 1)
        self.assertEqual(summary["measured_iterations"], 3)
        self.assertEqual(summary["steady_state_iterations"], 2)
        self.assertEqual(summary["phase_median_s"]["iteration_s"], 5.0)
        self.assertEqual(summary["phase_mean_s"]["evaluate_s"], 4.0)

    def test_single_process_executor_broadcast_object_is_passthrough(self):
        executor = SingleProcessPopulationExecutor()
        payload = {"bucket": 4096, "indices": [1, 2, 3]}
        self.assertEqual(
            executor.broadcast_object("any_tag", payload),
            payload,
        )


if __name__ == "__main__":
    unittest.main()
