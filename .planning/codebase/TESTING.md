# Testing Patterns

**Analysis Date:** 2026-02-25

## Test Framework

**Runner:**
- Not detected - no `pytest`, `unittest`, or other test runner configuration found
- No test configuration files: `pytest.ini`, `setup.cfg`, `tox.ini` not present
- No test runner specified in `pyproject.toml` (file does not exist)

**Assertion Library:**
- Not applicable - no automated test suite detected

**Run Commands:**
- Not established - manual testing approach appears to be in use
- No test scripts in repository

## Test File Organization

**Location:**
- No test files detected in codebase
- File search for `*test*.py`, `test_*.py`, `*_test.py` returned no results
- Testing appears to be manual or external to this codebase

**Naming:**
- Not applicable

**Structure:**
- Not applicable

## Test Strategy

**Observed Approach:**
This codebase appears to follow a **manual testing and experimental evaluation** approach rather than automated unit testing:

1. **Evaluation-Driven**: Core validation happens through `memory_evaluator.py` which evaluates memory policies against benchmark tasks
2. **Benchmark Datasets**: Uses LongBench and ChouBun datasets for evaluation (defined in `utils_longbench.py`)
3. **Metric Computation**: Dedicated metric files compute evaluation scores:
   - `longbench_metrics.py`: 20+ metric functions for different task types (QA F1, ROUGE, retrieval, code similarity)
   - `choubun_metrics.py`: Japanese-specific metrics (ROUGE-JA, QA F1 for Japanese)

4. **Trainer Validation**: `memory_trainer.py` includes evaluation loops:
   - `eval_interval`: Periodic evaluation during training
   - `eval_iters`: Number of evaluation iterations
   - Early stopping based on patience

## Validation and Metrics

**Evaluation Functions** (`utils_longbench.py`):

```python
def get_score(task, predictions, answers, all_classes):
    """Main scoring function for all benchmarks"""
    total_score = 0.
    all_scores = []

    # Task-specific tokenizer (Japanese requires Fugashi)
    if task in ["wiki_qa", "edinet_qa", "corp_sec_qa", "corp_sec_sum"]:
        from fugashi import Tagger
        tokenizer = Tagger('-Owakati')
    else:
        tokenizer = None

    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if task in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            cur_score = dataset2metric[task](
                prediction,
                ground_truth,
                tokenizer=tokenizer,
                all_classes=all_classes
            )
            score = max(score, cur_score)
        all_scores.append(score)
        total_score += score
    mean_score = 100 * total_score / len(predictions)
    return mean_score, all_scores
```

**Metric Registry** (line 22-51 in `utils_longbench.py`):
```python
dataset2metric = {
    # LongBench tasks
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "hotpotqa": qa_f1_score,
    "gov_report": rouge_score,
    "passage_retrieval_en": retrieval_score,
    "lcc": code_sim_score,
    # ChouBun tasks (Japanese)
    "wiki_qa": qa_f1_ja_score,
    "corp_sec_sum": rouge_ja_score,
}
```

## Metric Implementations

**Metric Function Pattern** (`longbench_metrics.py`):

```python
def count_score(prediction, ground_truth, **kwargs):
    """Extract and match numeric values in prediction vs ground truth"""
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)

def retrieval_score(prediction, ground_truth, **kwargs):
    """Extract paragraph number from prediction and match with ground truth"""
    pattern = r'Paragraph (\d+)'
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    # ... matching logic
    return float(final_score)
```

**Metric Types Supported:**
- **QA F1**: Token-level F1 score using Counter intersection (`qa_f1_score`)
- **ROUGE**: Using Rouge library for sequence overlap (`rouge_score`)
- **Classification**: Exact match or fuzzy matching for classification tasks
- **Retrieval**: Paragraph ID extraction and matching
- **Code Similarity**: Fuzzy string matching using fuzzywuzzy
- **Count**: Numeric value extraction and matching
- **Language-specific**: Japanese token-aware metrics with Fugashi tokenizer

## Model Validation

**Trainer Evaluation Loop** (inferred from `memory_trainer.py`):

1. **Periodic Evaluation** (line 59-62 in `memory_trainer.py`):
   - `eval_interval`: Frequency of evaluation (every N iterations)
   - `eval_iters`: Number of samples per evaluation
   - `eval_only`: Optional evaluation-only mode

2. **Early Stopping** (lines 203-207):
   ```python
   self.early_stop_patience = trainer_config.early_stop_patience
   self.early_stop_counter = torch.zeros(1).to(device)
   self.early_stop_flag = torch.zeros(1).to(device)
   ```

3. **Distributed Evaluation** (line 46):
   - `allow_distributed_eval`: Enable multi-GPU evaluation
   - Per-rank evaluation with synchronization

4. **Advanced Stats** (line 68-70):
   - `record_advanced_eval_stats`: Collect fine-grained metrics
   - `store_eval_results_locally`: Save evaluation results
   - `record_per_task_eval_stats`: Track per-benchmark performance

## Test Data

**Benchmark Datasets:**
- **LongBench**: 20 long-context language understanding tasks (English/Chinese)
  - Tasks: NarrativeQA, QASPER, HotpotQA, Gov Report, QMSum, TREC, TriviaQA, etc.
  - Context length: 4k-8k tokens

- **ChouBun**: Japanese benchmark suite
  - Tasks: Wiki QA, EDINET QA, Corporate Sec QA, Corporate Sec Summarization
  - Requires Japanese-specific tokenization

**Data Loading** (`task_sampler.py`):
- Uses Hugging Face `datasets` library: `from datasets import load_dataset`
- Task-based sampling with configurable batch sizes
- Per-task and per-population evaluation options

## Coverage and Statistics

**Requirements:** No explicit coverage requirements detected

**View Coverage:**
- Not applicable - no coverage tool configured

**Statistics Collection** (`utils.py`):
```python
def pop_stats_from_dict_of_lists(aggregated_result_dict: dict, prefix=None):
    """Compute mean, std, best, worst statistics from result lists"""
    stats_dict = {}
    if prefix is None:
        prefix = ''
    for k, l in aggregated_result_dict.items():
        stats_dict[prefix + k + '_mean'] = np.mean(l)
        stats_dict[prefix + k + '_std'] = np.std(l)
        stats_dict[prefix + k + '_best'] = np.max(l)
        stats_dict[prefix + k + '_worst'] = np.min(l)
    return stats_dict
```

## Manual Testing Approach

**Memory Policy Validation:**
- Test by evaluating on benchmark tasks using `memory_evaluator.py`
- Population-based search uses evolution algorithm (`memory_evolution/cma_es.py`)
- Best parameters tracked across iterations

**Distributed Training Validation:**
- Multi-GPU synchronization via `torch.distributed`
- Rank-aware logging and checkpoint coordination
- Buffer merging for synchronized parameters

**Gradient/Parameter Correctness:**
- Checked implicitly through model forward pass success and numerical stability
- OOM detection via `is_oom_exception()` indicates resource validation
- Nan/Inf checking likely done during loss computation (not visible in provided files)

## No Unit Tests

**Rationale:**
- This is a research/experimental codebase focused on memory-augmented language models
- Validation occurs through empirical evaluation on standard benchmarks
- Test-driven development not applied
- Code changes validated through re-evaluation on benchmarks

**Impact:**
- Changes require full benchmark re-evaluation to validate
- No fast feedback loop on code changes
- Potential for regressions in non-critical paths
- Metric implementations in `*_metrics.py` are untested by automated tests

## Integration Testing Pattern

**Trainer Integration** (`memory_trainer.py`):
- Full pipeline test: model initialization → sampling → evaluation → parameter update
- Spans: `memory_evaluator.py`, `memory_policy/`, `memory_llms/`, `memory_evolution/`
- Validates end-to-end training flow

**Evaluation Integration** (`utils_longbench.py`):
- End-to-end scoring: prediction generation → metric selection → score aggregation
- Tests interaction between `memory_evaluator.py` and metric functions

---

*Testing analysis: 2026-02-25*
