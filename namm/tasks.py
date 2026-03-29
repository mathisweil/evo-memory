import os
import random
import itertools
import json
import collections
import logging
import sys
from typing import Optional, Union
from datasets import load_dataset

import copy

import numpy as np

from namm.evaluation.longbench import get_score


def _shortest_answer_words(json_obj):
    """Return word count of the shortest answer in a LongBench sample."""
    answers = json_obj.get("answers", [])
    if not answers:
        return 0
    if isinstance(answers, str):
        return len(answers.split())
    return min(len(a.split()) for a in answers)


def merge_list_of_dicts(dicts, other_dicts):
    assert len(dicts) == len(other_dicts)
    new_dicts = [{**d1, **d2} for d1, d2 in zip(dicts, other_dicts)]
    return new_dicts


class TaskSampler():
    def __init__(
        self,
        tasks,  # list of tasks to load
        metrics,  # list of metrics per task
        training_tasks_subset: Optional[list] = None,
        test_tasks_subset: Optional[list] = None,
        store_gen_outputs: bool = False,
        store_gen_outputs_path: Optional[str] = None,
        max_conditioning_length: Optional[int] = None,
        max_answer_tokens: Optional[int] = None,
        train_split: Optional[float] = None,
        split_seed: int = 42,
    ):

        self.store_gen_outputs = store_gen_outputs

        if store_gen_outputs_path is not None:
            assert store_gen_outputs
        elif store_gen_outputs:
            store_gen_outputs_path = 'generated_outputs/temp/'

        self.store_gen_outputs_path = store_gen_outputs_path

        if store_gen_outputs:
            if not os.path.exists(store_gen_outputs_path):
                os.makedirs(store_gen_outputs_path)

        if type(tasks) == str:
            tasks = [tasks]
        else:
            tasks = list(tasks)
        if type(metrics) == str:
            metrics = [metrics for _ in tasks]
        else:
            metrics = list(metrics)

        assert len(metrics) == len(tasks)

        self.lb_tasks = []
        self.lb_metrics = []
        self.lb_datasets = []
        for t, m in zip(tasks, metrics):
            if t.startswith('lb/'):
                self.add_long_bench_task(task=t, metric=m)
            elif t.startswith('choubun/'):
                self.add_choubun_task(task=t, metric=m)
            else:
                raise NotImplementedError

        self.lb_training_tasks = training_tasks_subset or tasks
        self.lb_test_tasks = test_tasks_subset or tasks
        self.training_tasks_subset = self.lb_training_tasks
        self.test_tasks_subset = self.lb_test_tasks
        self.prefetched_task_tensors = {t: None for t in tasks}
        self.loaded_cached_model_data = False
        self.cached_per_task_stats = {}
        self.max_conditioning_length = max_conditioning_length
        self.max_answer_tokens = max_answer_tokens
        self.train_split = train_split
        self.split_seed = split_seed
        self.init_tasks()
        self._build_split()
        # Populated by apply_train_val_test_split(); None means "use _build_split"
        self._val_idxs_per_task = None

    def get_cached_per_task_stats(self, reset=True) -> dict:
        cached_per_task_stats = copy.deepcopy(self.cached_per_task_stats)
        if reset:
            self.cached_per_task_stats = {}
        return cached_per_task_stats

    def add_long_bench_task(self, task, metric):
        bench_name, task_name = task.split('/')
        assert bench_name == 'lb'
        dataset = load_dataset('THUDM/LongBench', task_name, split='test',
                               trust_remote_code=True)
        self.lb_datasets.append(dataset)
        self.lb_tasks.append(task)
        self.lb_metrics.append(metric)

    def add_choubun_task(self, task, metric):
        bench_name, task_name = task.split('/')
        assert bench_name == 'choubun'
        dataset = load_dataset('SakanaAI/ChouBun', task_name, split='test',
                               trust_remote_code=True)
        self.lb_datasets.append(dataset)
        self.lb_tasks.append(task)
        self.lb_metrics.append(metric)

    def init_tasks(self,):
        # LongBench
        _data_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "data", "longbench")
        with open(os.path.join(_data_dir, "dataset2prompt.json")) as f:
            self.lb_task2prompt = json.load(f)
        self.lb_task2prompt = {'lb/' + t: v
                               for t, v in self.lb_task2prompt.items()}
        with open(os.path.join(_data_dir, "dataset2maxlen.json")) as f:
            self.lb_task2maxlen = json.load(f)
        self.lb_task2maxlen = {'lb/' + t: v
                               for t, v in self.lb_task2maxlen.items()}
        self.lb_taskstopgen = {t: [] for t in self.lb_task2maxlen}
        self.lb_taskstopgen["lb/samsum"].append('\n')

        self.lb_dataset_per_task = {t: d for t, d in zip(
            self.lb_tasks, self.lb_datasets)}

        # unpacked utils
        self.lb_jsons_per_task = {t: [p for p in d] for t, d in zip(
            self.lb_tasks, self.lb_datasets)}

        # Filter examples exceeding max_conditioning_length
        if self.max_conditioning_length is not None:
            max_words = self.max_conditioning_length / 1.3
            for t in self.lb_jsons_per_task:
                before = len(self.lb_jsons_per_task[t])
                self.lb_jsons_per_task[t] = [
                    j for j in self.lb_jsons_per_task[t]
                    if j.get('length', 0) < max_words
                ]
                after = len(self.lb_jsons_per_task[t])
                print(f"Length filter {t}: {before} -> {after} examples")

        # Filter examples whose shortest answer exceeds max_answer_tokens
        if self.max_answer_tokens is not None:
            max_answer_words = self.max_answer_tokens / 1.3
            for t in self.lb_jsons_per_task:
                before = len(self.lb_jsons_per_task[t])
                self.lb_jsons_per_task[t] = [
                    j for j in self.lb_jsons_per_task[t]
                    if _shortest_answer_words(j) <= max_answer_words
                ]
                after = len(self.lb_jsons_per_task[t])
                if before != after:
                    print(f"Answer length filter {t}: {before} -> {after} examples "
                          f"(max_answer_tokens={self.max_answer_tokens})")

        self.lb_prompts_per_task = {}
        for task, jsons in self.lb_jsons_per_task.items():
            prompt_format = self.lb_task2prompt[task]
            self.lb_prompts_per_task[task] = []
            for json_file in jsons:
                prompt = prompt_format.format(**json_file)
                self.lb_prompts_per_task[task].append(prompt)

        self.num_prompts_per_lb_task = {k: len(
            ps) for k, ps in self.lb_prompts_per_task.items()}

        self.latest_sampled_idxs_per_lb_task = None
        self.latest_lb_tasks_names = None

    def apply_chat_template_to_prompts(self, tokenizer):
        """Wrap all eval prompts in the model's chat template.

        This ensures evaluation prompts match the format used during SFT
        training (which applies the chat template via apply_chat_template).
        The BOS token is stripped from the resulting string because the
        evaluator's tok_batch_encode adds it via add_special_tokens=True.
        """
        bos = getattr(tokenizer, 'bos_token', None) or ''
        for task in self.lb_prompts_per_task:
            wrapped = []
            for prompt in self.lb_prompts_per_task[task]:
                messages = [{"role": "user", "content": prompt}]
                text = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False,
                )
                # Strip BOS to avoid double-add (evaluator adds it separately)
                if bos and text.startswith(bos):
                    text = text[len(bos):]
                wrapped.append(text)
            self.lb_prompts_per_task[task] = wrapped
        print(f"Applied chat template to eval prompts for {len(self.lb_prompts_per_task)} tasks")

    def _build_split(self):
        """Build deterministic train/test index split for each task."""
        if self.train_split is None or self.train_split >= 1.0:
            self._train_idxs_per_task = None
            self._test_idxs_per_task = None
            return

        rng = np.random.RandomState(self.split_seed)
        self._train_idxs_per_task = {}
        self._test_idxs_per_task = {}

        for task in self.lb_jsons_per_task:
            n = len(self.lb_jsons_per_task[task])
            idxs = np.arange(n)
            rng.shuffle(idxs)
            n_train = int(n * self.train_split)
            self._train_idxs_per_task[task] = idxs[:n_train]
            self._test_idxs_per_task[task] = idxs[n_train:]
            print(f"Train/test split {task}: {n_train} train, "
                  f"{n - n_train} test ({self.train_split:.0%})")

    # ------------------------------------------------------------------
    # Deterministic train / val / test split (3-way)
    # ------------------------------------------------------------------

    def apply_train_val_test_split(self, train_frac=0.8, val_frac=0.1,
                                    max_conditioning_length=None,
                                    tokenizer=None):
        """Partition each task's prompts into deterministic train/val/test sets.

        When max_conditioning_length and tokenizer are provided, prompts
        exceeding that token length are filtered out BEFORE splitting so that
        each partition has proportional representation of usable prompts.

        After calling this method:
          - resample_requests_lb(train=True) samples only from train indices
          - resample_requests_lb(train=False) samples only from test indices
          - get_split_indices('val') returns val indices for held-out evaluation
        """
        self._train_idxs_per_task = {}
        self._val_idxs_per_task = {}
        self._test_idxs_per_task = {}

        for task_n in self.lb_tasks:
            prompts = self.lb_prompts_per_task[task_n]
            all_idxs = np.arange(len(prompts))

            # Filter out prompts that exceed max_conditioning_length
            if max_conditioning_length is not None and tokenizer is not None:
                eligible = []
                for idx in all_idxs:
                    n_tok = len(tokenizer.encode(prompts[idx], add_special_tokens=False))
                    if n_tok <= max_conditioning_length:
                        eligible.append(idx)
                n_dropped = len(all_idxs) - len(eligible)
                eligible = np.array(eligible)
            else:
                eligible = all_idxs
                n_dropped = 0

            n_eligible = len(eligible)
            n_train = int(n_eligible * train_frac)
            n_val = int(n_eligible * val_frac)
            n_test = n_eligible - n_train - n_val

            self._train_idxs_per_task[task_n] = eligible[:n_train]
            self._val_idxs_per_task[task_n] = eligible[n_train:n_train + n_val]
            self._test_idxs_per_task[task_n] = eligible[n_train + n_val:]

            print(f"  {task_n}: {len(all_idxs)} total, {n_dropped} filtered "
                  f"(>{max_conditioning_length} tok) -> "
                  f"{n_train} train / {n_val} val / {n_test} test")

    def get_split_indices(self, split='val'):
        """Return {task_name: np.array of indices} for the given split."""
        if split == 'train':
            return self._train_idxs_per_task
        elif split == 'val':
            return self._val_idxs_per_task
        elif split == 'test':
            return self._test_idxs_per_task
        else:
            raise ValueError(f"Unknown split: {split}")

    def filter_by_token_count(self, tokenizer, max_tokens):
        """Re-filter samples by actual token count using the model tokenizer.

        Call after construction to replace the approximate word-based filter
        with an exact token-based filter. Rebuilds prompts and counts.
        """
        for task in self.lb_jsons_per_task:
            jsons = self.lb_jsons_per_task[task]
            prompts = self.lb_prompts_per_task[task]
            before = len(jsons)

            # Tokenize all prompts and filter
            keep = []
            token_counts = []
            for i, prompt in enumerate(prompts):
                n_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
                if n_tokens <= max_tokens:
                    keep.append(i)
                    token_counts.append(n_tokens)

            self.lb_jsons_per_task[task] = [jsons[i] for i in keep]
            self.lb_prompts_per_task[task] = [prompts[i] for i in keep]
            after = len(keep)

            if token_counts:
                avg = sum(token_counts) / len(token_counts)
                print(f"Token filter {task}: {before} -> {after} examples "
                      f"(max={max_tokens}, avg={avg:.0f} tokens)")
            else:
                print(f"Token filter {task}: {before} -> 0 examples "
                      f"(max={max_tokens})")

        self.num_prompts_per_lb_task = {k: len(ps)
                                        for k, ps in self.lb_prompts_per_task.items()}
        self._build_split()

    def filter_answers_by_token_count(self, tokenizer, max_tokens=None):
        """Remove samples whose shortest gold answer exceeds max_tokens.

        Ensures every remaining sample can be fully generated within
        max_new_tokens budget. If max_tokens is None, uses the per-task
        max generation length from lb_task2maxlen.
        """
        for task in self.lb_jsons_per_task:
            task_max = max_tokens if max_tokens is not None else self.lb_task2maxlen.get(task)
            if task_max is None:
                continue

            jsons = self.lb_jsons_per_task[task]
            prompts = self.lb_prompts_per_task[task]
            before = len(jsons)

            keep = []
            for i, j in enumerate(jsons):
                answers = j.get("answers", [])
                if isinstance(answers, str):
                    answers = [answers]
                if not answers:
                    keep.append(i)
                    continue
                shortest = min(
                    len(tokenizer.encode(a, add_special_tokens=False))
                    for a in answers)
                if shortest <= task_max:
                    keep.append(i)

            self.lb_jsons_per_task[task] = [jsons[i] for i in keep]
            self.lb_prompts_per_task[task] = [prompts[i] for i in keep]
            after = len(keep)
            print(f"Answer token filter {task}: {before} -> {after} examples "
                  f"(max={task_max})")

        self.num_prompts_per_lb_task = {k: len(ps)
                                        for k, ps in self.lb_prompts_per_task.items()}
        self._build_split()

    def resample_requests(self, train: bool = True,
                          sampled_requests_per_task: Optional[int] = None,
                          task_batch_size: Optional[int] = None,
                          split: Optional[str] = None,
                          ) -> None:
        self.resample_requests_lb(
            train=train,
            sampled_requests_per_task=sampled_requests_per_task,
            task_batch_size=task_batch_size,
            split=split)

    def set_requests_per_task(self, requests_dict):
        self.latest_lb_tasks_names = []
        self.latest_sampled_idxs_per_lb_task = {}

        for task_n, task_idxs in requests_dict.items():
            if task_n in self.lb_tasks:
                self.latest_lb_tasks_names.append(task_n)
                self.latest_sampled_idxs_per_lb_task.update(
                    {task_n: task_idxs})
            else:
                raise ValueError(
                    'Invalid task name passed when setting task idxs')

    def get_requests_per_task(self,):
        out_dict = {}
        out_dict.update(self.latest_sampled_idxs_per_lb_task)
        return out_dict

    def resample_requests_lb(self, train: bool = True,
                             sampled_requests_per_task: Optional[int] = None,
                             task_batch_size: Optional[int] = None,
                             split: Optional[str] = None,
                             ) -> None:
        # Resolve split: explicit split parameter takes precedence over train bool
        if split is None:
            split = 'train' if train else 'test'

        if split in ('train', 'val'):
            tasks_subset = self.lb_training_tasks
        else:
            tasks_subset = self.lb_test_tasks

        if tasks_subset is not None:
            num_tasks = len(tasks_subset)
            self.latest_lb_tasks_names = tasks_subset
        else:
            self.latest_lb_tasks_names = self.lb_tasks
            num_tasks = self.num_lb_tasks

        if task_batch_size is not None and num_tasks > 0:
            tasks_idxs = np.random.choice(num_tasks, replace=False,
                                          size=task_batch_size)
            self.latest_lb_tasks_names = [self.latest_lb_tasks_names[i]
                                          for i in tasks_idxs]

        tasks_names = self.latest_lb_tasks_names

        sampled_idxs_per_lb_task = {}
        for task_n in tasks_names:
            # Use split indices if available
            if self._train_idxs_per_task is not None:
                if split == 'train':
                    eligible = self._train_idxs_per_task[task_n]
                elif split == 'val' and self._val_idxs_per_task is not None:
                    eligible = self._val_idxs_per_task[task_n]
                else:
                    eligible = self._test_idxs_per_task[task_n]
            else:
                eligible = np.arange(self.num_prompts_per_lb_task[task_n])

            if sampled_requests_per_task is not None:
                size = min(sampled_requests_per_task, len(eligible))
                sampled_idxs = np.random.choice(
                    eligible, replace=False, size=size)
            else:
                sampled_idxs = eligible
            sampled_idxs_per_lb_task[task_n] = sampled_idxs
        self.latest_sampled_idxs_per_lb_task = sampled_idxs_per_lb_task

    def evaluate(
        self,
        lm,
        train: bool = True,
        evolved_model: bool = False,
        pop_reps: int = 1,
        pop_idxs: Optional[np.array] = None,
        resample_requests: bool = True,
        sampled_requests_per_task: Optional[int] = None,
        task_batch_size: Optional[int] = None,
        limit: Optional[int] = None,
        replicate_requests: Optional[int] = None,
        build_chat_interface: bool = False,
        performance_per_request: bool = False,
        cache_param_stats_per_task: bool = False,
        model_kwargs: Optional[dict] = None,
        split: Optional[str] = None,
    ):
        # Resolve split: explicit split parameter takes precedence over train bool
        if split is None:
            split = 'train' if train else 'test'

        out_dicts = [{} for _ in range(pop_reps)]
        if split in ('train', 'val'):
            tasks_subset = self.lb_training_tasks
        else:
            tasks_subset = self.lb_test_tasks
        if len(tasks_subset) > 0:
            lb_dicts, lb_stats = self.evaluate_lb_tasks_for_pop(
                lm=lm,
                train=train,
                pop_reps=pop_reps, pop_idxs=pop_idxs,
                resample_requests=resample_requests,
                sampled_requests_per_task=sampled_requests_per_task,
                tasks_subset=tasks_subset, task_batch_size=task_batch_size,
                limit=limit, build_chat_interface=build_chat_interface,
                performance_per_request=performance_per_request,
                cache_param_stats_per_task=cache_param_stats_per_task,
                model_kwargs=model_kwargs,
                split=split)

            out_dicts = merge_list_of_dicts(out_dicts, lb_dicts)
            out_dicts = merge_list_of_dicts(out_dicts, lb_stats)

        return out_dicts

    def get_latest_sampled_idxs(self, train=True, split=None):
        lb_tasks_names = self.latest_lb_tasks_names
        # Resolve split: explicit split parameter takes precedence over train bool
        if split is None:
            split = 'train' if train else 'test'

        if split in ('train', 'val'):
            tasks_subset = self.lb_training_tasks
        else:
            tasks_subset = self.lb_test_tasks
        all_idxs = {}
        if lb_tasks_names is not None:
            lb_tasks_names = [t_n for t_n in lb_tasks_names
                              if t_n in tasks_subset]
            for task_n in lb_tasks_names:
                sampled_idxs = self.latest_sampled_idxs_per_lb_task[task_n]
                if sampled_idxs is None:
                    all_idxs[task_n] = np.arange(
                        self.num_prompts_per_lb_task[task_n])
                else:
                    all_idxs[task_n] = sampled_idxs

        return all_idxs

    def evaluate_lb_tasks_for_pop(
        self,
        lm,
        train: bool = False,
        pop_reps: int = 1,
        pop_idxs: Optional[np.array] = None,
        resample_requests: bool = True,
        sampled_requests_per_task: Optional[int] = None,
        tasks_subset: Optional[list] = None,
        task_batch_size: Optional[int] = None,
        # only used for debugging in the absence of sampled_requests_per_task
        limit: Optional[int] = None,
        use_cached_kv_if_available: bool = True,
        build_chat_interface: bool = False,
        performance_per_request: bool = False,
        cache_param_stats_per_task: bool = False,
        model_kwargs: Optional[dict] = None,
        split: Optional[str] = None,
    ):

        # Resolve split: explicit split parameter takes precedence over train bool
        if split is None:
            split = 'train' if train else 'test'

        model_kwargs = model_kwargs or {}
        stats = [{} for _ in range(pop_reps)]

        if resample_requests:
            if tasks_subset is not None:
                num_tasks = len(tasks_subset)
                self.latest_lb_tasks_names = tasks_subset
            else:
                self.latest_lb_tasks_names = self.lb_tasks
                num_tasks = self.num_lb_tasks

            if task_batch_size is not None:
                tasks_idxs = np.random.choice(num_tasks, replace=False,
                                              size=task_batch_size)
                self.latest_lb_tasks_names = [self.latest_lb_tasks_names[i]
                                              for i in tasks_idxs]

        model_kwargs = dict(pop_reps=pop_reps, pop_idxs=pop_idxs,
                            model_kwargs=model_kwargs)

        tasks_names = self.latest_lb_tasks_names

        sampled_idxs_per_lb_task = {}
        sampled_task_prompts = {}
        sampled_task_jsons = {}
        for task_n in tasks_names:
            task_prompts = self.lb_prompts_per_task[task_n]
            task_jsons = self.lb_jsons_per_task[task_n]

            # Get eligible indices for this split
            if self._train_idxs_per_task is not None:
                if split == 'train':
                    eligible = self._train_idxs_per_task[task_n]
                elif split == 'val' and self._val_idxs_per_task is not None:
                    eligible = self._val_idxs_per_task[task_n]
                else:
                    eligible = self._test_idxs_per_task[task_n]
            else:
                eligible = None  # no split, use all

            if not resample_requests:
                sampled_idxs = self.latest_sampled_idxs_per_lb_task[task_n]
                prompts = [task_prompts[i] for i in sampled_idxs]
                jsons = [task_jsons[i] for i in sampled_idxs]

            elif sampled_requests_per_task is not None:
                if eligible is not None:
                    size = min(sampled_requests_per_task, len(eligible))
                    sampled_idxs = np.random.choice(
                        eligible, replace=False, size=size)
                else:
                    sampled_idxs = np.random.choice(
                        len(task_prompts), replace=False,
                        size=sampled_requests_per_task)
                prompts = [task_prompts[i] for i in sampled_idxs]
                jsons = [task_jsons[i] for i in sampled_idxs]
            else:
                if eligible is not None:
                    sampled_idxs = eligible
                    prompts = [task_prompts[i] for i in eligible]
                    jsons = [task_jsons[i] for i in eligible]
                else:
                    sampled_idxs = None
                    if limit is not None:
                        prompts = task_prompts[:limit]
                        jsons = task_jsons[:limit]
                    else:
                        prompts = task_prompts
                        jsons = task_jsons
            sampled_idxs_per_lb_task[task_n] = sampled_idxs

            sampled_task_jsons[task_n] = jsons
            sampled_task_prompts[task_n] = prompts

        self.latest_sampled_idxs_per_lb_task = sampled_idxs_per_lb_task

        resps_per_task = {}
        pop_task_scores = [{} for _ in range(pop_reps)]
        if performance_per_request:
            for pop_i in range(pop_reps):
                stats[pop_i]['performance_per_request'] = {}
        for task_n, prompts in sampled_task_prompts.items():
            if (self.prefetched_task_tensors[task_n] is not None
                    and use_cached_kv_if_available):
                raise NotImplementedError

            build_chat_interface_for_task = False
            if build_chat_interface:
                dataset_n = task_n.split('/')[1]
                if dataset_n not in ["trec", "triviaqa", "samsum", "lsht",
                                     "lcc", "repobench-p"]:
                    build_chat_interface_for_task = True

            all_classes = None

            jsons = sampled_task_jsons[task_n]
            if len(prompts) == 0:
                print(f"  [tasks] skipping {task_n}: no eligible prompts in this split")
                continue

            task_kwargs = dict(
                max_gen_tokens=self.lb_task2maxlen[task_n],
                stop_gen=self.lb_taskstopgen[task_n],
                build_chat_interface=build_chat_interface_for_task)

            task_outputs = lm.evaluate_lb(dataset_samples=prompts,
                                          **task_kwargs, **model_kwargs)
            task_outputs = task_outputs
            n_task_outputs = len(task_outputs)
            n_outputs_per_pop_idx = n_task_outputs // pop_reps

            assert n_outputs_per_pop_idx == len(jsons)

            task_ouputs_per_pop_idx = [
                task_outputs[i:i + n_outputs_per_pop_idx]
                for i in range(0, n_task_outputs, n_outputs_per_pop_idx)]
            dicts_to_store = []
            has_length = False

            for j in range(pop_reps):
                prediction_list, answers_list, length_list = [], [], []
                for i, (json_obj, prompt) in enumerate(zip(jsons, prompts)):
                    all_classes = json_obj["all_classes"]
                    answers = json_obj["answers"]
                    if "length" in json_obj:
                        length = json_obj["length"]
                        has_length = True
                    else:
                        length = -1
                        assert not has_length

                    pred = task_ouputs_per_pop_idx[j][i]
                    prediction_list.append(pred)
                    answers_list.append(answers)
                    length_list.append(length)

                    if self.store_gen_outputs:
                        prompt_dict = dict(
                            pred=pred,
                            answers=answers,
                            all_classes=all_classes,
                            length=length,
                        )
                        if pop_idxs is not None:
                            prompt_dict['pop_idx'] = pop_idxs[j]
                        dicts_to_store.append(prompt_dict)

                score, all_scores = get_score(
                    task=task_n[task_n.index('/') + 1:],  # strip task prefix
                    predictions=prediction_list,
                    answers=answers_list,
                    all_classes=all_classes)

                pop_task_scores[j][task_n] = score
                if performance_per_request:
                    if sampled_idxs_per_lb_task[task_n] is None:
                        if limit is None:
                            sampled_prompt_idxs = list(range(len(all_scores)))
                        else:
                            sampled_prompt_idxs = list(range(limit))
                    else:
                        sampled_prompt_idxs = sampled_idxs_per_lb_task[task_n]
                    assert (len(sampled_prompt_idxs) ==
                            len(all_scores))
                    stats[j]['performance_per_request'][task_n] = {
                        prompt_idx: prompt_score for prompt_idx, prompt_score in
                        zip(sampled_prompt_idxs, all_scores)}
            if cache_param_stats_per_task:
                memory_policy_stats = lm.model.get_param_stats()
                for k, v in memory_policy_stats.items():
                    self.cached_per_task_stats[
                        f'{task_n[task_n.index("/") + 1:]}/' + k] = v
            if self.store_gen_outputs:
                pass

        return pop_task_scores, stats
