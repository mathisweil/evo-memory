#!/usr/bin/env python3
"""
make_report.py — generate a human-readable markdown report from eval JSONL outputs.

Reads the JSONL files saved by run_eval.py into results/{method}/{seed}/eval_outputs/
and writes a report.md alongside them.

Usage:
  # Single run report:
  python make_report.py results/base_instruct_fc/1337

  # Compare multiple runs side by side:
  python make_report.py results/base_instruct_fc/1337 results/m1_sft/1337 results/namm_instruct/1337

  # Limit examples shown per task:
  python make_report.py results/base_instruct_fc/1337 --n 10
"""
import argparse
import json
import os
import sys


TASKS = ['lb_qasper', 'lb_narrativeqa', 'lb_passage_retrieval_en']
TASK_DISPLAY = {
    'lb_qasper': 'QASPER',
    'lb_narrativeqa': 'NarrativeQA',
    'lb_passage_retrieval_en': 'Passage Retrieval',
}


def qa_f1_score(pred: str, answers: list) -> float:
    """Token-level F1, same logic as LongBench scorer."""
    import re
    from collections import Counter

    def normalize(s):
        s = s.lower()
        s = re.sub(r'\b(a|an|the)\b', ' ', s)
        s = re.sub(r'[^a-z0-9 ]', '', s)
        return s.split()

    pred_tokens = normalize(pred)
    if not pred_tokens:
        return 0.0
    best = 0.0
    for ans in answers:
        ans_tokens = normalize(ans)
        if not ans_tokens:
            continue
        common = Counter(pred_tokens) & Counter(ans_tokens)
        n_common = sum(common.values())
        if n_common == 0:
            continue
        p = n_common / len(pred_tokens)
        r = n_common / len(ans_tokens)
        f1 = 2 * p * r / (p + r)
        best = max(best, f1)
    return best


def load_jsonl(path: str) -> list:
    records = []
    if not os.path.exists(path):
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def score_file(artifact_dir: str, task: str) -> str:
    p = os.path.join(artifact_dir, 'eval_outputs', f'{task[3:]}_score.txt')
    if os.path.exists(p):
        return open(p).read().strip()
    return 'N/A'


def run_label(artifact_dir: str) -> str:
    """Short label from path, e.g. results/base_instruct_fc/1337 -> base_instruct_fc/1337"""
    parts = artifact_dir.rstrip('/').split('/')
    if len(parts) >= 2:
        return '/'.join(parts[-2:])
    return artifact_dir


def write_report(artifact_dirs: list, n: int, out_path: str):
    lines = []
    lines.append('# Eval Output Report\n')

    # Summary table
    lines.append('## Score Summary\n')
    header = '| Run | ' + ' | '.join(TASK_DISPLAY[t] for t in TASKS) + ' |'
    sep = '|' + '|'.join(['---'] * (len(TASKS) + 1)) + '|'
    lines.append(header)
    lines.append(sep)
    for ad in artifact_dirs:
        scores = [score_file(ad, t) for t in TASKS]
        lines.append('| ' + run_label(ad) + ' | ' + ' | '.join(scores) + ' |')
    lines.append('')

    # Per-task output comparison
    lines.append('## Per-Example Outputs\n')
    for task in TASKS:
        lines.append(f'### {TASK_DISPLAY[task]}\n')

        # Load records for each run
        all_records = {}
        for ad in artifact_dirs:
            jsonl_path = os.path.join(ad, 'eval_outputs', f'{task}.jsonl')
            recs = load_jsonl(jsonl_path)
            if recs:
                all_records[run_label(ad)] = recs

        if not all_records:
            lines.append('_No outputs saved for this task._\n')
            continue

        # Use first run as reference for question/answers
        ref_label = list(all_records.keys())[0]
        ref_recs = all_records[ref_label]
        n_show = min(n, len(ref_recs))

        for i in range(n_show):
            ref = ref_recs[i]
            question = ref.get('input', '').strip()
            answers = ref.get('answers', [])
            if isinstance(answers, str):
                answers = [answers]
            ans_str = ' | '.join(answers) if answers else '_(none)_'

            lines.append(f'#### Example {i + 1}')
            if question:
                lines.append(f'**Question:** {question}\n')
            lines.append(f'**Reference answer(s):** {ans_str}\n')

            for label, recs in all_records.items():
                if i < len(recs):
                    pred = recs[i].get('pred', '').strip()
                    f1 = qa_f1_score(pred, answers)
                    lines.append(f'**{label}** (F1: {f1:.2f})')
                    lines.append(f'> {pred}')
                    lines.append('')
            lines.append('---\n')

    report = '\n'.join(lines)
    with open(out_path, 'w') as f:
        f.write(report)
    print(f'Report written to: {out_path}')


def main():
    p = argparse.ArgumentParser(description='Generate markdown report from eval JSONL outputs')
    p.add_argument('artifact_dirs', nargs='+', help='One or more artifact dirs (results/method/seed)')
    p.add_argument('--n', type=int, default=20, help='Max examples to show per task (default: 20)')
    p.add_argument('--out', default=None, help='Output path (default: report.md in first artifact dir)')
    args = p.parse_args()

    out_path = args.out or os.path.join(args.artifact_dirs[0], 'report.md')
    write_report(args.artifact_dirs, args.n, out_path)


if __name__ == '__main__':
    main()
