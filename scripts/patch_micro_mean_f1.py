"""Add micro_mean_f1 + n_prompts_total + n_prompts_per_task to existing
results.json files in-place, computed from per_prompt_f1.

Walks results/main_table_5t/**/results.json and eval_results/**/results.json
by default. Idempotent — re-running is a no-op.

micro_mean_f1 = mean of all per-prompt F1s (raw 0-1) * 100, the prompt-count
weighted average that matches the val_lb_avg_f1 metric used during LoRA
training. Macro mean_f1 (1/n_tasks per task) is left unchanged.

Usage:
    python scripts/patch_micro_mean_f1.py [--dry-run] [--root path/to/dir]
"""
import argparse
import glob
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def patch_file(path, dry_run=False):
    with open(path) as f:
        d = json.load(f)
    typ = d.get('type')
    if typ == 'plain_llama_baseline':
        container = d.get('results')
    elif typ == 'eval_namm_splits':
        container = d.get('scores_per_split')
    else:
        return None  # unknown schema, skip
    if not container:
        return None

    changed_splits = []
    for split_name, split_scores in container.items():
        ppf1 = split_scores.get('per_prompt_f1')
        if not ppf1:
            continue
        all_prompt_scores = []
        n_per_task = {}
        for task, task_dict in ppf1.items():
            vals = list(task_dict.values())
            all_prompt_scores.extend(vals)
            n_per_task[task] = len(vals)
        if not all_prompt_scores:
            continue
        micro = sum(all_prompt_scores) / len(all_prompt_scores) * 100.0
        new_micro = round(micro, 6)
        old_micro = split_scores.get('micro_mean_f1')
        old_total = split_scores.get('n_prompts_total')
        old_per_task = split_scores.get('n_prompts_per_task')
        if (old_micro == new_micro
                and old_total == len(all_prompt_scores)
                and old_per_task == n_per_task):
            continue  # already up-to-date
        split_scores['micro_mean_f1'] = new_micro
        split_scores['n_prompts_total'] = len(all_prompt_scores)
        split_scores['n_prompts_per_task'] = n_per_task
        changed_splits.append(
            (split_name, new_micro, len(all_prompt_scores)))

    if changed_splits and not dry_run:
        with open(path, 'w') as f:
            json.dump(d, f, indent=2)
    return changed_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=str, default=None,
                        help='Root directory to search for results.json. '
                             'Defaults to results/main_table_5t and '
                             'eval_results.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would change without writing.')
    args = parser.parse_args()

    if args.root:
        roots = [args.root]
    else:
        roots = [
            os.path.join(REPO_ROOT, 'results', 'main_table_5t'),
            os.path.join(REPO_ROOT, 'eval_results'),
        ]
    files = []
    for r in roots:
        if not os.path.isdir(r):
            continue
        files.extend(sorted(glob.glob(
            os.path.join(r, '**', 'results.json'), recursive=True)))
    if not files:
        print(f'No results.json files found under: {roots}')
        return 1

    print(f'Scanning {len(files)} results.json files'
          f'{" (dry run)" if args.dry_run else ""}...')
    n_changed = 0
    n_skipped = 0
    for fpath in files:
        rel = os.path.relpath(fpath, REPO_ROOT)
        try:
            changed = patch_file(fpath, dry_run=args.dry_run)
        except Exception as e:
            print(f'  ERROR {rel}: {e}')
            continue
        if changed is None:
            print(f'  SKIP  {rel}: unrecognized schema')
            n_skipped += 1
        elif not changed:
            print(f'  ok    {rel}: already up-to-date')
        else:
            n_changed += 1
            for split, micro, n in changed:
                print(f'  PATCH {rel} [{split}]: '
                      f'micro_mean_f1={micro:.4f} (n={n})')

    print(f'\nDone. Patched {n_changed} files, skipped {n_skipped}.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
