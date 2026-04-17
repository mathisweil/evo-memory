"""Per-prompt case study: attention-entropy trajectory across chunks, per-layer KV ghost distortion, and aligned predictions from all conditions.

For each case, shows:
  Row 1: Attention entropy trajectory across chunks (the collapse story)
  Row 2: Per-layer KV cosine (ghost distortion) + per-position ghost at key layers
  Row 3: Predictions from all conditions + gold answer + summary stats

Runs NAMM forward pass with per-chunk attention hooks to capture the collapse.
Predictions are loaded from existing eval results (no generation needed).

Usage:
    python scripts/reporting/case_study_entropy.py \\
        --namm_checkpoint <path> --cache_size 1024 \\
        --output_dir analysis_out/case_studies
"""

import argparse
import json
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import hydra
from hydra import compose, initialize
from namm.run_utils import make_eval_model, make_task_sampler
from es_finetuning.device import get_device


CASES = [
    # M4 fails, Trunc succeeds
    {"task": "lb/qasper_e", "idx": 126, "split": "extended_test",
     "title": "FAIL: Ghost causes wrong answer (citation missed)"},
    {"task": "lb/2wikimqa", "idx": 95, "split": "extended_test",
     "title": "FAIL: Ghost causes cross-passage hallucination"},
    # M4 succeeds, Trunc fails (NAMM helps)
    {"task": "lb/2wikimqa", "idx": 147, "split": "test",
     "title": "SUCCESS: NAMM retains answer Trunc misses"},
    {"task": "lb/2wikimqa", "idx": 155, "split": "test",
     "title": "SUCCESS: NAMM retains 'Milan' from middle of doc"},
    {"task": "lb/hotpotqa_e", "idx": 88, "split": "test",
     "title": "SUCCESS: NAMM gets yes/no right, Trunc inverts"},
]


class ChunkAttentionCapture:
    """Captures per-chunk attention entropy from every layer."""
    def __init__(self, model):
        self.chunks = []  # list of dicts: {layer: (n_heads, q_len, kv_len)}
        self._current = {}
        self._hooks = []
        for i, layer in enumerate(model.model.layers):
            hook = layer.self_attn.register_forward_hook(self._make_hook(i))
            self._hooks.append(hook)

    def _make_hook(self, layer_idx):
        def fn(module, input, output):
            attn = output[1]
            if attn is not None:
                self._current[layer_idx] = attn[0].detach().cpu().float()
        return fn

    def reset(self):
        self.chunks = []
        self._current = {}

    def flush(self):
        if self._current:
            self.chunks.append(dict(self._current))
            self._current = {}

    def remove(self):
        for h in self._hooks:
            h.remove()


def load_predictions():
    """Load predictions from existing eval results."""
    PROJ = REPO_ROOT
    preds = {}  # (split, task, idx) → {"M4": pred, "Trunc": pred, ...}

    cond_dirs = {
        "M4 LoRA+NAMM": "lora_m4_cs1024_5t",
        "Trunc LoRA": "trunc_lora_m1_1024_5t",
        "M1 full": "lora_m1_5t",
    }
    for label, dirname in cond_dirs.items():
        gen_files = sorted(glob.glob(f"{PROJ}/eval_results/{dirname}/*/generations.json"))
        if not gen_files:
            continue
        gen = json.load(open(gen_files[-1]))
        for split in ["test", "extended_test"]:
            if split not in gen:
                continue
            for task, entries in gen[split].items():
                for e in entries:
                    key = (split, task, e["prompt_idx"])
                    if key not in preds:
                        preds[key] = {"gold": e["answers"][0] if isinstance(e["answers"], list) and e["answers"] else str(e.get("answers",""))}
                    preds[key][label] = e["pred"].strip()

    # Load F1
    f1s = {}
    for label, dirname in cond_dirs.items():
        files = sorted(glob.glob(f"{PROJ}/eval_results/{dirname}/*/results.json"))
        if not files:
            continue
        r = json.load(open(files[-1]))
        sps = r.get("scores_per_split", r.get("results", {}))
        for split in ["test", "extended_test"]:
            s = sps.get(split, {})
            for task, prompts in s.get("per_prompt_f1", {}).items():
                for idx_str, f1 in prompts.items():
                    key = (split, task, int(idx_str))
                    if key not in f1s:
                        f1s[key] = {}
                    f1s[key][label] = f1
    return preds, f1s


def extract_kv(past_key_values):
    kvs = []
    for k, v in past_key_values:
        kvs.append((k[0].cpu().float(), v[0].cpu().float()))
    return kvs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--namm_checkpoint", type=str, required=True)
    p.add_argument("--cache_size", type=int, default=1024)
    p.add_argument("--run_config", type=str, default="namm_bam_i1_llama32_1b_5t")
    p.add_argument("--filter_by_length", type=int, default=8192)
    p.add_argument("--output_dir", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = get_device()

    overrides = [
        f"run@_global_={args.run_config}",
        "wandb_log=false", "wandb_project=Experiments",
        f"filter_by_length={args.filter_by_length}",
        f"cache_size={args.cache_size}", f"max_memory_length={args.cache_size}",
        "+protected_tail_n=5",
    ]
    with initialize(version_base=None, config_path="../config",
                    job_name="case_study_v2"):
        cfg = compose(config_name="config", overrides=overrides)

    print("Building model...")
    with torch.no_grad():
        (memory_policy, memory_model, memory_evaluator,
         _evo, _aux) = make_eval_model(cfg=cfg)
    memory_model.to(device)

    ckpt = torch.load(args.namm_checkpoint, map_location="cpu", weights_only=False)
    evo = ckpt['evolution_state']
    params_vec = evo.get('mean', evo['best_member'])
    memory_model.set_memory_params(params_vec.unsqueeze(0).to(device))
    bp = 'stored_buffers_to_save.'
    bd = {k[len(bp):]: v.to(device) for k, v in evo.items() if k.startswith(bp)}
    if bd:
        memory_model.load_buffers_dict(buffers_dict=bd)
    memory_policy.set_params_batch_idxs(np.zeros([1]))
    memory_policy.record_eval_stats = True
    memory_policy.initialize_stat_objects()

    capture = ChunkAttentionCapture(memory_model)
    orig_fwd = memory_model.forward
    def patched_fwd(*a, **kw):
        r = orig_fwd(*a, **kw)
        capture.flush()
        return r
    memory_model.forward = patched_fwd

    # Task sampler
    task_sampler = make_task_sampler(cfg=cfg)
    tokenizer = hydra.utils.call(cfg.tokenizer)
    max_answer_tok = cfg.get('max_answer_tokens', cfg.get('max_new_tokens', 64))
    task_sampler.filter_answers_by_token_count(tokenizer, max_answer_tok)
    task_sampler.apply_train_val_test_split(
        train_frac=cfg.get('train_frac', 0.7),
        val_frac=cfg.get('val_frac', 0.15),
        max_conditioning_length=cfg.get('split_max_conditioning_length',
                                        cfg.get('max_conditioning_length', 6500)),
        min_conditioning_length=cfg.get('min_conditioning_length', None),
        tokenizer=tokenizer,
        extended_max_conditioning_length=8192,
    )
    task_sampler.apply_chat_template_to_prompts(tokenizer)
    raw_prompts = task_sampler.lb_prompts_per_task

    # Load predictions from eval results
    preds, f1s = load_predictions()

    for case in CASES:
        task = case["task"]
        idx = case["idx"]
        split = case["split"]
        print(f"\n{'='*70}")
        print(f"  {case['title']}: {task} idx={idx} ({split})")
        print(f"{'='*70}")

        prompt = raw_prompts[task][idx]
        enc = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        n_tok = int(input_ids.shape[-1])

        # Get predictions
        key = (split, task, idx)
        pred_info = preds.get(key, {})
        f1_info = f1s.get(key, {})

        # === NAMM forward with chunk capture ===
        memory_policy.initialize_stat_objects()
        capture.reset()
        with torch.no_grad():
            out_namm = memory_model(
                input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=True, output_attentions=True,
                use_cache=True, apply_memory_policy=True)
        capture.flush()
        kv_namm = extract_kv(out_namm.past_key_values)

        # === Truncation forward ===
        trunc_ids = input_ids[..., -args.cache_size:]
        trunc_mask = attention_mask[..., -args.cache_size:]
        memory_policy.initialize_stat_objects()
        capture_trunc_chunks = []
        # For truncation, no split processing, so capture directly
        trunc_attn = {}
        def trunc_hook(layer_idx):
            def fn(module, input, output):
                a = output[1]
                if a is not None:
                    trunc_attn[layer_idx] = a[0].detach().cpu().float()
            return fn
        trunc_hooks = []
        for i, layer in enumerate(memory_model.model.layers):
            trunc_hooks.append(layer.self_attn.register_forward_hook(trunc_hook(i)))
        with torch.no_grad():
            out_trunc = memory_model(
                input_ids=trunc_ids, attention_mask=trunc_mask,
                output_hidden_states=True, output_attentions=True,
                use_cache=True, apply_memory_policy=True)
        for h in trunc_hooks:
            h.remove()
        kv_trunc = extract_kv(out_trunc.past_key_values)

        # === Compute metrics ===
        n_layers = len(kv_namm)
        n_chunks = len(capture.chunks)

        # Per-chunk entropy for selected layers
        show_layers = [0, 3, 7, 11, 15]
        entropy_per_layer = {l: [] for l in show_layers}
        kv_lens = []
        for chunk in capture.chunks:
            kv_len = 0
            for l in show_layers:
                if l in chunk:
                    a = chunk[l]  # (n_heads, q_len, kv_len)
                    last_row = a[:, -1, :]
                    ent = -(last_row * torch.log(last_row.clamp(min=1e-10))).sum(-1).mean().item()
                    entropy_per_layer[l].append(ent)
                    kv_len = a.shape[-1]
            kv_lens.append(kv_len)

        # Per-layer KV cosine
        key_cos_per_layer = []
        for l in range(n_layers):
            kn, vn = kv_namm[l]
            kt, vt = kv_trunc[l]
            shared = min(kn.shape[1], kt.shape[1])
            kn_s = kn[:, -shared:, :]
            kt_s = kt[:, -shared:, :]
            kcos = F.cosine_similarity(
                kn_s.reshape(-1, kn_s.shape[-1]),
                kt_s.reshape(-1, kt_s.shape[-1])
            ).reshape(kn_s.shape[0], shared).mean(0).mean().item()
            key_cos_per_layer.append(kcos)

        # === FIGURE ===
        fig = plt.figure(figsize=(16, 14))
        gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3,
                      height_ratios=[1, 1, 0.6])

        # Row 1 left: Entropy trajectory
        ax1 = fig.add_subplot(gs[0, 0])
        colors_layers = {0: '#1f77b4', 3: '#ff7f0e', 7: '#2ca02c', 11: '#d62728', 15: '#9467bd'}
        for l in show_layers:
            if entropy_per_layer[l]:
                ax1.plot(entropy_per_layer[l], label=f'Layer {l}',
                         color=colors_layers[l], linewidth=1.5)
        # Add uniform entropy reference
        if kv_lens:
            max_kv = max(kv_lens)
            ax1.axhline(np.log(max_kv), color='gray', linestyle=':', linewidth=1,
                        label=f'Uniform (log {max_kv})')
        # Mark where eviction starts
        for i, kvl in enumerate(kv_lens):
            if i > 0 and kvl == kv_lens[i-1] and kvl > 0:
                ax1.axvline(i, color='red', linestyle='--', alpha=0.5, linewidth=1)
                ax1.text(i, ax1.get_ylim()[0] + 0.2, 'eviction\nstarts',
                         fontsize=7, color='red', ha='center')
                break
        ax1.set_xlabel("Chunk index (each = 256 tokens)")
        ax1.set_ylabel("Attention entropy (last token)")
        ax1.set_title("Attention entropy collapses after eviction")
        ax1.legend(fontsize=7, loc='center right')
        ax1.grid(alpha=0.3)

        # Row 1 right: Truncation attention vs NAMM attention (layer 7, last chunk)
        ax2 = fig.add_subplot(gs[0, 1])
        # Truncation attention
        if 7 in trunc_attn:
            attn_t = trunc_attn[7][:, -1, :].mean(0).numpy()
            ax2.plot(np.arange(len(attn_t)), attn_t, color='#2ca02c',
                     linewidth=0.8, alpha=0.8, label='Truncation')
        # NAMM attention (last chunk)
        if capture.chunks and 7 in capture.chunks[-1]:
            attn_n = capture.chunks[-1][7][:, -1, :].mean(0).numpy()
            ax2.plot(np.arange(len(attn_n)), attn_n, color='#d62728',
                     linewidth=0.8, alpha=0.8, label='NAMM (uniform)')
        ax2.set_xlabel("KV cache position")
        ax2.set_ylabel("Attention weight")
        ax2.set_title("Layer 7: Last-token attention (NAMM = flat, Trunc = structured)")
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)

        # Row 2 left: Per-layer KV cosine
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.bar(range(n_layers), key_cos_per_layer, color='#d62728',
                edgecolor='black', linewidth=0.3, alpha=0.7)
        ax3.axhline(0, color='gray', linewidth=0.8)
        ax3.set_xlabel("Layer")
        ax3.set_ylabel("Key cosine (NAMM vs Trunc)")
        ax3.set_title("Ghost distortion per layer (negative = keys inverted)")
        ax3.grid(axis='y', alpha=0.3)

        # Row 2 right: Entropy for NAMM vs truncation at layer 7
        ax4 = fig.add_subplot(gs[1, 1])
        # Truncation entropy per chunk
        trunc_entropy_l7 = []
        if 7 in trunc_attn:
            a = trunc_attn[7]  # (n_heads, q_len, kv_len) — single chunk
            # Compute entropy for each query position (not just last)
            for q in range(a.shape[1]):
                row = a[:, q, :]
                ent = -(row * torch.log(row.clamp(min=1e-10))).sum(-1).mean().item()
                trunc_entropy_l7.append(ent)
        # NAMM entropy per chunk (already computed)
        namm_entropy_l7 = entropy_per_layer.get(7, [])

        if namm_entropy_l7:
            ax4.plot(namm_entropy_l7, 'o-', color='#d62728', markersize=3,
                     linewidth=1.5, label='NAMM (per chunk)')
        if trunc_entropy_l7:
            # Show as horizontal band (truncation has 1 chunk, but many query positions)
            ax4.axhline(np.mean(trunc_entropy_l7), color='#2ca02c', linewidth=2,
                        linestyle='--', label=f'Truncation (mean={np.mean(trunc_entropy_l7):.1f})')
            ax4.fill_between(range(len(namm_entropy_l7)),
                             np.min(trunc_entropy_l7), np.max(trunc_entropy_l7),
                             alpha=0.15, color='#2ca02c')
        if kv_lens:
            max_kv = max(kv_lens)
            ax4.axhline(np.log(max_kv), color='gray', linestyle=':', linewidth=1,
                         label=f'Uniform (log {max_kv})')
        ax4.set_xlabel("Chunk index (NAMM) / query position (Trunc)")
        ax4.set_ylabel("Attention entropy (layer 7)")
        ax4.set_title("NAMM collapses to uniform; Truncation stays sharp")
        ax4.legend(fontsize=7)
        ax4.grid(alpha=0.3)

        # Row 3: Predictions text box
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')

        gold = pred_info.get("gold", "?")
        lines = [
            f"Task: {task}  |  Prompt idx: {idx}  |  Split: {split}  |  "
            f"Length: {n_tok} tokens  |  Cache: {args.cache_size}  |  "
            f"Chunks: {n_chunks}",
            "",
            f"Gold answer:     {gold[:90]}",
            f"M4 LoRA+NAMM:    {pred_info.get('M4 LoRA+NAMM', '?')[:90]}"
            f"   (F1={f1_info.get('M4 LoRA+NAMM', 0)*100:.1f})",
            f"Trunc LoRA:      {pred_info.get('Trunc LoRA', '?')[:90]}"
            f"   (F1={f1_info.get('Trunc LoRA', 0)*100:.1f})",
            f"M1 full cache:   {pred_info.get('M1 full', '?')[:90]}"
            f"   (F1={f1_info.get('M1 full', 0)*100:.1f})",
            "",
            f"Mean key cosine: {np.mean(key_cos_per_layer):.3f}  |  "
            f"Chunks before collapse: {sum(1 for e in entropy_per_layer[7] if e < 6.0)}/"
            f"{len(entropy_per_layer[7])}  |  "
            f"Trunc entropy (L7): {np.mean(trunc_entropy_l7):.1f}" if trunc_entropy_l7 else
            f"Mean key cosine: {np.mean(key_cos_per_layer):.3f}  |  "
            f"Chunks before collapse: {sum(1 for e in entropy_per_layer[7] if e < 6.0)}/"
            f"{len(entropy_per_layer[7])}",
        ]
        text = "\n".join(lines)
        ax5.text(0.02, 0.95, text, transform=ax5.transAxes,
                 fontsize=9.5, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        fig.suptitle(f"{case['title']}\n{task.replace('lb/','')} idx={idx}",
                     fontsize=13, fontweight='bold')

        out_path = os.path.join(args.output_dir,
                                f"case_v2_{task.replace('/','_')}_{idx}_{split}.png")
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_path}")

        del out_namm, out_trunc, kv_namm, kv_trunc
        torch.cuda.empty_cache()

    capture.remove()
    print("\nDone.")


if __name__ == "__main__":
    main()
