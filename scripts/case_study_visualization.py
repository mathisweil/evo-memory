"""Generate per-prompt case study visualizations comparing NAMM vs truncation.

For each specified prompt, runs forward passes under NAMM eviction and
truncation, captures the final-token attention weights across all layers,
and produces:
  1. Attention heatmap: where does the last token attend under each condition?
  2. KV cosine comparison: per-position ghost distortion
  3. Text annotation: which parts of the prompt are retained/evicted

Usage:
    See _claude_case_study_viz.sh
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
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
from datasets import load_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--namm_checkpoint", type=str, required=True)
    p.add_argument("--lora_m4", type=str, default=None)
    p.add_argument("--cache_size", type=int, required=True)
    p.add_argument("--run_config", type=str, default="namm_bam_i1_llama32_1b_5t")
    p.add_argument("--filter_by_length", type=int, default=8192)
    p.add_argument("--cases", nargs="+", required=True,
                   help="task:orig_idx:split, e.g. lb/qasper_e:126:extended_test")
    p.add_argument("--output_dir", type=str, required=True)
    return p.parse_args()


class AttentionCapture:
    def __init__(self, model):
        self.attention_per_layer = {}
        self._hooks = []
        for i, layer in enumerate(model.model.layers):
            hook = layer.self_attn.register_forward_hook(self._make_hook(i))
            self._hooks.append(hook)

    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            attn = output[1]
            if attn is not None:
                self.attention_per_layer[layer_idx] = attn[0].detach().cpu().float()
        return hook_fn

    def reset(self):
        self.attention_per_layer = {}

    def remove(self):
        for h in self._hooks:
            h.remove()


def extract_kv(past_key_values):
    kvs = []
    for k, v in past_key_values:
        kvs.append((k[0].cpu().float(), v[0].cpu().float()))
    return kvs


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
        f"cache_size={args.cache_size}",
        f"max_memory_length={args.cache_size}",
        "+protected_tail_n=5",
    ]

    with initialize(version_base=None, config_path="../config",
                    job_name="case_study_viz"):
        cfg = compose(config_name="config", overrides=overrides)

    print("Building model...")
    with torch.no_grad():
        (memory_policy, memory_model, memory_evaluator,
         _evo, _aux) = make_eval_model(cfg=cfg)
    memory_model.to(device)

    # Load NAMM
    ckpt = torch.load(args.namm_checkpoint, map_location="cpu", weights_only=False)
    evo_state = ckpt['evolution_state']
    params_vec = evo_state.get('mean', evo_state['best_member'])
    memory_model.set_memory_params(params_vec.unsqueeze(0).to(device))
    buffers_prefix = 'stored_buffers_to_save.'
    buffers_dict = {k[len(buffers_prefix):]: v.to(device)
                    for k, v in evo_state.items() if k.startswith(buffers_prefix)}
    if buffers_dict:
        memory_model.load_buffers_dict(buffers_dict=buffers_dict)
    batch_idxs = np.zeros([1])
    memory_policy.set_params_batch_idxs(batch_idxs)
    memory_policy.record_eval_stats = True
    memory_policy.initialize_stat_objects()

    # Register hooks
    capture = AttentionCapture(memory_model)

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

    # Load LongBench for gold answers
    lb_data = {}
    prompt_templates = json.load(open(f"{REPO_ROOT}/data/longbench/dataset2prompt.json"))
    prompt_templates = {f"lb/{k}": v for k, v in prompt_templates.items()}
    for task in ["qasper", "qasper_e", "2wikimqa", "2wikimqa_e", "hotpotqa_e"]:
        lb_data[f"lb/{task}"] = load_dataset("THUDM/LongBench", task, split="test",
                                              trust_remote_code=True)

    # Process each case
    for case_str in args.cases:
        task, orig_idx_str, split = case_str.split(":")
        orig_idx = int(orig_idx_str)
        print(f"\n{'='*70}")
        print(f"  Case: {task} idx={orig_idx} split={split}")
        print(f"{'='*70}")

        prompt = raw_prompts[task][orig_idx]
        enc = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        n_tok = int(input_ids.shape[-1])
        print(f"  Prompt length: {n_tok} tokens")

        # Gold answer
        ds = lb_data[task]
        example = ds[orig_idx]
        gold = example["answers"]
        if isinstance(gold, list):
            gold = gold[0] if gold else ""
        print(f"  Gold: {repr(gold[:80])}")

        # === Condition A: NAMM eviction ===
        print("  Running NAMM forward pass...")
        memory_policy.initialize_stat_objects()
        capture.reset()
        with torch.no_grad():
            outputs_namm = memory_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True,
                use_cache=True,
                apply_memory_policy=True,
            )
        attn_namm = dict(capture.attention_per_layer)
        kv_namm = extract_kv(outputs_namm.past_key_values)
        hidden_namm = outputs_namm.hidden_states[-1][0, -1, :].cpu()

        # === Condition B: Truncation ===
        print("  Running truncation forward pass...")
        trunc_ids = input_ids[..., -args.cache_size:]
        trunc_mask = attention_mask[..., -args.cache_size:]
        memory_policy.initialize_stat_objects()
        capture.reset()
        with torch.no_grad():
            outputs_trunc = memory_model(
                input_ids=trunc_ids,
                attention_mask=trunc_mask,
                output_hidden_states=True,
                output_attentions=True,
                use_cache=True,
                apply_memory_policy=True,  # no-op since input <= cache_size
            )
        attn_trunc = dict(capture.attention_per_layer)
        kv_trunc = extract_kv(outputs_trunc.past_key_values)
        hidden_trunc = outputs_trunc.hidden_states[-1][0, -1, :].cpu()

        # === Build visualization ===
        n_layers = len(kv_namm)

        # Per-layer KV cosine (NAMM vs Trunc, shared tail positions)
        key_cosines = []
        val_cosines = []
        for l in range(n_layers):
            k_n, v_n = kv_namm[l]
            k_t, v_t = kv_trunc[l]
            shared = min(k_n.shape[1], k_t.shape[1])
            # Align from end
            kn = k_n[:, -shared:, :]; kt = k_t[:, -shared:, :]
            vn = v_n[:, -shared:, :]; vt = v_t[:, -shared:, :]
            # Per-position cosine, averaged over heads
            kcos = F.cosine_similarity(kn.reshape(-1, kn.shape[-1]),
                                        kt.reshape(-1, kt.shape[-1])).reshape(
                                            kn.shape[0], shared).mean(0)
            vcos = F.cosine_similarity(vn.reshape(-1, vn.shape[-1]),
                                        vt.reshape(-1, vt.shape[-1])).reshape(
                                            vn.shape[0], shared).mean(0)
            key_cosines.append(kcos.numpy())
            val_cosines.append(vcos.numpy())

        # Attention from last token (averaged over heads) for selected layers
        show_layers = [0, 3, 7, 11, 15]

        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

        # Top left: Per-layer key cosine (NAMM vs Trunc)
        ax1 = fig.add_subplot(gs[0, 0])
        mean_kcos = [np.mean(kc) for kc in key_cosines]
        mean_vcos = [np.mean(vc) for vc in val_cosines]
        ax1.plot(range(n_layers), mean_kcos, 'o-', color='#d62728', label='Key cosine', linewidth=2)
        ax1.plot(range(n_layers), mean_vcos, 's-', color='#1f77b4', label='Value cosine', linewidth=2)
        ax1.axhline(0, color='gray', linestyle=':', linewidth=0.8)
        ax1.set_xlabel("Layer")
        ax1.set_ylabel("Cosine similarity")
        ax1.set_title("Ghost distortion: NAMM vs Truncation KV vectors")
        ax1.legend(fontsize=9)
        ax1.grid(axis='y', linestyle=':', alpha=0.3)

        # Top right: Per-position key cosine at selected layers
        ax2 = fig.add_subplot(gs[0, 1])
        shared_len = len(key_cosines[0])
        positions = np.arange(shared_len)
        for l in [3, 7, 15]:
            # Smooth with running average
            window = min(50, shared_len // 10)
            if window > 1:
                smoothed = np.convolve(key_cosines[l], np.ones(window)/window, mode='valid')
                ax2.plot(np.arange(len(smoothed)), smoothed, label=f'Layer {l}', linewidth=1.5)
            else:
                ax2.plot(positions, key_cosines[l], label=f'Layer {l}', linewidth=1)
        ax2.axhline(0, color='gray', linestyle=':', linewidth=0.8)
        ax2.set_xlabel(f"Token position (last {shared_len} tokens)")
        ax2.set_ylabel("Key cosine")
        ax2.set_title("Ghost distortion by position")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)

        # Middle: Attention from last token under NAMM (layer 7)
        ax3 = fig.add_subplot(gs[1, 0])
        if 7 in attn_namm:
            attn_row = attn_namm[7]  # (n_heads, q_len, kv_len)
            # Average over heads, last query token
            avg_attn = attn_row[:, -1, :].mean(0).numpy()
            ax3.plot(avg_attn, color='#d62728', linewidth=0.8, alpha=0.8)
            ax3.set_title(f"NAMM: Last-token attention (layer 7, avg over heads)\nkv_len={len(avg_attn)}")
            ax3.set_xlabel("KV cache position")
            ax3.set_ylabel("Attention weight")
            ax3.grid(alpha=0.3)

        # Middle right: Attention from last token under Truncation (layer 7)
        ax4 = fig.add_subplot(gs[1, 1])
        if 7 in attn_trunc:
            attn_row = attn_trunc[7]
            avg_attn = attn_row[:, -1, :].mean(0).numpy()
            ax4.plot(avg_attn, color='#2ca02c', linewidth=0.8, alpha=0.8)
            ax4.set_title(f"Truncation: Last-token attention (layer 7, avg over heads)\nkv_len={len(avg_attn)}")
            ax4.set_xlabel("Token position (last {0} tokens)".format(args.cache_size))
            ax4.set_ylabel("Attention weight")
            ax4.grid(alpha=0.3)

        # Bottom: Summary text
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        summary = (
            f"Task: {task}  |  Prompt idx: {orig_idx}  |  Split: {split}  |  "
            f"Length: {n_tok} tokens  |  Cache: {args.cache_size}\n\n"
            f"Gold answer: {gold[:100]}\n"
            f"NAMM prediction: [would need generation pass]\n"
            f"Trunc prediction: [would need generation pass]\n\n"
            f"Hidden state cosine (last token, NAMM vs Trunc): "
            f"{F.cosine_similarity(hidden_namm.unsqueeze(0), hidden_trunc.unsqueeze(0)).item():.4f}\n"
            f"Mean key cosine across layers: {np.mean(mean_kcos):.4f}\n"
            f"Mean value cosine across layers: {np.mean(mean_vcos):.4f}"
        )
        ax5.text(0.05, 0.95, summary, transform=ax5.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        fig.suptitle(f"Case Study: {task.replace('lb/','')} idx={orig_idx} — Ghost Information Analysis",
                     fontsize=13, fontweight='bold')

        out_path = os.path.join(args.output_dir,
                                f"case_{task.replace('/','_')}_{orig_idx}_{split}.png")
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_path}")

        del outputs_namm, outputs_trunc
        torch.cuda.empty_cache()

    capture.remove()
    print("\nDone.")


if __name__ == "__main__":
    main()
