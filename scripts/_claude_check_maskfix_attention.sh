#!/bin/bash
set -u
PROJ=/cs/student/project_msc/2025/csml/rhautier/evo-memory-giacommo
ENV=/cs/student/project_msc/2025/csml/rhautier/envs/th2
cd "$PROJ"
source /cs/student/project_msc/2025/csml/rhautier/miniforge3/etc/profile.d/conda.sh
conda activate "$ENV"
export CUDA_VISIBLE_DEVICES=0

# Copy the latest checkpoint to avoid reading while training writes
CKPT_DIR="experiments/namm_only_runs/memory_evolution_hf/Llama-3.2-1B-Instruct/NAMM/attn-spec-norm/bam/binary-1024cs/rh-multi-qa-5t-cma-es-p8-rMeanTrue-shared-8pop-8qs-256fixDel-llama32-1b-5t-cs1024-maskfix/1337"
# Use ckpt.pt (best checkpoint, not actively being written to)
# latest.pt may be mid-write by the training process
CKPT_FILE="$CKPT_DIR/ckpt.pt"

python -c "
import torch, numpy as np, sys, os
os.chdir('$PROJ')
sys.path.insert(0, '.')
from transformers import AutoTokenizer
from hydra import compose, initialize
from namm.run_utils import make_eval_model
from es_finetuning.device import get_device
from datasets import load_dataset
import json

device = get_device()
tok = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B-Instruct')

overrides = [
    'run@_global_=namm_bam_i1_llama32_1b_5t',
    'wandb_log=false', 'wandb_project=Experiments',
    'filter_by_length=8192', 'cache_size=1024', 'max_memory_length=1024',
    '+protected_tail_n=5',
]
with initialize(version_base=None, config_path='config', job_name='check_attn'):
    cfg = compose(config_name='config', overrides=overrides)

with torch.no_grad():
    (memory_policy, memory_model, _, _, _) = make_eval_model(cfg=cfg)
memory_model.to(device)

# Load the maskfix checkpoint
ckpt = torch.load('$CKPT_FILE', map_location='cpu', weights_only=False)
evo = ckpt['evolution_state']
print(f'Checkpoint iter: {ckpt[\"iter_num\"]}')
print(f'Best val: {ckpt[\"best_val_loss\"]:.6f}')
memory_model.set_memory_params(evo['mean'].unsqueeze(0).to(device))
bp = 'stored_buffers_to_save.'
bd = {k[len(bp):]: v.to(device) for k, v in evo.items() if k.startswith(bp)}
if bd: memory_model.load_buffers_dict(buffers_dict=bd)
memory_policy.set_params_batch_idxs(np.zeros([1]))
memory_policy.record_eval_stats = True
memory_policy.initialize_stat_objects()

# Hook multiple layers
captured = {l: [] for l in [0, 3, 7, 11, 15]}
def make_hook(layer_idx):
    def fn(module, input, output):
        attn = output[1]
        if attn is not None:
            a = attn[0].detach().cpu().float()
            last_row = a[:, -1, :]
            entropy = -(last_row * torch.log(last_row.clamp(min=1e-10))).sum(-1).mean().item()
            captured[layer_idx].append({
                'kv_len': int(a.shape[-1]),
                'entropy': entropy,
                'max_attn': float(last_row.max()),
                'std': float(last_row.std()),
            })
    return fn
for l in captured:
    memory_model.model.layers[l].self_attn.register_forward_hook(make_hook(l))

# Test on 3 prompts of different lengths
ds = load_dataset('THUDM/LongBench', '2wikimqa', split='test', trust_remote_code=True)
prompt_templates = json.load(open('data/longbench/dataset2prompt.json'))

for prompt_idx in [147, 3, 95]:
    example = ds[prompt_idx]
    prompt = prompt_templates['2wikimqa'].format(**example)
    messages = [{'role': 'user', 'content': prompt}]
    text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    bos = tok.bos_token or ''
    if bos and text.startswith(bos): text = text[len(bos):]
    ids = tok(text, add_special_tokens=True, return_tensors='pt').input_ids.to(device)
    mask = torch.ones_like(ids)
    n_tok = ids.shape[1]

    memory_policy.initialize_stat_objects()
    for l in captured: captured[l] = []
    with torch.no_grad():
        out = memory_model(ids, attention_mask=mask, output_attentions=True,
                           use_cache=True, apply_memory_policy=True)

    print(f'\n=== Prompt idx={prompt_idx}, {n_tok} tokens, {len(captured[7])} chunks ===')
    print(f'{\"Chunk\":>6s} {\"kv_len\":>7s} {\"L0 ent\":>8s} {\"L7 ent\":>8s} {\"L15 ent\":>8s} {\"L7 max\":>8s} {\"L7 std\":>10s} {\"uniform?\":>9s}')
    for i in range(len(captured[7])):
        l0 = captured[0][i] if i < len(captured[0]) else {}
        l7 = captured[7][i] if i < len(captured[7]) else {}
        l15 = captured[15][i] if i < len(captured[15]) else {}
        is_u = l7.get('std', 1) < 1e-6
        print(f'{i:6d} {l7.get(\"kv_len\",0):7d} {l0.get(\"entropy\",0):8.3f} '
              f'{l7.get(\"entropy\",0):8.3f} {l15.get(\"entropy\",0):8.3f} '
              f'{l7.get(\"max_attn\",0):8.6f} {l7.get(\"std\",0):10.6f} '
              f'{\"YES\" if is_u else \"no\":>9s}')
" 2>&1
