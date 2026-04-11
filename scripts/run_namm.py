"""Run NAMM eviction-policy training via Hydra (DDP-compatible)."""

import os
import sys
import torch
from torch.distributed import destroy_process_group
from omegaconf import DictConfig
import hydra

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from namm.run_utils import (
    ddp_setup,
    get_dist_info,
    wandb_init,
    make_eval_model,
    make_task_sampler,
    stochasticity_setup,
)
from utils.hydra_helpers import assert_fair01_test_size


@hydra.main(version_base=None, config_path='../config', config_name='config')
def main(cfg: DictConfig):
    _, global_rank, n_ddp = get_dist_info()
    is_ddp = global_rank > -1
    if is_ddp:
        ddp_setup(backend=cfg.backend, ddp_timeout_limit=cfg.ddp_timeout_limit)
        master_process = global_rank == 0
        seed_offset = global_rank
    else:
        master_process = True
        seed_offset = 0

    if master_process:
        print(f"SHARED Working directory: {os.getcwd()}")
        print(f"SHARED Output directory: " +
              f"{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")

    log_prefix = ''
    if is_ddp:
        log_prefix = f'RANK {global_rank} ({n_ddp} total): '

    stochasticity_setup(cfg=cfg, seed_offset=seed_offset,
                        log_prefix=log_prefix)

    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    with torch.no_grad():
        (memory_policy, memory_model, memory_evaluator, evolution_algorithm,
            auxiliary_loss) = make_eval_model(cfg=cfg, log_prefix=log_prefix)

        # ── Threshold-only eviction mode ─────────────────────────────────────
        # When threshold_only=True, NAMM evicts purely by score threshold
        # (s_i < 0) with no hard top-k cap, matching the original NAMM paper.
        # Two caps must both be lifted:
        #   1. selection_criteria.cache_size → drives the topk cut inside NAMM
        #   2. memory_evaluator.max_memory_length → physical KV buffer truncation
        #      that re-applies independently after NAMM eviction
        if cfg.get('threshold_only', False):
            if not hasattr(memory_policy, 'selection_criteria'):
                raise ValueError(
                    "threshold_only=True requested but memory_policy has no "
                    "selection_criteria attribute. Check your policy config "
                    "(requires a deep policy such as policy/deep.yaml)."
                )
            memory_policy.selection_criteria.cache_size = None
            memory_evaluator.max_memory_length = (
                memory_evaluator.max_conditioning_length)
            if master_process:
                print(
                    "[threshold_only=True] selection_criteria.cache_size=None "
                    "— eviction driven purely by score threshold (s_i < 0), "
                    "no hard top-k cap."
                )
                print(
                    f"[threshold_only=True] evaluator.max_memory_length set to "
                    f"{memory_evaluator.max_conditioning_length} (max_conditioning"
                    f"_length) to remove secondary KV buffer truncation."
                )
            # Warn if the scoring network is initialised at 0 — in threshold mode
            # scores must stay > 0 to retain tokens, but scoring_initializer=0
            # places CMA-ES right at the decision boundary.  A positive shift
            # (e.g. scoring_initializer=2) gives a stable starting point.
            scoring_init = cfg.get('scoring_initializer', None)
            if master_process and (scoring_init is None or scoring_init == 0):
                print(
                    "[threshold_only=True] WARNING: scoring_initializer=0 "
                    "(default). In threshold mode all tokens start at score≈0 "
                    "and can collapse below threshold immediately. Consider "
                    "adding scoring_initializer=2 to your run config or CLI "
                    "to start CMA-ES in a non-degenerate state."
                )

        task_sampler = make_task_sampler(cfg=cfg, log_prefix=log_prefix)

        # Apply answer-length filter + 3-way split with exact tokenizer-based filtering.
        # This ensures NAMM and LoRA use identical eligible sets and split indices.
        tokenizer = hydra.utils.call(cfg.tokenizer)
        max_answer_tok = cfg.get('max_answer_tokens', cfg.get('max_new_tokens', 64))
        task_sampler.filter_answers_by_token_count(tokenizer, max_answer_tok)

        train_frac = cfg.get('train_frac', 0.7)
        val_frac = cfg.get('val_frac', 0.15)
        # split_max_conditioning_length controls which prompts are eligible for
        # splitting; it is intentionally separate from max_conditioning_length
        # (which sets the model's KV buffer size) so that reducing buffer size
        # for memory reasons does not silently empty the training split.
        max_cond = cfg.get('split_max_conditioning_length',
                           cfg.get('max_conditioning_length', 6500))
        min_cond = cfg.get('min_conditioning_length', None)
        task_sampler.apply_train_val_test_split(
            train_frac=train_frac,
            val_frac=val_frac,
            max_conditioning_length=max_cond,
            min_conditioning_length=min_cond,
            tokenizer=tokenizer,
        )

        if master_process:
            run_choice = (
                hydra.core.hydra_config.HydraConfig.get()
                .runtime.choices.get('run@_global_', '')
            )
            assert_fair01_test_size(
                task_sampler,
                run_config=run_choice,
                train_frac=train_frac,
                val_frac=val_frac,
                min_conditioning_length=min_cond,
                max_conditioning_length=max_cond,
            )

        trainer = hydra.utils.instantiate(
            cfg.trainer,
            evaluation_model=memory_evaluator,
            task_sampler=task_sampler,
            evolution_algorithm=evolution_algorithm,
            auxiliary_loss=auxiliary_loss,
        )

        if cfg.wandb_config.wandb_log and master_process:
            wandb_init(cfg=cfg)

        with torch.no_grad():
            trainer.train()

        if is_ddp:
            destroy_process_group()


if __name__ == "__main__":
    with torch.no_grad():
        main()
