import os
import torch
from torch.distributed import destroy_process_group
from omegaconf import DictConfig
import hydra

from namm.run_utils import (
    ddp_setup,
    get_dist_info,
    wandb_init,
    make_eval_model,
    make_task_sampler,
    stochasticity_setup,
)


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

        task_sampler = make_task_sampler(cfg=cfg, log_prefix=log_prefix)

        # Apply 3-way train/val/test split with exact tokenizer-based filtering.
        # This ensures NAMM and LoRA use identical eligible sets and split indices.
        tokenizer = hydra.utils.call(cfg.tokenizer)
        train_frac = cfg.get('train_frac', 0.7)
        val_frac = cfg.get('val_frac', 0.15)
        max_cond = cfg.get('max_conditioning_length', 6500)
        task_sampler.apply_train_val_test_split(
            train_frac=train_frac,
            val_frac=val_frac,
            max_conditioning_length=max_cond,
            tokenizer=tokenizer,
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
