"""Shared setup helpers for NAMM training and ES/LoRA fine-tuning scripts.

These functions were previously embedded in scripts/run_namm.py, making it a
hybrid library+entry-point that other scripts and utils/hydra_helpers.py had
to import via sys.path manipulation.  Moving them here breaks that coupling:
any module can do ``from namm.run_utils import make_eval_model`` cleanly.
"""

import os
import random
import datetime

import numpy as np
import torch
from torch.distributed import init_process_group, destroy_process_group  # noqa: F401
from omegaconf import DictConfig, OmegaConf
import hydra


# ── Distributed training ─────────────────────────────────────────────────────

def ddp_setup(
        backend: str = "nccl",
        ddp_timeout_limit: str = '0:6:0',  # days:hours:minutes
):
    days, hours, seconds = ddp_timeout_limit.split(':')
    timeout_delta = datetime.timedelta(
        days=int(days), hours=int(hours), seconds=int(seconds))
    init_process_group(backend=backend, timeout=timeout_delta)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def get_dist_info():
    local_rank = int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK', -1))
    global_rank = int(os.getenv('OMPI_COMM_WORLD_RANK', -1))
    world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE', '-1'))

    local_rank = int(os.getenv(
        'LOCAL_RANK', -1)) if local_rank == -1 else local_rank
    global_rank = int(os.getenv(
        'RANK', -1)) if global_rank == -1 else global_rank
    world_size = int(os.getenv(
        'WORLD_SIZE', -1)) if world_size == -1 else world_size

    if global_rank != -1:
        os.environ['RANK'] = str(global_rank)  # Needed by torch distributed.
        os.environ['LOCAL_RANK'] = str(local_rank)
        os.environ['WORLD_SIZE'] = str(world_size)

    return local_rank, global_rank, world_size


# ── Experiment setup ──────────────────────────────────────────────────────────

def wandb_init(cfg):
    import wandb
    config_dict = OmegaConf.to_container(
        # allow missing values for memory experiments
        cfg, resolve=True, throw_on_missing=False,
    )
    # wandb has a 128-size character limit on the group name
    wandb.init(
        project=cfg.wandb_config.wandb_project,
        entity=getattr(cfg.wandb_config, 'wandb_entity', None),
        group=cfg.wandb_config.wandb_group_name[:127],
        name=cfg.wandb_config.wandb_run_name[:127],
        config=config_dict,
    )
    return wandb


def stochasticity_setup(cfg, seed_offset=0, log_prefix=''):
    print(log_prefix + f'Global rank used for seed offset {seed_offset}')
    np.random.seed(cfg.seed + seed_offset)
    torch.manual_seed(cfg.seed + seed_offset)
    random.seed(cfg.seed + seed_offset)

    # NOTE: likely can remove offset
    if cfg.deterministic_behavior:
        print('WARNING: training with deterministic behavior')
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)


# ── Model / task instantiation ────────────────────────────────────────────────

def make_eval_model(cfg, log_prefix='...', namm_active=True, cache_size=None):
    """Instantiate the eval-time model bundle.

    When ``namm_active=False`` the NAMM scoring network is still constructed
    from ``cfg`` (so checkpoints can be loaded against the same parameter
    layout) but is then swapped out for a passthrough ``Recency`` policy.
    Without this swap an A4 NAMM-off arm or a baseline run that pointed at
    a NAMM Hydra config would silently use a randomly-initialised NAMM
    instead of the full cache. ``cache_size`` is forwarded to ``Recency``;
    pass ``None`` for "no eviction".
    """
    print(log_prefix + 'Instantialting llm...')
    pretrained_llm = hydra.utils.call(cfg.pretrained_llm, _convert_="object")
    tokenizer = hydra.utils.call(cfg.tokenizer)

    adapter_path = getattr(cfg, 'adapter_path', None)
    if adapter_path:
        from peft import PeftModel
        print(log_prefix + f'Loading LoRA adapter from {adapter_path}...')
        pretrained_llm = PeftModel.from_pretrained(
            pretrained_llm, adapter_path)
        pretrained_llm = pretrained_llm.merge_and_unload()
        print(log_prefix + 'LoRA adapter merged into base weights.')

    print(log_prefix + 'Instantialting memory policy...')
    memory_policy = hydra.utils.instantiate(cfg.memory_policy,
                                            _convert_="object")

    print(log_prefix + 'Instantialting memory llm...')
    memory_model = hydra.utils.instantiate(
        cfg.memory_model, model=pretrained_llm, memory_policy=memory_policy,)

    # Load ES-fine-tuned weights into the model (for M3-ES sequential).
    # Must happen after memory_model construction because ES checkpoints use
    # fully-qualified memory_model parameter names (e.g. model.layers.0.*).
    es_checkpoint_path = getattr(cfg, 'es_checkpoint_path', None)
    if es_checkpoint_path:
        print(log_prefix + f'Loading ES checkpoint from {es_checkpoint_path}...')
        es_state = torch.load(es_checkpoint_path, map_location='cpu',
                              weights_only=True)
        is_delta = es_state.pop('__format__', None) == 'delta'
        model_params = dict(memory_model.named_parameters())
        loaded = 0
        for name, val in es_state.items():
            if name in model_params:
                target = model_params[name]
                val = val.to(dtype=target.dtype)
                if is_delta:
                    target.data.add_(val)
                else:
                    target.data.copy_(val)
                loaded += 1
        fmt = 'delta' if is_delta else 'absolute'
        print(log_prefix + f'ES checkpoint loaded: {loaded} params ({fmt} format).')

    print(log_prefix + 'Instantialting evaluation module...')
    memory_evaluator = hydra.utils.instantiate(
        cfg.memory_evaluator, model=memory_model, tokenizer=tokenizer)

    print(log_prefix + 'Instantialting evolution module...')
    evolution_algorithm = hydra.utils.instantiate(
        cfg.evolution_algorithm, param_size=memory_policy.param_size,
        _recursive_=False)

    init_param = memory_policy.get_init_param_values()

    evolution_algorithm.load_init(init_param=init_param)

    if cfg.auxiliary_loss is not None:
        auxiliary_loss = hydra.utils.instantiate(cfg.auxiliary_loss,
                                                 memory_policy=memory_policy)
    else:
        auxiliary_loss = None

    if not namm_active:
        # Replace the just-constructed NAMM with a passthrough Recency
        # policy. evolution_algorithm and auxiliary_loss are unused at
        # eval time, so they remain bound to the NAMM-shaped objects —
        # this matches the previous in-script swap pattern.
        from namm.policy.base import Recency
        recency_policy = Recency(cache_size=cache_size)
        memory_evaluator.swap_memory_policy(recency_policy)
        memory_policy = recency_policy
        # The Hydra config (e.g. namm_bam_i1_llama32_1b_5t) tunes
        # max_memory_length and batch_size for the small NAMM cache.
        # Without eviction the KV cache grows to the full prompt length,
        # so widen the buffer and clamp auto batch sizing.
        memory_evaluator.max_memory_length = (
            memory_evaluator.max_conditioning_length)
        if memory_evaluator.batch_size == "auto":
            memory_evaluator.batch_size = 1
        print(log_prefix + f'NAMM disabled; swapped to Recency '
              f'(cache_size={cache_size}, '
              f'eval_batch_size={memory_evaluator.batch_size}).')

    print(log_prefix + 'Finished instantiations.')
    return (memory_policy, memory_model, memory_evaluator, evolution_algorithm,
            auxiliary_loss)


def make_task_sampler(cfg, log_prefix='', **task_sampler_kwargs):
    print(log_prefix + f'Instantiating tasks: {cfg.task_sampler.tasks}; with '
          f' corresponding metrics: {cfg.task_sampler.metrics}')
    task_sampler = hydra.utils.instantiate(
        cfg.task_sampler, _convert_='none',
        **task_sampler_kwargs)
    return task_sampler
