from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from torch.nn import functional as F
import torch
import transformers
import numpy as np


from hydra import compose, initialize
import os, sys
from namm.run_utils import make_eval_model, make_task_sampler, wandb_init
import omegaconf
import hydra
import time


# File containing imports to be used in hydra configs/for instantiating hydra
# objects outside main


class LlamaCompatModel:
    """Loads LLaMA 3.2+ models with transformers 4.41.x by patching the
    rope_scaling config, which changed format in 4.45 (llama3 rope_type).
    Also supports overriding max_position_embeddings to reduce RoPE cache size.
    """

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path,
                        max_position_embeddings=None, **kwargs):
        import json
        import os
        from transformers import LlamaConfig

        # Resolve relative paths from the original working directory
        # (Hydra changes cwd to the output dir at runtime)
        if not os.path.isabs(pretrained_model_name_or_path):
            try:
                from hydra.utils import get_original_cwd
                base = get_original_cwd()
            except ValueError:
                # Running via hydra.initialize() (not @hydra.main) — use CWD
                base = os.getcwd()
            candidate = os.path.join(base, pretrained_model_name_or_path)
            if os.path.isdir(candidate):
                pretrained_model_name_or_path = candidate
            # else: treat as HuggingFace repo ID (e.g. "meta-llama/Llama-3.2-1B-Instruct")

        config_file = os.path.join(pretrained_model_name_or_path, 'config.json')
        try:
            with open(config_file) as f:
                cfg = json.load(f)
        except FileNotFoundError:
            # Not a local path — resolve via HuggingFace Hub
            from huggingface_hub import hf_hub_download
            resolved = hf_hub_download(pretrained_model_name_or_path,
                                       'config.json')
            with open(resolved) as f:
                cfg = json.load(f)

        # Replace new-style llama3 rope_scaling with None so the NAMM
        # _init_rope falls back to standard LlamaRotaryEmbedding.
        # Safe for sequences up to original_max_position_embeddings (8192).
        cfg['rope_scaling'] = None

        if max_position_embeddings is not None:
            cfg['max_position_embeddings'] = max_position_embeddings

        config = LlamaConfig(**cfg)
        kwargs.pop('rope_scaling', None)  # prevent conflicting kwarg
        return AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, config=config, **kwargs)

def initialize_cfg(
        config_path="config",
        hydra_overrides: dict = {},
        log_yaml: bool = False,

):
    with initialize(version_base=None, config_path=config_path,
                     job_name="test_app"):
        cfg = compose(config_name="config",
                      overrides=hydra_overrides,
                      )
        if log_yaml:
            print('Loading the following configurations:')
            print(omegaconf.OmegaConf.to_yaml(cfg))
    return cfg


def load_run_cfgs_trainer(
    run_file_path_location: str,
    hydra_overrides_from_dict: dict = dict(
        wandb_log="false",
        wandb_project="Experiments",
    ),
    task_sampler_kwargs: dict = dict(
        tasks=["lb/passage_retrieval_en"],
    ),
    config_path="config",
    batch_size: int = 1,
):
    print(f"Loading model specified in {run_file_path_location}")
    start_time = time.time()
    hydra_overrides = [
        f"run@_global_={run_file_path_location}",
    ]

    hydra_overrides_from_dict = [f"{k}={v}" for k, v in
                                 hydra_overrides_from_dict.items()]

    hydra_overrides = hydra_overrides + hydra_overrides_from_dict

    cfg = initialize_cfg(
        config_path=config_path,
        hydra_overrides=hydra_overrides,
    )

    (memory_policy, memory_model, memory_evaluator, evolution_algorithm,
     auxiliary_loss) = make_eval_model(cfg=cfg)

    task_sampler = make_task_sampler(cfg=cfg, **task_sampler_kwargs)

    trainer = hydra.utils.instantiate(
        cfg.trainer,
        evaluation_model=memory_evaluator,
        task_sampler=task_sampler,
        evolution_algorithm=evolution_algorithm,
        auxiliary_loss=auxiliary_loss,
    )
    params, buffers = trainer.sample_and_synchronize_params(best=True)
    memory_model = memory_model
    memory_model.set_memory_params(params=params)

    if memory_model.memory_policy_has_buffers_to_merge():
        memory_model.load_buffers_dict(buffers_dict=buffers)

    batch_idxs = np.zeros([batch_size])
    memory_policy.set_params_batch_idxs(batch_idxs)
    print("Time taken:", round(time.time() - start_time))
    return trainer  # contains all other models
