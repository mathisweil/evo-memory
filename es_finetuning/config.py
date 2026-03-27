from dataclasses import dataclass
from typing import Optional


@dataclass
class ESConfig:
    """Configuration for Evolution Strategies fine-tuning."""

    sigma: float = 0.001  # Noise scale for weight perturbations
    alpha: float = 0.0005  # Learning rate
    population_size: int = 8  # Population members per iteration
    num_iterations: int = 50  # Total ES iterations
    noise_mode: str = "correlated"  # "correlated" (shared seed) or "iid" (per-param seed)
    initial_seed: int = 33  # Initial random seed for numpy
    mini_batch_size: int = 16  # Samples per population eval (passed to evaluate_fn)
    log_dir: str = "es_runs"  # Directory for results, checkpoints, examples
    checkpoint_every: int = 0  # Periodic checkpoint interval (0 = final only)
    save_every: int = 0  # Permanent save interval (0 = disabled); kept forever, never cleaned up
    temperature: float = 0.0  # Generation temperature for training evals (0 = greedy)
    eval_temperature: float = 0.0  # Generation temperature for full eval (0 = greedy)
    num_samples: int = 1  # Number of generation samples per question during training (averaged)
    eval_num_samples: int = 1  # Number of generation samples per question during full eval (averaged)

    # Wandb logging
    wandb_log: bool = False
    wandb_project: str = "Experiments"
    wandb_entity: Optional[str] = "SNLP_NAMM"
    wandb_run_name: Optional[str] = None
    wandb_group_name: Optional[str] = None
