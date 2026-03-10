from dataclasses import dataclass


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
