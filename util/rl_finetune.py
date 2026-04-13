"""Reinforcement learning finetuning utilities."""

from dataclasses import dataclass


@dataclass
class RLFineTuneConfig:
    """Configuration for RL finetuning in simulation."""

    env_name: str = "mujoco_sim"
    output_dir: str = "outputs/rl"
    chunk_size: int = 8
    total_steps: int = 100000
    consistency_weight: float = 1.0


def run_rl_finetuning(config: RLFineTuneConfig) -> None:
    """Placeholder RL finetuning entry with consistency-based scoring."""
    print("[RL] Starting RL finetuning:")
    print(config)
    print("[RL] TODO: compare post-chunk real image with Dreamer prediction in reward.")
