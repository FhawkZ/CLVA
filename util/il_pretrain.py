"""Imitation learning pretraining utilities."""

from dataclasses import dataclass


@dataclass
class ILPretrainConfig:
    """Configuration for IL pretraining."""

    dataset_dir: str = "data"
    output_dir: str = "outputs/il"
    va_model_name: str = "dit"
    chunk_size: int = 8
    epochs: int = 10


def run_il_pretraining(config: ILPretrainConfig) -> None:
    """Placeholder IL pretraining entry for the VA model."""
    print("[IL] Starting VA imitation pretraining:")
    print(config)
    print("[IL] TODO: implement dataset loading and DiT training loop.")
