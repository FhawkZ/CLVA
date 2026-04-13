"""CLI entrypoint for DreamerV3 training."""

from util.dreamer_train import DreamerTrainConfig, run_dreamer_training


if __name__ == "__main__":
    run_dreamer_training(DreamerTrainConfig())
