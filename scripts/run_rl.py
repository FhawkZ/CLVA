"""CLI entrypoint for RL finetuning."""

from util.rl_finetune import RLFineTuneConfig, run_rl_finetuning


if __name__ == "__main__":
    run_rl_finetuning(RLFineTuneConfig())
