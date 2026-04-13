"""CLI entrypoint for IL pretraining."""

from util.il_pretrain import ILPretrainConfig, run_il_pretraining


if __name__ == "__main__":
    run_il_pretraining(ILPretrainConfig())
