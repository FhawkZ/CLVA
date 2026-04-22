"""CLI entrypoint for IL world-model future prediction."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from util.il_pretrain import load_world_model_checkpoint, predict_future_frames


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict future frames from one observation + action chunk."
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to msgpack checkpoint.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input .npz containing init_observation [B,H,W,C] and action_seq [B,T,A].",
    )
    parser.add_argument("--output", type=str, required=True, help="Output .npz for predictions.")
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--action-dim", type=int, required=True)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--deter-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    raw = np.load(args.input)
    if "init_observation" not in raw or "action_seq" not in raw:
        raise ValueError("Input npz must contain 'init_observation' and 'action_seq'.")
    init_observation = raw["init_observation"]
    action_seq = raw["action_seq"]

    model, params = load_world_model_checkpoint(
        ckpt_path=args.ckpt,
        image_shape=(args.height, args.width, args.channels),
        action_dim=args.action_dim,
        latent_dim=args.latent_dim,
        deter_dim=args.deter_dim,
    )
    preds = predict_future_frames(
        model=model,
        params=params,
        init_observation=init_observation,
        action_seq=action_seq,
        seed=args.seed,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, predictions=preds)
    print(f"[IL-Predict] saved predictions to: {out_path}")
    print(f"[IL-Predict] predictions shape: {preds.shape}")


if __name__ == "__main__":
    main()
