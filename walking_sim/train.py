import argparse
from pathlib import Path

import torch
from stable_baselines3 import SAC

from common import DEFAULT_MODEL, ROOT, make_environment


def main():
    parser = argparse.ArgumentParser(description="Continue training the pretrained SAC model.")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "sac_finetuned.zip")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if not args.model.is_file():
        parser.error(f"Model file not found: {args.model}")
    if args.output.resolve() == args.model.resolve():
        parser.error("Output must not overwrite the input checkpoint.")
    torch.set_num_threads(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with make_environment() as env:
        model = SAC.load(args.model, env=env, device="cpu", seed=args.seed)
        # A published checkpoint does not contain its original replay buffer.
        model.learning_starts = 1000
        model.tensorboard_log = None
        model.verbose = 1
        model.learn(total_timesteps=args.steps, reset_num_timesteps=True, progress_bar=False)
        model.save(args.output)
    print(f"Saved checkpoint: {args.output}")


if __name__ == "__main__":
    main()
