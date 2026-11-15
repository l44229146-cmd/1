import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC

from common import ROOT, make_environment


def main():
    parser = argparse.ArgumentParser(description="Compare pretrained walking policies on identical seeds.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(100, 110)))
    parser.add_argument("--target-speed", type=float, default=1.3)
    parser.add_argument("--output", type=Path, default=ROOT / "comparison")
    args = parser.parse_args()
    if not np.isfinite(args.target_speed) or args.target_speed <= 0:
        parser.error("--target-speed must be positive and finite")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    rows = []
    with make_environment() as env:
        for label in ("simple", "medium", "expert"):
            path = ROOT / "models" / f"humanoid-v5-sac-{label}.zip"
            model = SAC.load(path, device="cpu")
            if model.observation_space != env.observation_space or model.action_space != env.action_space:
                raise ValueError(f"{path} does not match the environment spaces")
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            for seed in args.seeds:
                obs, _ = env.reset(seed=seed)
                start = float(env.unwrapped.data.qpos[0])
                for step in range(1000):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, _ = env.step(action)
                    if terminated or truncated:
                        break
                data = env.unwrapped.data
                duration = float(data.time)
                speed = (float(data.qpos[0]) - start) / duration
                rows.append({
                    "policy": label, "sha256": checksum, "seed": seed,
                    "steps": step + 1, "duration_s": duration,
                    "average_forward_speed_m_s": speed,
                    "target_speed_m_s": args.target_speed,
                    "absolute_mean_speed_error_m_s": abs(speed - args.target_speed),
                    "terminated_unhealthy": bool(terminated),
                    "truncated_time_limit": bool(truncated),
                })
    with (args.output / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for label in ("simple", "medium", "expert"):
        subset = [row for row in rows if row["policy"] == label]
        summary[label] = {
            "episodes": len(subset),
            "full_time_limit_episodes": sum(row["truncated_time_limit"] and not row["terminated_unhealthy"] for row in subset),
            "mean_speed_m_s": float(np.mean([row["average_forward_speed_m_s"] for row in subset])),
            "mean_absolute_speed_error_m_s": float(np.mean([row["absolute_mean_speed_error_m_s"] for row in subset])),
            "mean_duration_s": float(np.mean([row["duration_s"] for row in subset])),
        }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
