import argparse
import csv
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
from stable_baselines3 import SAC

from common import DEFAULT_MODEL, ROOT, make_environment


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Humanoid-v5 SAC policy.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--render", action="store_true", help="Display an interactive window.")
    parser.add_argument("--gif", action="store_true", help="Save a sampled animation.")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.render and args.gif:
        parser.error("Choose either --render or --gif")
    if not args.model.is_file():
        parser.error(f"Model file not found: {args.model}")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    model_hash = hashlib.sha256(args.model.read_bytes()).hexdigest()
    model = SAC.load(args.model, device="cpu")
    render_mode = "human" if args.render else "rgb_array" if args.gif else None
    with make_environment(render_mode) as env:
        if model.observation_space != env.observation_space or model.action_space != env.action_space:
            raise ValueError("The model spaces do not match the Humanoid-v5 environment.")
        obs, _ = env.reset(seed=args.seed)
        sim = env.unwrapped
        m, d = sim.model, sim.data
        joints = [
            (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i),
             int(m.jnt_qposadr[i]), int(m.jnt_dofadr[i]))
            for i in range(m.njnt)
            if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
        ]
        geoms = {
            name: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ("floor", "right_foot", "left_foot")
        }
        if any(value < 0 for value in geoms.values()):
            raise ValueError("Could not locate expected floor and foot geometries.")
        rows = []
        frames = []
        start_x = float(d.qpos[0])
        total_reward = 0.0
        terminated = truncated = False
        for step in range(args.steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            # Recompute forces at the state and action reported in this row.
            mujoco.mj_forward(m, d)
            contacts = set()
            for contact in d.contact[:d.ncon]:
                contacts.add(frozenset((int(contact.geom1), int(contact.geom2))))
            row = {
                "step": step + 1,
                "time_s": float(d.time),
                "root_x_m": float(d.qpos[0]),
                "root_z_m": float(d.qpos[2]),
                "forward_velocity_m_s": float(info["x_velocity"]),
                "right_foot_ground_contact": int(frozenset((geoms["floor"], geoms["right_foot"])) in contacts),
                "left_foot_ground_contact": int(frozenset((geoms["floor"], geoms["left_foot"])) in contacts),
                "reward": float(reward),
                "terminated": int(terminated),
                "truncated": int(truncated),
            }
            for name, pos, dof in joints:
                row[f"{name}_angle_rad"] = float(d.qpos[pos])
                row[f"{name}_velocity_rad_s"] = float(d.qvel[dof])
                row[f"{name}_actuator_torque_nm"] = float(d.qfrc_actuator[dof])
            rows.append(row)
            if args.gif and step % 4 == 0:
                frames.append(env.render())
            if terminated or truncated:
                break

        with (args.output / "joint_data.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        elapsed = float(d.time)
        distance = float(d.qpos[0]) - start_x
        summary = {
            "environment": "Humanoid-v5",
            "model_sha256": model_hash,
            "seed": args.seed,
            "steps": len(rows),
            "duration_s": elapsed,
            "forward_distance_m": distance,
            "mean_forward_speed_m_s": distance / elapsed,
            "reward": total_reward,
            "terminated_unhealthy": bool(terminated),
            "truncated_time_limit": bool(truncated),
        }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for side in ("left", "right"):
            for axis, suffix in zip(axes, ("angle_rad", "velocity_rad_s", "actuator_torque_nm")):
                axis.plot([row["time_s"] for row in rows],
                          [row[f"{side}_hip_y_{suffix}"] for row in rows], label=side)
        for axis, label in zip(axes, ("Hip angle (rad)", "Hip velocity (rad/s)", "Actuator torque (Nm)")):
            axis.set_ylabel(label)
            axis.grid(alpha=0.3)
            axis.legend()
        axes[-1].set_xlabel("Time (s)")
        fig.tight_layout()
        fig.savefig(args.output / "hip_curves.png", dpi=150)
        plt.close(fig)
        if args.gif:
            imageio.mimsave(args.output / "walking.gif", frames, duration=4 * sim.dt * 1000, loop=0)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
