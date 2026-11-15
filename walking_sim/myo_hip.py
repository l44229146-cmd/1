"""MyoLeg smoke test with an idealised, externally applied hip torque."""

import argparse
import csv
import math
from pathlib import Path

import gymnasium as gym
import mujoco
import myosuite  # Registers MyoSuite environments.
import numpy as np


JOINTS = ("hip_flexion_r", "hip_flexion_l", "knee_angle_r", "knee_angle_l", "ankle_angle_r", "ankle_angle_l")


def create_environment():
    env = gym.make("myoLegWalk-v0")
    model = env.unwrapped.model
    dofs = {}
    for name in JOINTS:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0:
            env.close()
            raise RuntimeError(f"Required joint is missing: {name}")
        if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
            env.close()
            raise RuntimeError(f"Expected a rotational hinge joint: {name}")
        dofs[name] = (int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint]))
    return env, dofs


def simulate(steps, torque_nm, seed, output):
    env, dofs = create_environment()
    try:
        env.reset(seed=seed)
        state = env.unwrapped.data
        # Zero excitation is intentionally a smoke test, not a walking policy.
        muscle_action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        output.parent.mkdir(parents=True, exist_ok=True)
        columns = ["time_s", "hip_torque_right_nm", "hip_torque_left_nm"]
        for name in JOINTS:
            columns.extend([f"{name}_angle_rad", f"{name}_velocity_rad_s"])
        columns.extend(["terminated", "truncated"])
        with output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            for step in range(steps):
                state.qfrc_applied[dofs["hip_flexion_r"][1]] = torque_nm
                state.qfrc_applied[dofs["hip_flexion_l"][1]] = torque_nm
                _, _, terminated, truncated, _ = env.step(muscle_action)
                row = {
                    "time_s": float(state.time),
                    "hip_torque_right_nm": torque_nm,
                    "hip_torque_left_nm": torque_nm,
                    "terminated": terminated,
                    "truncated": truncated,
                }
                for name, (qpos, dof) in dofs.items():
                    row[f"{name}_angle_rad"] = float(state.qpos[qpos])
                    row[f"{name}_velocity_rad_s"] = float(state.qvel[dof])
                writer.writerow(row)
                if terminated or truncated:
                    break
        print(f"Recorded {step + 1} steps; simulated {state.time:.3f} s; output: {output}")
        print("Zero muscle excitation is not a walking controller; hip torque is an idealised device proxy.")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--torque-nm", type=float, default=0.0, help="Constant torque per hip, within -10 to 10 Nm.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("walking_sim") / "results_myo" / "smoke_test.csv")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if not math.isfinite(args.torque_nm) or abs(args.torque_nm) > 10:
        parser.error("--torque-nm must be finite and between -10 and 10 Nm")
    simulate(args.steps, args.torque_nm, args.seed, args.output)


if __name__ == "__main__":
    main()
