#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Any, Dict
from pathlib import Path

import numpy as np

from applied_planning.envs.lite6_pick_place_env import Lite6PickPlaceEnv
from applied_planning.control.controllers import SimpleJointVelocityController


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a simple Lite6 pick-and-place episode.")
    parser.add_argument("--headless", action="store_true", help="Run without a simulator backend")
    parser.add_argument("--steps", type=int, default=50, help="Number of steps to run")
    parser.add_argument(
        "--mujoco-model",
        type=str,
        default=str(
            Path(__file__).resolve().parents[1]
            / "src/applied_planning/sim/assets/ufactory_lite6/lite6.xml"
        ),
        help="Path to the MuJoCo MJCF file for uFactory Lite 6 (if present).",
    )
    parser.add_argument(
        "--force-mujoco",
        action="store_true",
        help="Exit with an error if MuJoCo backend cannot be initialized.",
    )
    args = parser.parse_args()

    adapter = None  # Default to headless unless we can load MuJoCo
    if not args.headless:
        model_path = Path(args.mujoco_model)
        try:
            from applied_planning.sim.adapters.mujoco_backend import MujocoLite6Adapter

            if model_path.exists():
                adapter = MujocoLite6Adapter(str(model_path), viewer=True)
                print(f"Using MuJoCo backend with model: {model_path}")
            else:
                print(
                    "MuJoCo model not found at default path. Running headless.\n"
                    f"Expected at: {model_path}\n"
                    "Place Menagerie Lite6 MJCF and meshes under src/applied_planning/sim/assets/ufactory_lite6/\n"
                    "Tip: python scripts/fetch_menagerie_lite6.py to auto-download."
                )
        except Exception as e:
            print(
                "Could not initialize MuJoCo backend. Running headless.\n"
                f"Reason: {e}\n"
                "Hints: `uv pip install mujoco` and ensure the MJCF exists at the expected path\n"
                "       or pass --mujoco-model to point to your lite6.xml"
            )

    if adapter is None:
        if args.force_mujoco:
            raise SystemExit(
                "--force-mujoco set but MuJoCo backend is not active. See messages above."
            )
        print("Mode: headless (no simulator backend)")
    else:
        print("Mode: MuJoCo backend active")
    env = Lite6PickPlaceEnv(adapter=adapter, render_mode=None)
    ctrl = SimpleJointVelocityController(kp=0.5)

    obs, _ = env.reset()
    goal: Dict[str, Any] = {"joint_pos": np.zeros(6, dtype=np.float32)}
    ctrl.reset()

    for t in range(args.steps):
        action = ctrl.act(obs, goal)["qvel"].astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    print("Run complete.")


if __name__ == "__main__":
    main()
