#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path


def format_tau(tau: float) -> str:
    return f"{tau:.2f}"


def format_gamma(gamma: float) -> str:
    out = f"{gamma:.6f}".rstrip("0").rstrip(".")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Gym datasets in batch.")
    parser.add_argument("--dataset", choices=["offline", "target"], default="offline")
    parser.add_argument("--N", type=int, default=50)
    parser.add_argument("--T", type=int, default=50)
    parser.add_argument("--n-rep", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--taus", nargs="+", type=float, default=[0.05, 0.10, 0.20, 0.30])
    parser.add_argument("--seed", type=int, default=10002)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument(
        "--envs",
        nargs="+",
        default=["MountainCar-v0", "CartPole-v1"],
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent),
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    generator = script_dir / "gym_data.py"
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for env_idx, env_name in enumerate(args.envs, start=1):
        env_dir = output_root / env_name
        env_dir.mkdir(parents=True, exist_ok=True)

        if args.dataset == "target":
            target_dir = env_dir / "target_mc"
            target_dir.mkdir(parents=True, exist_ok=True)
            out_file = target_dir / (
                f"N_{args.N}_T_{args.T}_gamma_{format_gamma(args.gamma)}_seed_{args.seed}.json"
            )
            cmd = [
                sys.executable,
                str(generator),
                "--env",
                env_name,
                "--dataset",
                "target",
                "--N",
                str(args.N),
                "--T",
                str(args.T),
                "--tau",
                "0",
                "--gamma",
                str(args.gamma),
                "--seed",
                str(args.seed),
                "--output",
                str(out_file),
            ]
            if args.summary_only:
                cmd.append("--summary-only")
            subprocess.run(cmd, check=True)
            print(f"generated {out_file}")
            continue

        for tau_idx, tau in enumerate(args.taus, start=1):
            tau_dir = env_dir / f"tau_{format_tau(tau)}"
            tau_dir.mkdir(parents=True, exist_ok=True)

            for rep in range(1, args.n_rep + 1):
                seed = env_idx * 100000 + tau_idx * 1000 + rep
                out_file = tau_dir / f"rep_{rep:03d}.json"
                cmd = [
                    sys.executable,
                    str(generator),
                    "--env",
                    env_name,
                    "--dataset",
                    args.dataset,
                    "--N",
                    str(args.N),
                    "--T",
                    str(args.T),
                    "--tau",
                    str(tau),
                    "--gamma",
                    str(args.gamma),
                    "--seed",
                    str(seed),
                    "--output",
                    str(out_file),
                ]
                subprocess.run(cmd, check=True)
                print(f"generated {out_file}")


if __name__ == "__main__":
    main()