#!/usr/bin/env python3

import argparse
from pathlib import Path

from gym_data import derive_offline_dataset_from_oracle, rollout_dataset, write_payload


def format_tau(tau: float) -> str:
    return f"{tau:.2f}"


def format_gamma(gamma: float) -> str:
    out = f"{gamma:.6f}".rstrip("0").rstrip(".")
    return out


def offline_oracle_seed(env_idx: int, rep: int) -> int:
    return env_idx * 100000 + rep


def offline_label_seed(env_idx: int, tau_idx: int, rep: int) -> int:
    return env_idx * 100000 + tau_idx * 1000 + rep


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Gym datasets in batch.")
    parser.add_argument("--dataset", choices=["offline", "target"], default="offline")
    parser.add_argument("--N", type=int, default=50)
    parser.add_argument("--T", type=int, default=50)
    parser.add_argument("--n-rep", type=int, default=50)
    parser.add_argument("--rep-start", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--taus", nargs="+", type=float, default=[0.05, 0.10, 0.20, 0.30])
    parser.add_argument("--seed", type=int, default=10002)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--misclassification", choices=["constant", "state_dependent"], default="constant")
    parser.add_argument("--state-strength", type=float, default=0.75)
    parser.add_argument("--error-center", type=float, default=0.0)
    parser.add_argument("--error-scale", type=float, default=1.0)
    parser.add_argument("--target-x-threshold", type=float, default=-0.5)
    parser.add_argument("--target-theta-threshold", type=float, default=0.1)
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

    if args.n_rep < 1:
        raise ValueError("--n-rep must be at least 1.")
    if args.rep_start < 1:
        raise ValueError("--rep-start must be at least 1.")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for env_idx, env_name in enumerate(args.envs, start=1):
        env_dir = output_root / env_name
        env_dir.mkdir(parents=True, exist_ok=True)

        if args.dataset == "target":
            target_dir = env_dir / "target_mc"
            target_dir.mkdir(parents=True, exist_ok=True)
            out_file = target_dir / (
                f"N_{args.N}_T_{args.T}_gamma_{format_gamma(args.gamma)}_seed_{args.seed}"
                + (f"_policy_x_{args.target_x_threshold:g}_theta_{args.target_theta_threshold:g}" if env_name == "CartPole-v1" else "")
                + ".json"
            )
            if out_file.exists():
                raise FileExistsError(f"Use a fresh output directory: {out_file}")
            payload = rollout_dataset(
                env_name=env_name,
                dataset="target",
                n_traj=args.N,
                horizon=args.T,
                tau=0.0,
                gamma=args.gamma,
                seed=args.seed,
                summary_only=args.summary_only,
                target_x_threshold=args.target_x_threshold,
                target_theta_threshold=args.target_theta_threshold,
            )
            write_payload(payload, out_file)
            print(f"generated {out_file}")
            continue

        tau_dirs = {}
        if args.misclassification == "state_dependent":
            if env_name != "CartPole-v1":
                raise ValueError("Pass --envs CartPole-v1 for state-dependent recording.")
            env_dir = env_dir / "state_dependent"
        for tau in args.taus:
            tau_dir = env_dir / f"tau_{format_tau(tau)}"
            tau_dir.mkdir(parents=True, exist_ok=True)
            tau_dirs[tau] = tau_dir

        rep_stop = args.rep_start + args.n_rep
        for rep in range(args.rep_start, rep_stop):
            oracle_payload = rollout_dataset(
                env_name=env_name,
                dataset="offline",
                n_traj=args.N,
                horizon=args.T,
                tau=0.0,
                gamma=args.gamma,
                seed=offline_oracle_seed(env_idx, rep),
                apply_label_noise=False,
            )

            for tau_idx, tau in enumerate(args.taus, start=1):
                out_file = tau_dirs[tau] / f"rep_{rep:03d}.json"
                if out_file.exists():
                    raise FileExistsError(f"Use a fresh output directory: {out_file}")
                payload = derive_offline_dataset_from_oracle(
                    oracle_payload,
                    tau=tau,
                    seed=offline_label_seed(env_idx, tau_idx, rep),
                    misclassification=args.misclassification, state_strength=args.state_strength,
                    error_center=args.error_center, error_scale=args.error_scale,
                )
                write_payload(payload, out_file)
                print(f"generated {out_file}")


if __name__ == "__main__":
    main()
