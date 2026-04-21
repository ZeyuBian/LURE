#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym


CARTPOLE_TRANSITION_NOISE_SD = 0.1
CARTPOLE_REWARD_NOISE_SD = 0.1
CARTPOLE_ACTION_REWARD_COEF = 0.8


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def clip_prob(p: float) -> float:
    return float(np.clip(p, 0.15, 0.85))


def binary_to_env_action(env_name: str, action_bin: int) -> int:
    if env_name == "MountainCar-v0":
        return 0 if int(action_bin) == 0 else 2
    if env_name == "CartPole-v1":
        return int(action_bin)
    raise ValueError(f"Unsupported environment: {env_name}")


def target_policy(env_name: str, state: np.ndarray) -> float:
    if env_name == "MountainCar-v0":
        return 0.5
    if env_name == "CartPole-v1":
        score = 0.10 * state[0] + 0.25 * state[1] + 1.20 * state[2] + 0.35 * state[3]
        return sigmoid(score)
    raise ValueError(f"Unsupported environment: {env_name}")


def behavior_prob(env_name: str, state: np.ndarray) -> float:
    if env_name == "MountainCar-v0":
        score = 1.5 * state[0] + 6.0 * state[1]
        return clip_prob(sigmoid(score))
    if env_name == "CartPole-v1":
        return 0.5
    raise ValueError(f"Unsupported environment: {env_name}")


def reset_env(env, seed: int) -> np.ndarray:
    out = env.reset(seed=int(seed))
    if isinstance(out, tuple):
        obs = out[0]
    else:
        obs = out
    return np.asarray(obs, dtype=float)


def step_env(env, action: int, rng: np.random.Generator):
    if env.spec is not None and env.spec.id == "CartPole-v1":
        return step_cartpole(env, action, rng)

    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
        return np.asarray(obs, dtype=float), float(reward), done, info
    obs, reward, done, info = out
    return np.asarray(obs, dtype=float), float(reward), bool(done), info


def step_cartpole(env, action: int, rng: np.random.Generator):
    cartpole = env.unwrapped
    x, x_dot, theta, theta_dot = [float(v) for v in cartpole.state]

    force = cartpole.force_mag if int(action) == 1 else -cartpole.force_mag
    costheta = np.cos(theta)
    sintheta = np.sin(theta)

    temp = (force + cartpole.polemass_length * theta_dot**2 * sintheta) / cartpole.total_mass
    thetaacc = (cartpole.gravity * sintheta - costheta * temp) / (
        cartpole.length * (4.0 / 3.0 - cartpole.masspole * costheta**2 / cartpole.total_mass)
    )
    xacc = temp - cartpole.polemass_length * thetaacc * costheta / cartpole.total_mass

    if cartpole.kinematics_integrator == "euler":
        x = x + cartpole.tau * x_dot
        x_dot = x_dot + cartpole.tau * xacc
        theta = theta + cartpole.tau * theta_dot
        theta_dot = theta_dot + cartpole.tau * thetaacc
    else:
        x_dot = x_dot + cartpole.tau * xacc
        x = x + cartpole.tau * x_dot
        theta_dot = theta_dot + cartpole.tau * thetaacc
        theta = theta + cartpole.tau * theta_dot

    noise = rng.normal(loc=0.0, scale=CARTPOLE_TRANSITION_NOISE_SD, size=4)
    next_state = np.asarray([x, x_dot, theta, theta_dot], dtype=float) + noise
    cartpole.state = tuple(float(v) for v in next_state)

    terminated = bool(
        next_state[0] < -cartpole.x_threshold
        or next_state[0] > cartpole.x_threshold
        or next_state[2] < -cartpole.theta_threshold_radians
        or next_state[2] > cartpole.theta_threshold_radians
    )
    reward_mean = (
        1.0
        - (x ** 2) / 11.52
        - (theta ** 2) / 288.0
        + CARTPOLE_ACTION_REWARD_COEF * float(action)
    )
    reward = reward_mean + float(rng.normal(loc=0.0, scale=CARTPOLE_REWARD_NOISE_SD))

    return next_state, float(reward), terminated, {}


def choose_action(dataset: str, env_name: str, state: np.ndarray, rng: np.random.Generator) -> int:
    if dataset == "offline":
        return int(rng.random() < behavior_prob(env_name, state))
    return int(rng.random() < target_policy(env_name, state))


def corrupt_label(action_bin: int, dataset: str, tau: float, rng: np.random.Generator) -> int:
    if dataset != "offline":
        return int(action_bin)
    if rng.random() < tau:
        return int(1 - action_bin)
    return int(action_bin)


def corrupt_action_labels(action_mat: np.ndarray, tau: float, seed: int) -> np.ndarray:
    action_mat = np.asarray(action_mat, dtype=int)
    rng = np.random.default_rng(seed)
    flip_mask = rng.random(size=action_mat.shape) < tau
    return np.where(flip_mask, 1 - action_mat, action_mat).astype(int)


def derive_offline_dataset_from_oracle(oracle_payload: dict, tau: float, seed: int) -> dict:
    if bool(oracle_payload.get("summary_only", False)):
        raise ValueError("Oracle offline payload must include full trajectories.")
    if "A" not in oracle_payload:
        raise ValueError("Oracle offline payload must contain true actions A.")

    payload = dict(oracle_payload)
    payload["dataset"] = "offline"
    payload["Atilde"] = corrupt_action_labels(payload["A"], tau=tau, seed=seed).tolist()
    return payload


def write_payload(payload: dict, output_path: Path) -> None:
    output_path.write_text(json.dumps(payload), encoding="utf-8")


def rollout_dataset(env_name: str, dataset: str, n_traj: int, horizon: int,
                    tau: float, gamma: float, seed: int,
                    summary_only: bool = False,
                    apply_label_noise: bool = True):
    rng = np.random.default_rng(seed)
    env = gym.make(env_name)

    obs_dim = int(env.observation_space.shape[0])
    if summary_only:
        S = None
        Sp = None
        A = None
        Atilde = None
        R = None
    else:
        S = np.zeros((n_traj, horizon, obs_dim), dtype=float)
        Sp = np.zeros((n_traj, horizon, obs_dim), dtype=float)
        A = np.zeros((n_traj, horizon), dtype=int)
        Atilde = np.zeros((n_traj, horizon), dtype=int)
        R = np.zeros((n_traj, horizon), dtype=float)
    init_states = np.zeros((n_traj, obs_dim), dtype=float)
    discounted_returns = np.zeros(n_traj, dtype=float)

    for i in range(n_traj):
        obs = reset_env(env, seed + i)
        init_states[i] = obs
        absorbing = False
        absorb_state = None

        for t in range(horizon):
            state_t = absorb_state if absorbing else obs
            if not summary_only:
                S[i, t] = state_t

            action_bin = choose_action(dataset, env_name, state_t, rng)
            if dataset == "offline" and not apply_label_noise:
                observed_action = int(action_bin)
            else:
                observed_action = corrupt_label(action_bin, dataset, tau, rng)
            if not summary_only:
                A[i, t] = action_bin
                Atilde[i, t] = observed_action

            if absorbing:
                next_state = np.array(absorb_state, copy=True)
                reward = 0.0
            else:
                next_state, reward, done, _ = step_env(
                    env,
                    binary_to_env_action(env_name, action_bin),
                    rng,
                )
                if done:
                    absorbing = True
                    absorb_state = np.array(next_state, copy=True)

            if not summary_only:
                Sp[i, t] = next_state
                R[i, t] = reward
            discounted_returns[i] += (gamma ** t) * reward
            obs = next_state

    env.close()

    payload = {
        "env_name": env_name,
        "dataset": dataset,
        "summary_only": bool(summary_only),
        "N": int(n_traj),
        "T": int(horizon),
        "state_names": [f"x{j + 1}" for j in range(obs_dim)],
        "init_states": init_states.tolist(),
        "discounted_returns": discounted_returns.tolist(),
    }
    if not summary_only:
        payload.update({
            "S": S.tolist(),
            "A": A.tolist(),
            "Atilde": Atilde.tolist(),
            "R": R.tolist(),
            "Sp": Sp.tolist(),
        })
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Gym datasets for LURE.")
    parser.add_argument("--env", required=True)
    parser.add_argument("--dataset", choices=["offline", "target"], required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--T", type=int, required=True)
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    if args.summary_only and args.dataset != "target":
        raise ValueError("--summary-only is only supported for target datasets.")

    payload = rollout_dataset(
        env_name=args.env,
        dataset=args.dataset,
        n_traj=args.N,
        horizon=args.T,
        tau=args.tau,
        gamma=args.gamma,
        seed=args.seed,
        summary_only=args.summary_only,
    )

    output_path = Path(args.output)
    write_payload(payload, output_path)


if __name__ == "__main__":
    main()