# Gym

This folder contains a Gym-specific version of the LURE pipeline.

Files:

- `Methods_gym.R`: Gym data helpers, EM nuisance estimation, density-ratio estimation, weighted FQE, and the LURE estimator.
- `baseline_gym.R`: naive FQE, MIS, DRL, SIS, and LSTD baselines for the Gym setting.
- `simulation_mountaincar.R`: MountainCar-only experiment script with `N = 50` and `T = 50`.
- `simulation_cartpole.R`: CartPole-only experiment script with `N = 50` and `T = 50`.
- `gym_data.py`: Python data generator that creates two datasets:
  - offline trajectories from a behavior policy with label corruption by `tau`
  - target-policy trajectories for Monte Carlo evaluation

Target policy:

- For CartPole, the current target policy is stochastic with probability `expit(0.10 x + 0.25 x_dot + 1.20 theta + 0.35 theta_dot)` of action `1`.
- For MountainCar, the current target policy remains randomized and state-independent with action `1` probability `0.5`.

Behavior policy:

- For CartPole, the current offline behavior policy is randomized and state-independent with action `1` probability `0.5`.
- For MountainCar, the offline behavior policy remains the clipped logistic policy used in the generator.

Monte Carlo truth evaluation:

- Offline experiments use `N = 50` and `T = 50`.
- Target-policy Monte Carlo evaluation uses `N = 10000` and `T = 2000`.
- Large target-policy evaluations can use a summary-only payload with initial states and discounted returns instead of full trajectories.
- Saved target-policy MC files are keyed by `N`, `T`, `gamma`, and `seed`, since discounted returns depend on `gamma`.

CartPole details:

- The transition now uses the original deterministic CartPole dynamics plus additive Gaussian noise with standard deviation `0.1` on each next-state coordinate.
- The reward is `1 - x^2 / 11.52 - theta^2 / 288` plus independent Gaussian noise with standard deviation `0.1`, where `x` is cart position and `theta` is pole angle.

Dependencies:

- Python: `gymnasium` or `gym`, and `numpy`
- R: `jsonlite`, `dplyr`, and `ggplot2`

Usage:

Run either `simulation_mountaincar.R` or `simulation_cartpole.R` from this folder, or set `options(lure.gym.dir = '<path-to-Gym>')` before sourcing the files from another working directory.