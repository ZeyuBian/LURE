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

State convention:

- `init_states` stores the clean reset observation from the environment.
- The saved current states `S` and next states `Sp` use a recursively noisy-state convention with mean-zero Gaussian state noise of standard deviation `0.05`.
- Both the behavior policy and the target policy act on the noisy current state.
- The next state and reward are generated from the previous noisy state, for both offline data and target-policy Monte Carlo data.

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
- Saved target-policy `init_states` use the same clean-reset convention as the offline data, while discounted returns are still generated from the noisy-state rollout.
- Saved target-policy MC files are keyed by `N`, `T`, `gamma`, and `seed`, since discounted returns depend on `gamma`.

CartPole details:

- CartPole keeps the clean reset observation in `init_states`, but the recorded current-state tensor `S` and next-state tensor `Sp` are the noisy states used recursively by the rollout.
- The transition applies the original CartPole dynamics to the current noisy state, then adds `0.3 a` to each coordinate of the deterministic next-state vector before adding mean-zero Gaussian state noise with standard deviation `0.05`, saving the result, and carrying it forward.
- The reward is `1 - x^2 / 11.52 - theta^2 / 288 + 0.5 a` plus independent Gaussian noise with standard deviation `0.1`, where `x` is cart position and `theta` is pole angle from the current noisy state, and `a` is the true executed binary action.

MountainCar details:

- MountainCar keeps the environment reward definition unchanged.
- Each step is taken from the current noisy state, and the returned next state is perturbed by mean-zero Gaussian state noise with standard deviation `0.05` before being saved and reused.

Bridge-state selection:

- If `bridge_index` is left unspecified in `generate_gym_dgp()`, LURE now selects the bridge next-state coordinate by residual partial correlation: it regresses `Atilde` and each component of `S'` on the current state, then picks the coordinate with the largest absolute residual correlation.
- Passing `bridge_index` still overrides the automatic selector.

Dependencies:

- Python: `gymnasium` or `gym`, and `numpy`
- R: `jsonlite`, `dplyr`, and `ggplot2`

Usage:

Run either `simulation_mountaincar.R` or `simulation_cartpole.R` from this folder, or set `options(lure.gym.dir = '<path-to-Gym>')` before sourcing the files from another working directory.