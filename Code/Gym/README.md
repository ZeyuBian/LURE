# Gym

This folder contains a Gym-specific version of the LURE pipeline.

Files:

- `Methods_gym.R`: Gym data helpers, EM nuisance estimation, density-ratio estimation, weighted FQE, and the LURE estimator.
- `baseline_gym.R`: naive FQE, MIS, DRL, SIS, and LSTD baselines for the Gym setting.
- `simulation_cartpole.R`: CartPole-only experiment script with `N = 50` and `T = 50`.
- `state_dependent_error.R`: bounded, current-position-dependent action-recording errors and independent calibration.
- `run_state_dependent_example.R`: reproducible paired 10-replication example with the updated IF CI.
- `report_state_dependent_example.R`: summaries, diagnostics, and scientific figures for that example.
- `gym_data.py`: Python data generator that creates two datasets:
  - offline trajectories from a behavior policy with label corruption by `tau`
  - target-policy trajectories for Monte Carlo evaluation

State convention:

- `init_states` stores the clean reset observation from the environment.
- The saved current states `S` and next states `Sp` use a recursively noisy-state convention with mean-zero Gaussian state noise of standard deviation `0.05`.
- Both the behavior policy and the target policy act on the noisy current state.
- The next state and reward are generated from the previous noisy state, for both offline data and target-policy Monte Carlo data.

Target policy:

- The updated R CartPole policy chooses action `1` iff `x > -2.4` and `theta < 0.2`. R-generated target MC now explicitly passes those same thresholds to Python. Options `lure.cartpole.target_x` and `lure.cartpole.target_theta` configure both together.
- Direct Python CLI calls retain the previous defaults `x > -0.5`, `theta < 0.1`; pass `--target-x-threshold -2.4 --target-theta-threshold 0.2` to match the updated R defaults. Target MC paths now include the policy thresholds to avoid reusing a mismatched target.
- For MountainCar, the current target policy remains randomized and state-independent with action `1` probability `0.5`.

Behavior policy:

- For CartPole, the current offline behavior policy is randomized and state-independent with action `1` probability `0.5`.
- For MountainCar, the offline behavior policy remains the clipped logistic policy used in the generator.

Monte Carlo truth evaluation:

- Offline experiments use `N = 50` and `T = 50`.
- Target-policy Monte Carlo evaluation uses `N = 10000` and `T = 400`.
- Large target-policy evaluations can use a summary-only payload with initial states and discounted returns instead of full trajectories.
- Saved target-policy `init_states` use the same clean-reset convention as the offline data, while discounted returns are still generated from the noisy-state rollout.
- Saved target-policy MC files are keyed by `N`, `T`, `gamma`, and `seed`, since discounted returns depend on `gamma`.

CartPole details:

- CartPole keeps the clean reset observation in `init_states`, but the recorded current-state tensor `S` and next-state tensor `Sp` are the noisy states used recursively by the rollout.
- The transition applies the original CartPole dynamics to the current noisy state, then adds `0.8 a` to each coordinate of the deterministic next-state vector before adding mean-zero Gaussian state noise with standard deviation `0.05`, saving the result, and carrying it forward.
- The reward is `1 - x^2 / 11.52 - theta^2 / 288 + 0.8 a` plus independent Gaussian noise with standard deviation `0.2`, where `x` is cart position and `theta` is pole angle from the current noisy state, and `a` is the true executed binary action. These values describe the uploaded Python code; this extension does not change them.

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

Run `simulation_cartpole.R` from this folder, or set `options(lure.gym.dir = '<path-to-Gym>')` before sourcing the files from another working directory. `LURE_PYTHON` or `options(lure.python=...)` can select a Python environment with Gym and NumPy installed.

## State-dependent CartPole example (10 replications)

From the repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  Rscript Code/Gym/run_state_dependent_example.R \
  Code/simulation_results/cartpole_state_example_new 10 2
```

The final arguments are outer replications and parallel workers. Use a fresh output
folder. The runner uses N=50,T=50, rates .05,.10,.20,.30, and both constant and
state-dependent errors on the same trajectories and recording uniforms. It keeps
the uploaded nuisance fitting and IF-based CI calculation unchanged; no bootstrap
or SE inflation is used. All seven point estimators (including DIRECT) and every
IF interval, warning, and failed fit are retained.

The state-dependent mechanism is

```text
p(flip | S) = tau + strength * min(tau, 0.5-tau) * tanh((x-center)/scale)
strength = 0.75
```

Only the current cart position enters this probability. For tau < .5 and
strength < 1 it is bounded below .5. Setting strength to zero recovers constant
recording error. The example fixes scale to the position IQR in 1,000 independent
calibration trajectories, and solves for center so the calibration mean of the
tanh score is zero. Thus the calibration average flip probability equals tau;
actual rates fluctuate on independent datasets. Neither outcomes nor method
performance are used to calibrate the mechanism.

To choose another matched target for the example, set `LURE_CARTPOLE_TARGET_X`
and `LURE_CARTPOLE_TARGET_THETA`. Without overrides, the updated R policy is used.

For existing in-memory CartPole data, source `state_dependent_error.R` and call
`cartpole_record_errors(dat, tau, uniforms, center=..., scale=...)`. Python generation
also accepts `--misclassification state_dependent --state-strength .75
--error-center ... --error-scale ...`. These Python defaults are not independently
calibrated; use the center/scale recorded by the example or your own independent
calibration sample when comparing nominal marginal rates.

`generate_offline_batch.py --envs CartPole-v1 --misclassification state_dependent`
saves data under `CartPole-v1/state_dependent/tau_*/`, separate from existing constant
datasets. It refuses to overwrite existing files. To use those files in the standard
simulator, set `LURE_MISCLASSIFICATION=state_dependent`; its results are saved separately
as `res_cartpole_state_dependent.RData`.

The JSON loader now preserves trajectory/time/coordinate order when transferring
Python arrays into R. This necessary alignment correction affects how old files are
read but does not modify their contents. The known clean-reset/noisy-initial-state
convention remains a limitation, and ten replications cannot establish nominal coverage.

Checks (run the R check from `Code/`):

```sh
Rscript tests/test_cartpole_state_error.R
python -m unittest discover -s tests -p 'test_cartpole_state_error.py' -v
```

To generate multiple offline datasets from the same clean rollout with one command, use `gym_data.py --dataset offline --taus ...`. If `--output` contains `{tau}`, the placeholder is replaced per corruption level; otherwise files are written under `tau_<value>/seed_<seed>.json` below the given output directory.

Example:

`python3 gym_data.py --env CartPole-v1 --dataset offline --N 50 --T 50 --taus 0.05 0.10 0.20 0.30 --seed 1 --output './CartPole-v1/tau_{tau}/rep_001.json'`
