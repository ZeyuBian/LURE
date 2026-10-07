# State-dependent errors in the simulation folders

State-dependent action recording is supported in **Tabular**, **Continuous**, and
**Gym (CartPole)**. MIMIC is unchanged and excluded, as requested. Existing fitted
estimators, nuisance procedures, and influence-function confidence intervals are
not modified. Constant errors remain the default for backward compatibility.

The common construction is

```text
p(flip | current state) = epsilon + amplitude * score(current state)
amplitude = state_strength * min(epsilon, 0.5-epsilon)
```

`state_strength` defaults to 0.75 and must be in [0,1); state-dependent epsilon
must be below 0.5. Scores lie in [-1,1], so probabilities remain nonnegative and
strictly below 0.5. Strength zero recovers the constant-error data exactly.
Only the recorded action changes: the generators retain the same RNG draws,
true actions, transitions, and rewards for paired seeds.

| Folder | Current-state score | How to enable |
| --- | --- | --- |
| Tabular | -1, 0, +1 for states 1, 2, 3 | `misclassification="state_dependent"` in `generate_data()` or `one_rep()` |
| Continuous | `tanh(((S1-c1)/s1+(S2-c2)/s2)/2)`; default centers 0 and scales 1 | Same argument in `generate_data_continuous()` or `one_rep_continuous()` |
| Gym / CartPole | `tanh((position-center)/scale)` | Existing `state_dependent_error.R` helper, R generator arguments, or Python CLI |

For example, at epsilon=0.2 and strength=0.75, the Tabular probabilities are
0.05, 0.20, and 0.35 in states 1, 2, and 3. For Tabular and Continuous, epsilon
is a reference level, **not necessarily the marginal error rate**. The generators
and simulation outputs record expected and realized rates, minimum/maximum
probabilities, and mean state scores. Continuous `state_center` and `state_scale`
can be fixed from an independent calibration sample; do not choose them to obtain
favorable coverage. The earlier CartPole example already calibrates independently.

## Run ten replications

From the respective simulation folder:

```sh
# From Code/Tabular
LURE_MISCLASSIFICATION=state_dependent LURE_STATE_STRENGTH=0.75 LURE_N_REP=10 \
  Rscript simulation_tabular.R

# From Code/Continuous
LURE_MISCLASSIFICATION=state_dependent LURE_STATE_STRENGTH=0.75 LURE_N_REP=10 \
  Rscript simulation_continuous.R
```

These run the existing four error rates and retain the existing simulation
designs. The state-dependent outputs use separate filenames:

- Tabular: `res_state_dependent_strength_0.75.RData`
- Continuous: `res_continuous_state_dependent_strength_0.75.RData`

Set `LURE_OUTPUT_FILE` to a new file for additional runs you want to keep.
The original constant-error results are not overwritten by a state-dependent run.
`sim_out$misclassification_diagnostics` contains the recording-error summaries.
No bootstrap inference is used. A small example cannot establish 95% coverage.

For CartPole, see [Gym/README.md](Gym/README.md) and the existing paired ten-run
example. No MountainCar simulations are added or run.

## Checks

From `Code/`:

```sh
Rscript tests/test_all_state_errors.R
Rscript tests/test_cartpole_state_error.R
python -m unittest discover -s tests -p 'test_cartpole_state_error.py' -v
```
