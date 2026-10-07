Proposed numerical experiments for Algorithm 2

Execution update, 2026-09-13: the user authorized 20 independent simulation datasets per setting and explicitly excluded Gym. The executed study uses Tabular and Continuous only, with 20 datasets in each main and stress cell. The stopping/tuning and value subsets remain the first 10 datasets per main cell. The original proposed design below is retained as design history; the run configuration and final report document the actual experiment.

The primary aim is to address the computational assessment comment and Reviewer 1 Comments 4 and 6: numerical convergence, initialization sensitivity, label alignment, and stopping/tuning choices. Most experiments concern the generative nuisance stage. A smaller downstream study evaluates the consequences for LURE. This plan does not replace the separate response to conditional-independence violations, nuisance convergence rates, or real-data sensitivity.

1. Establish the implementation being evaluated

- Freeze source versions, DGP parameters, seeds, and target policies before the main runs. Match the actual simulation drivers, rather than relying on default arguments or old comments in the code. The recent CartPole report uses a different target from the manuscript, so reconcile that specification before using its output as manuscript evidence.
- Add diagnostic output and explicit initialization inputs to each nuisance fitter. Retain every start, its final model, raw component orientation, iteration trace, warnings, status, and elapsed time. Existing wrappers select the best start and discard most of this information; the tabular fitter has one fixed start.
- Check that reported likelihoods, posteriors, and model predictions correspond to the same iteration. Record posterior clipping and other numerical safeguards.
- Compare the implemented label rule with the manuscript rule. The manuscript averages fitted measurement probabilities over a common state distribution; current code uses posterior-weighted surrogate proportions within components. These quantities need not agree when the state distributions differ. Report both in the diagnostic pilot, then make any chosen correction explicit and freeze the rule before the main experiment.
- Preserve current stopping criteria as the baseline. Do not silently turn the diagnostic study into a different estimator. Any proposed changes receive a separately labeled comparison.
- Verify a deliberate global permutation of all action-indexed quantities, including behavior probabilities, leaves the observed model unchanged and gives the same aligned output. In tabular models also inspect state-specific orientation: a global swap cannot repair inconsistent swaps across states.

2. Main experiment: convergence and initialization

Use all three simulation environments: Tabular, Continuous, and CartPole. Use the original nominal constant misclassification levels tau = 0.10 and 0.30 to represent typical and more difficult cases. Keep N = 50 trajectories and T = 50. Use 50 independent datasets per environment and error level.

This standalone nuisance experiment fits the full dataset. Record this training size explicitly. When checking the complete estimator later, respect its actual trajectory-level training folds; full-data nuisance fits cannot substitute for off-fold fits.

For each dataset, use eight specified initializations:

- Three surrogate-informed starts: posterior probability that A equals the recorded action is 0.55, 0.70, or 0.90.
- Two perturbed surrogate-informed starts: the corresponding probability is independently uniform on [0.55, 0.85], with two fixed seed streams.
- Two unanchored random starts: P(A=1 | O) is independently uniform on [0.10, 0.90], without using the recorded action, with two other seed streams.
- One deliberate reversal of the 0.70 start. This is a permutation check, not an additional independent search for a different solution. Show its results separately when summarizing initialization performance.

Use the same dataset across starts. Share underlying simulated trajectories and recording uniforms across error levels where the DGP permits, and retain the pairing in comparisons. Keep data-generation, initialization, and downstream integration seeds separate. Do not initialize all observations at exactly 0.5: identical components can persist under symmetric updates.

Core cost: 3 environments x 2 error levels x 50 datasets x 8 starts = 2,400 nuisance fits. No Q-function or density-ratio estimation is needed in this block.

For each iteration, save the average observed working log-likelihood and maximum and RMS change in posterior responsibilities. Save fitted-prediction changes where feasible. Report any objective decreases; the regression, clipping, and regularization steps do not automatically establish classical EM likelihood ascent.

For each completed start, report:

- Stopping status: criterion met, iteration cap, numerical failure; iteration count and elapsed time.
- Final objective, posterior discrepancy across starts after practical label alignment, and differences in measurement, reward, and transition predictions on a common independent evaluation sample.
- Nuisance RMSE against known simulation truth, using the same evaluation states across starts and separate scales for different outcomes. In tabular models also report transition-probability error. Where the true posterior is computable, compare with that posterior. A Brier score against realized latent A is a predictive score, not a direct estimate of error relative to the true posterior.
- The distribution of fitted measurement contrast mu(s,1)-mu(s,0), its sign violations, and the absolute transition-proxy contrast. Record lower quantiles and the frequency of near-zero contrasts, alongside continuous plots. A large average contrast can conceal weak separation in individual states.
- The result selected by the current restart rule, plus the default single start. Any proposed rule selecting the largest objective among converged candidates is a separate strategy. If no candidate converges, report that failure explicitly. Never select starts using true nuisance error or closeness to the true value.

3. Label alignment and weak-separation stress tests

On Tabular and Continuous, add three targeted settings, each with 50 datasets and the same eight starts:

- Weak surrogate orientation: tau = 0.45, with other DGP components unchanged. Under symmetric constant error, the true measurement contrast is 1-2*tau = 0.10. This weakens the label anchor while retaining the original reward and transition signals.
- Weak transition proxy: tau = 0.30 and reduce action-specific transition differences to one quarter of their baseline size, keeping their midpoint fixed. For tabular transitions use P_a^(rho) = P_bar + rho*(P_a-P_bar), rho = 0.25, where P_bar = (P_0+P_1)/2. For Continuous interpolate the two conditional transition means in the same way, retaining the noise distribution. Keep rewards unchanged. This tests weak transition information separately from weak surrogate orientation.
- State-dependent recording: reuse the established state-dependent mechanism with reference tau = 0.30 and strength 0.75. Report its actual error rates and state-specific contrasts. It is included because an existing tabular run exhibited near-zero fitted separation and an extreme value estimate.

Additional cost: 2 environments x 3 stress settings x 50 datasets x 8 starts = 2,400 nuisance fits. The full core nuisance design therefore has 4,800 fits. These are separate stress tests, not a factorial combination of every difficulty.

Distinguish three outcomes:

- A raw numerical reversal that the practical rule corrects.
- Incorrect orientation remaining after the practical rule.
- Unresolved or locally inconsistent components, for which calling the fit merely a global label swap would be misleading.

Use simulation truth only to evaluate alignment. Where the true posterior is available, compare both global orientations against it on an independent sample; report the loss gap and classify near ties as ambiguous. Inspect state-specific orientation in Tabular. Also show truth-based nuisance error under the practical orientation and under the better global permutation. This separates orientation error from poor component recovery without treating the oracle permutation as the deployed estimator.

Report rates by initialization strategy and by the deployed restart strategy. Datasets are the independent replication units; starts on the same dataset are dependent. Supply uncertainty intervals for failure and alignment rates. With 50 independent datasets and zero failures, the one-sided 95% binomial upper bound is about 5.8%, so the study cannot establish a failure risk below 1%. Such a claim would need at least 299 independent datasets with zero failures, under a fixed strategy. Any extension should be motivated by a prespecified precision target, rather than stopped when a favorable rate appears.

4. Stopping and tuning sensitivity

Use the first 10 prespecified datasets in each of the six main cells. For the default start and one fixed unanchored random start, continue the identical update path to 500 iterations, unless a numerical failure prevents continuation. Save checkpoints sufficient to reconstruct the first eligible stopping point under each rule.

Compare posterior-change tolerances 1e-3, 1e-4, and 1e-5, with caps of 100, 300, and 500; include the existing CartPole cap of 90 as its production reference. Evaluate the current posterior-only rule and a candidate joint rule requiring posterior change below tolerance and absolute average-log-likelihood change below 1e-6 for three consecutive iterations. These are proposed numerical thresholds, not a theorem about convergence.

For every candidate stopping point, assess iteration savings and subsequent drift in posteriors, nuisance predictions, and objective along the longer path. If no stable reference is reached at 500 iterations, label it unresolved rather than treating iteration 500 as truth. Checkpoint replay is valid only if the updates and randomness are identical and independent of the stopping configuration.

On the same subset, compare probability-clipping bounds at half, baseline, and twice their current size, one change at a time. For CartPole, compare spline degrees of freedom 4, 6, and 8 with cubic degree fixed. These are actual nuisance-stage tuning choices. Density-ratio regularization, Q-function fitting, and cross-fitting folds belong to the downstream estimator and should not be presented as Algorithm 2 convergence controls.

5. Limited downstream value check

Use the first 10 prespecified datasets in each main environment/error cell: 60 datasets total. Compute the complete LURE estimator for each of the seven non-reversal starts and for the selected restart strategy. The latter can reuse candidate results when training folds and fitted components match. Hold target policy, trajectory folds, bridge-selection procedure, and downstream integration draws fixed within each dataset. Recompute Q and omega for each corresponding nuisance fit; holding these fixed would understate initialization effects.

Cache generative nuisance fits by exact training indices, initialization, and configuration. Reuse only matching caches. If the complete estimator uses cross-fitting, generate and cache its training-fold nuisance fits separately. Record actual fold counts; the existing Tabular implementation explicitly has no cross-fitting, while Continuous uses two folds. A consistency change to the estimator's cross-fitting procedure must be handled explicitly, not slipped into this experiment.

For each dataset, report the range and SD of the aligned value estimates across starts, their paired differences from the default and selected strategy, and the selected strategy's reported IF standard error. Compare initialization spread with both that standard error and the empirical across-dataset variability, without treating the current IF standard error as established correct in a setting where diagnostics disagree. Retain extreme values and failed downstream fits.

Also analyze the already reported extreme Tabular state-dependent replication (seed 12, reference tau = 0.30) as an explicitly identified diagnostic case. Compare its starting points, convergence, state-specific separation, and resulting values. It is not an additional randomly chosen replication and must not be used to estimate population failure rates.

The small value block estimates sensitivity to initialization; it cannot establish nominal coverage. Broad nuisance stability also cannot explain away existing CartPole bias or poor coverage. If a new bias/coverage claim under weak separation is required, add a separately powered value experiment for a few fixed stress settings. Recompute the true target when transition dynamics change; changing only recorded-action error leaves the true target unchanged.

6. Execution and reporting

Begin with three independent pilot datasets per environment at tau = 0.10 and 0.30: 144 nuisance fits. Use the pilot for runtime measurement, trace validation, label-permutation checks, and specification reconciliation. Use separate seed ranges for the fixed main study. The pilot precedes any commitment to a wall-clock estimate.

Produce one convergence figure, one table of convergence/initialization and alignment results, one weak-separation figure, one compact stopping/tuning table, and one small value-sensitivity table. Show selected illustrative trajectories under a fixed rule (for example the first dataset), together with aggregate results, rather than choosing only the smoothest traces. Retain full machine-readable outcomes and source snapshots.

Frame conclusions separately: numerical stabilization of the iteration; reproducibility across starting points; accuracy of latent-model recovery; and stability of the downstream value estimate. None of these alone establishes global convergence or the nuisance convergence-rate assumptions in the asymptotic theorem.

Narrow the future-tense promises in response.tex to match this design. Proposed wording: "We will assess Algorithm 2 through repeated nuisance-level convergence and initialization experiments across all three simulation environments, with targeted weak-separation and label-alignment analyses. A smaller paired experiment will examine the effect of initialization on the final policy-value estimate."

Reference for simulation planning and reporting Monte Carlo uncertainty: Morris, White, and Crowther (2019), Using simulation studies to evaluate statistical methods, https://pmc.ncbi.nlm.nih.gov/articles/PMC6492164/.
