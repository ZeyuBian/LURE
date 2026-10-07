#!/usr/bin/env Rscript
# Improve reference-value precision only; do not refit any simulation estimator.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1L)
out_dir <- normalizePath(args[1])
result_file <- file.path(out_dir, "continuous_reference_100000.rds")
if (file.exists(result_file)) stop("Refusing to overwrite existing reference")
run <- readRDS(file.path(out_dir, "continuous.rds"))
stopifnot(isTRUE(run$complete))
snapshot <- file.path(out_dir, "source_snapshot", "Continuous")
setwd(file.path(snapshot, "Continuous"))
source("Methods_continuous.R")
dgp <- generate_dgp_continuous(
  s1_a_int = 0.4, s2_a_int = -0.3,
  init_mean = c(0.25, 0.05), init_sd = c(0.75, 0.75),
  pi_func = function(s1, s2) as.numeric(s1 >= 0.25 & s2 >= -0.10))
set.seed(run$config$truth_seed)
started <- Sys.time()
returns <- vapply(seq_len(100000L), function(i) {
  if (i %% 25000L == 0L) { cat("Reference rollouts:", i, "/ 100000\n"); flush.console() }
  compute_true_value_continuous(dgp, run$config$gamma,
                                n_mc = 1, TT_mc = run$config$truth_horizon)$V_value
}, numeric(1))
stopifnot(identical(returns[seq_along(run$truth$returns)], run$truth$returns))
reference <- list(value = mean(returns), mc_se = sd(returns)/sqrt(length(returns)),
                  type = "Monte Carlo target-policy rollouts (precision refinement)",
                  returns = returns, seed = run$config$truth_seed,
                  horizon = run$config$truth_horizon, n_mc = length(returns),
                  original_value = run$truth$value, original_mc_se = run$truth$mc_se,
                  original_returns_verified = TRUE, started = started, completed = Sys.time())
saveRDS(reference, result_file)
cat(sprintf("Refined reference: %.9f, MC SE: %.9f\n", reference$value, reference$mc_se))
