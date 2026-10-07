#!/usr/bin/env Rscript
# Diagnostic runner only: sources the current estimators without modifying them.
# Usage (from Code): Rscript run_state_dependent_check.R Tabular OUTPUT_DIR
#                   Rscript run_state_dependent_check.R Continuous OUTPUT_DIR
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L, args[1] %in% c("Tabular", "Continuous"))
setting <- args[1]
runner <- normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1]))
root <- dirname(runner)
dir.create(args[2], recursive = TRUE, showWarnings = FALSE)
out_dir <- normalizePath(args[2])
result_file <- file.path(out_dir, paste0(tolower(setting), ".rds"))
if (file.exists(result_file)) stop("Refusing to overwrite existing results: ", result_file)
sink(file.path(out_dir, paste0(tolower(setting), ".log")), split = TRUE)
started <- Sys.time()
cat(setting, "started", format(started, tz = "UTC"), "UTC\n")

source_files <- c("state_dependent_error.R", "run_state_dependent_check.R",
                  if (setting == "Tabular") c("Tabular/Methods.R", "Tabular/simulation_tabular.R")
                  else c("Continuous/Methods_continuous.R", "Continuous/baseline.R",
                         "Continuous/simulation_continuous.R"))
snapshot <- file.path(out_dir, "source_snapshot", setting)
for (f in source_files) {
  dest <- file.path(snapshot, f)
  dir.create(dirname(dest), recursive = TRUE, showWarnings = FALSE)
  stopifnot(file.copy(file.path(root, f), dest, overwrite = FALSE))
}
source_md5 <- tools::md5sum(file.path(root, source_files))
setwd(file.path(root, setting))
if (setting == "Tabular") {
  source("Methods.R")
  dgp <- generate_dgp()
  replicate_once <- one_rep
} else {
  source("Methods_continuous.R")
  source("baseline.R")
  # Exactly the configuration in simulation_continuous.R.
  dgp <- generate_dgp_continuous(
    s1_a_int = 0.4, s2_a_int = -0.3,
    init_mean = c(0.25, 0.05), init_sd = c(0.75, 0.75),
    pi_func = function(s1, s2) as.numeric(s1 >= 0.25 & s2 >= -0.10))
  replicate_once <- one_rep_continuous
}
config <- list(setting = setting, N = 50L, TT = 50L, gamma = 0.7,
               n_rep = 20L, seeds = 1:20, epsilon_grid = c(.05, .1, .2, .3),
               misclassification = "state_dependent", state_strength = .75,
               state_center = c(0, 0), state_scale = c(1, 1),
               truth_seed = 2324L, truth_n_mc = 10000L, truth_horizon = 200L,
               ci = "Unmodified estimator IF CI: estimate +/- 1.96 * SE",
               filtering = "None; every replication and nonfinite result retained")
saveRDS(config, file.path(out_dir, paste0(tolower(setting), "_config.rds")))

if (setting == "Tabular") {
  truth <- list(value = compute_true_value(dgp, config$gamma)$V_value,
                mc_se = 0, type = "Exact Bellman solution")
} else {
  # Collect individual returns using the original truth function, not a new DGP.
  # Verify that batching one trajectory at a time preserves both value and RNG.
  set.seed(config$truth_seed)
  check_full <- compute_true_value_continuous(dgp, config$gamma, n_mc = 5, TT_mc = 20)$V_value
  check_rng <- .Random.seed
  set.seed(config$truth_seed)
  check_parts <- vapply(1:5, function(i)
    compute_true_value_continuous(dgp, config$gamma, n_mc = 1, TT_mc = 20)$V_value, numeric(1))
  stopifnot(isTRUE(all.equal(check_full, mean(check_parts), tolerance = 1e-12)),
            identical(check_rng, .Random.seed))
  set.seed(config$truth_seed)
  returns <- vapply(seq_len(config$truth_n_mc), function(i) {
    if (i %% 2500L == 0L) {
      cat("Truth trajectories:", i, "/", config$truth_n_mc, "\n"); flush.console()
    }
    compute_true_value_continuous(dgp, config$gamma,
                                  n_mc = 1, TT_mc = config$truth_horizon)$V_value
  }, numeric(1))
  truth <- list(value = mean(returns), mc_se = sd(returns) / sqrt(length(returns)),
                type = "Monte Carlo target-policy rollouts", returns = returns,
                seed = config$truth_seed, horizon = config$truth_horizon)
}
cat("Reference value:", format(truth$value, digits = 10),
    "; MC SE:", format(truth$mc_se, digits = 5), "\n")
records <- list()
for (eps in config$epsilon_grid) {
  for (rep_id in config$seeds) {
    set.seed(rep_id)
    warnings <- character()
    failure <- NA_character_
    t0 <- proc.time()[["elapsed"]]
    est <- tryCatch(withCallingHandlers(
      replicate_once(dgp, N = config$N, TT = config$TT,
                     epsilon = eps, gamma = config$gamma,
                     misclassification = config$misclassification,
                     state_strength = config$state_strength),
      warning = function(w) {
        warnings <<- c(warnings, conditionMessage(w))
        invokeRestart("muffleWarning")
      }), error = function(e) {
        failure <<- conditionMessage(e)
        setNames(rep(NA_real_, 8L),
                 c("FQE", "SIS", "MIS", "DRL", "LSTD", "MR", "MR_ci_lo", "MR_ci_hi"))
      })
    record <- list(rep = rep_id, epsilon = eps, estimates = est,
                   misclassification = attr(est, "misclassification"),
                   warnings = warnings, error = failure,
                   elapsed_seconds = proc.time()[["elapsed"]] - t0)
    records[[length(records) + 1L]] <- record
    # Checkpoint after every replication; no prior experiment is overwritten.
    saveRDS(list(config = config, truth = truth, records = records,
                 source_md5 = source_md5, started = started, complete = FALSE), result_file)
    cat(sprintf("%s epsilon=%.2f rep=%02d/20: LURE=%.5f CI=[%.5f, %.5f], %.1fs, warnings=%d, nonfinite=%d\n",
                setting, eps, rep_id, est["MR"], est["MR_ci_lo"], est["MR_ci_hi"],
                record$elapsed_seconds, length(warnings), sum(!is.finite(est))))
    flush.console()
  }
}
stopifnot(identical(unname(source_md5), unname(tools::md5sum(file.path(root, source_files)))))
saveRDS(list(config = config, truth = truth, records = records, source_md5 = source_md5,
             started = started, completed = Sys.time(), complete = TRUE,
             session_info = sessionInfo()), result_file)
capture.output(sessionInfo(), file = file.path(out_dir, paste0(tolower(setting), "_session.txt")))
cat("Completed", length(records), "replications; elapsed",
    round(as.numeric(difftime(Sys.time(), started, units = "secs")), 1), "seconds\n")
sink()
