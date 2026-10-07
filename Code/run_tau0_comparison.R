#!/usr/bin/env Rscript

# Run the current LURE and baseline estimators with no action
# misclassification. This is a diagnostic driver only: estimator source files
# are loaded unchanged.
#
# Usage from Code/:
#   Rscript run_tau0_comparison.R Tabular simulation_results/tau0_20_YYYYMMDD
#   Rscript run_tau0_comparison.R Continuous simulation_results/tau0_20_YYYYMMDD

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L || !(args[1] %in% c("Tabular", "Continuous"))) {
  stop("Usage: Rscript run_tau0_comparison.R {Tabular|Continuous} OUTPUT_DIR")
}

setting <- args[1]
runner_arg <- grep("^--file=", commandArgs(), value = TRUE)[1]
runner <- normalizePath(sub("^--file=", "", runner_arg))
root <- dirname(runner)
dir.create(args[2], recursive = TRUE, showWarnings = FALSE)
out_dir <- normalizePath(args[2])
result_file <- file.path(out_dir, paste0(tolower(setting), ".rds"))
log_file <- file.path(out_dir, paste0(tolower(setting), ".log"))

if (file.exists(result_file)) {
  stop("Refusing to overwrite existing results: ", result_file)
}

sink(log_file, split = TRUE)
on.exit(sink(), add = TRUE)
started <- Sys.time()
cat(setting, "tau=0 run started", format(started, tz = "UTC"), "UTC\n")

source_files <- c(
  "state_dependent_error.R",
  "run_tau0_comparison.R",
  if (setting == "Tabular") {
    "Tabular/Methods.R"
  } else {
    c("Continuous/Methods_continuous.R", "Continuous/baseline.R")
  }
)
source_paths <- file.path(root, source_files)
if (!all(file.exists(source_paths))) {
  stop("Missing source files: ", paste(source_files[!file.exists(source_paths)], collapse = ", "))
}

snapshot <- file.path(out_dir, "source_snapshot", setting)
for (i in seq_along(source_files)) {
  destination <- file.path(snapshot, source_files[i])
  dir.create(dirname(destination), recursive = TRUE, showWarnings = FALSE)
  if (!file.copy(source_paths[i], destination, overwrite = FALSE)) {
    stop("Could not snapshot source file: ", source_files[i])
  }
}
source_md5 <- tools::md5sum(source_paths)

setwd(file.path(root, setting))
if (setting == "Tabular") {
  source("Methods.R")
  dgp <- generate_dgp()
  replicate_once <- one_rep
} else {
  source("Methods_continuous.R")
  source("baseline.R")
  # Matches Continuous/simulation_continuous.R.
  dgp <- generate_dgp_continuous(
    s1_a_int = 0.4,
    s2_a_int = -0.3,
    init_mean = c(0.25, 0.05),
    init_sd = c(0.75, 0.75),
    pi_func = function(s1, s2) as.numeric(s1 >= 0.25 & s2 >= -0.10)
  )
  replicate_once <- one_rep_continuous
}

config <- list(
  setting = setting,
  N = 50L,
  TT = 50L,
  gamma = 0.7,
  n_rep = 20L,
  seeds = 1:20,
  tau = 0,
  misclassification = "constant",
  methods = c("FQE", "SIS", "MIS", "DRL", "LSTD", "MR"),
  design = "Current simulation DGP and estimators; no estimator changes; all outcomes retained"
)

if (setting == "Tabular") {
  truth <- list(
    value = compute_true_value(dgp, config$gamma)$V_value,
    mc_se = 0,
    n = NA_integer_,
    horizon = NA_integer_,
    method = "Exact Bellman solution"
  )
} else {
  # Reuse the repository's high-precision independent reference for this exact
  # production DGP instead of introducing material target Monte Carlo noise.
  reference_file <- file.path(
    root, "simulation_results", "ci_20_20260913", "references",
    "Continuous_baseline.rds"
  )
  if (!file.exists(reference_file)) {
    stop("Continuous reference file is missing: ", reference_file)
  }
  reference <- readRDS(reference_file)
  dgp_fields <- c(
    "sigma_tr", "sigma_R", "b_prob", "tr_shift", "s1_a_int", "s2_a_int",
    "init_mean", "init_sd"
  )
  same_dgp <- all(vapply(dgp_fields, function(field) {
    isTRUE(all.equal(dgp[[field]], reference$dgp[[field]], tolerance = 0))
  }, logical(1)))
  if (!same_dgp || !isTRUE(all.equal(config$gamma, reference$gamma, tolerance = 0))) {
    stop("The saved continuous reference does not match the current simulation DGP.")
  }
  truth <- list(
    value = reference$value,
    mc_se = reference$mcse,
    n = reference$n,
    horizon = reference$horizon,
    seed = reference$seed,
    method = reference$method,
    source = reference_file,
    source_md5 = unname(tools::md5sum(reference_file))
  )
}

cat(
  "Reference value:", format(truth$value, digits = 10),
  "; MC SE:", format(truth$mc_se, digits = 5), "\n"
)
cat(
  "Design: N=", config$N, ", T=", config$TT,
  ", gamma=", config$gamma, ", tau=", config$tau,
  ", replications=", config$n_rep, "\n", sep = ""
)

records <- vector("list", config$n_rep)
for (rep_id in config$seeds) {
  set.seed(rep_id)
  warnings <- character()
  failure <- NA_character_
  t0 <- proc.time()[["elapsed"]]
  est <- tryCatch(
    withCallingHandlers(
      replicate_once(
        dgp,
        N = config$N,
        TT = config$TT,
        epsilon = config$tau,
        gamma = config$gamma,
        misclassification = config$misclassification
      ),
      warning = function(w) {
        warnings <<- c(warnings, conditionMessage(w))
        invokeRestart("muffleWarning")
      }
    ),
    error = function(e) {
      failure <<- conditionMessage(e)
      setNames(
        rep(NA_real_, 8L),
        c("FQE", "SIS", "MIS", "DRL", "LSTD", "MR", "MR_ci_lo", "MR_ci_hi")
      )
    }
  )
  elapsed <- proc.time()[["elapsed"]] - t0
  records[[rep_id]] <- list(
    rep = rep_id,
    estimates = est,
    misclassification = attr(est, "misclassification"),
    warnings = warnings,
    error = failure,
    elapsed_seconds = elapsed
  )
  saveRDS(
    list(
      config = config,
      truth = truth,
      records = records[seq_len(rep_id)],
      source_md5 = source_md5,
      started = started,
      complete = FALSE
    ),
    result_file
  )
  cat(sprintf(
    "%s tau=0 rep=%02d/20: LURE=%.6f CI=[%.6f, %.6f], %.1fs, warnings=%d, nonfinite=%d\n",
    setting, rep_id, est["MR"], est["MR_ci_lo"], est["MR_ci_hi"],
    elapsed, length(warnings), sum(!is.finite(est))
  ))
  flush.console()
}

if (!identical(unname(source_md5), unname(tools::md5sum(source_paths)))) {
  stop("Source files changed while the simulation was running.")
}

completed <- Sys.time()
saveRDS(
  list(
    config = config,
    truth = truth,
    records = records,
    source_md5 = source_md5,
    started = started,
    completed = completed,
    complete = TRUE,
    session_info = sessionInfo()
  ),
  result_file
)
capture.output(
  sessionInfo(),
  file = file.path(out_dir, paste0(tolower(setting), "_session.txt"))
)
cat(
  "Completed", length(records), "replications; elapsed",
  round(as.numeric(difftime(completed, started, units = "secs")), 1), "seconds\n"
)

