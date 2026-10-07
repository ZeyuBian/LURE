#!/usr/bin/env Rscript

# Posterior-imputation (IMP) baseline study for the reviewer request to
# compare against a measurement-error-adjusted approach (AE Major Comment 4,
# Reviewer 1 Comment 3).
#
# For each replication it (i) reruns the existing surrogate-as-truth baselines
# and LURE exactly as the simulation scripts do (same seed, same data) and
# (ii) adds the IMP baseline: impute A from eta(a | O), then run FQE, SIS,
# MIS, DRL, and LSTD on the completed data, averaging over M imputations.
# Estimator source files are loaded unchanged.
#
# Usage from Code/:
#   Rscript run_posterior_imputation.R {Tabular|Continuous|CartPole} OUTPUT_DIR \
#     [N_REP=100] [WORKERS=6] [M=20]
# Set LURE_MISCLASSIFICATION=state_dependent (Tabular/Continuous) for the
# state-dependent recording design. Set LURE_IMP_ONLY=1 to skip LURE and the
# surrogate-as-truth baselines and run only IMP on the same data sets.
# Per-replication files make runs resumable.

args <- commandArgs(trailingOnly = TRUE)
settings <- c("Tabular", "Continuous", "CartPole")
if (length(args) < 2L || !(args[1] %in% settings)) {
  stop("Usage: Rscript run_posterior_imputation.R {Tabular|Continuous|CartPole} OUTPUT_DIR [N_REP] [WORKERS] [M]")
}
setting <- args[1]
n_rep <- if (length(args) >= 3L) as.integer(args[3]) else 100L
workers <- if (length(args) >= 4L) as.integer(args[4]) else 6L
M <- if (length(args) >= 5L) as.integer(args[5]) else 20L
misclassification <- Sys.getenv("LURE_MISCLASSIFICATION", "constant")
state_strength <- as.numeric(Sys.getenv("LURE_STATE_STRENGTH", "0.75"))
imp_only <- Sys.getenv("LURE_IMP_ONLY", "0") %in% c("1", "true", "TRUE")
if (setting == "CartPole" && misclassification != "constant") {
  stop("The CartPole runner uses the saved constant-error datasets only.")
}

Sys.setenv(OPENBLAS_NUM_THREADS = "1", OMP_NUM_THREADS = "1", VECLIB_MAXIMUM_THREADS = "1")

runner_arg <- grep("^--file=", commandArgs(), value = TRUE)[1]
root <- dirname(normalizePath(sub("^--file=", "", runner_arg)))
dir.create(args[2], recursive = TRUE, showWarnings = FALSE)
out_dir <- normalizePath(args[2])
tag <- paste0(tolower(setting), if (misclassification != "constant") "_state_dependent" else "")
result_file <- file.path(out_dir, paste0(tag, ".rds"))
if (file.exists(result_file)) stop("Refusing to overwrite existing results: ", result_file)
rep_dir <- file.path(out_dir, "replications", tag)
dir.create(rep_dir, recursive = TRUE, showWarnings = FALSE)
log_file <- file.path(out_dir, paste0(tag, ".log"))
log_con <- file(log_file, open = "a")
sink(log_con, split = TRUE)
on.exit({ sink(); close(log_con) }, add = TRUE)

started <- Sys.time()
cat(setting, "posterior-imputation run started", format(started, tz = "UTC"), "UTC\n")

setting_dir <- switch(setting, Tabular = "Tabular", Continuous = "Continuous", CartPole = "Gym")
source_files <- c(
  "state_dependent_error.R", "posterior_imputation.R", "run_posterior_imputation.R",
  switch(setting,
         Tabular = c("Tabular/Methods.R", "Tabular/posterior_imputation_tabular.R"),
         Continuous = c("Continuous/Methods_continuous.R", "Continuous/baseline.R",
                        "Continuous/posterior_imputation_continuous.R"),
         CartPole = c("Gym/Methods_gym.R", "Gym/baseline_gym.R",
                      "Gym/posterior_imputation_gym.R"))
)
source_paths <- file.path(root, source_files)
if (!all(file.exists(source_paths))) {
  stop("Missing source files: ", paste(source_files[!file.exists(source_paths)], collapse = ", "))
}
snapshot <- file.path(out_dir, "source_snapshot", tag)
for (i in seq_along(source_files)) {
  destination <- file.path(snapshot, source_files[i])
  dir.create(dirname(destination), recursive = TRUE, showWarnings = FALSE)
  file.copy(source_paths[i], destination, overwrite = TRUE)
}
source_md5 <- tools::md5sum(source_paths)

setwd(file.path(root, setting_dir))
gamma <- 0.7
N <- 50L
TT <- 50L
grid <- c(0.05, 0.10, 0.20, 0.30)
existing_methods <- c("FQE", "SIS", "MIS", "DRL", "LSTD", "MR")

if (setting == "Tabular") {
  source("Methods.R")
  source("posterior_imputation_tabular.R")
  dgp <- generate_dgp()
  truth <- list(value = compute_true_value(dgp, gamma)$V_value, mc_se = 0,
                method = "Exact Bellman solution")
} else if (setting == "Continuous") {
  source("Methods_continuous.R")
  source("baseline.R")
  source("posterior_imputation_continuous.R")
  # Matches Continuous/simulation_continuous.R.
  dgp <- generate_dgp_continuous(
    s1_a_int = 0.4, s2_a_int = -0.3,
    init_mean = c(0.25, 0.05), init_sd = c(0.75, 0.75),
    pi_func = function(s1, s2) as.numeric(s1 >= 0.25 & s2 >= -0.10)
  )
  # High-precision independent reference for this exact DGP (see run_tau0_comparison.R).
  reference_file <- file.path(root, "simulation_results", "ci_20_20260913", "references",
                              "Continuous_baseline.rds")
  reference <- readRDS(reference_file)
  dgp_fields <- c("sigma_tr", "sigma_R", "b_prob", "tr_shift", "s1_a_int", "s2_a_int",
                  "init_mean", "init_sd")
  same_dgp <- all(vapply(dgp_fields, function(f) {
    isTRUE(all.equal(dgp[[f]], reference$dgp[[f]], tolerance = 0))
  }, logical(1)))
  if (!same_dgp || !isTRUE(all.equal(gamma, reference$gamma, tolerance = 0))) {
    stop("The saved continuous reference does not match the current simulation DGP.")
  }
  truth <- list(value = reference$value, mc_se = reference$mcse, method = reference$method,
                source = reference_file)
} else {
  options(lure.gym.dir = getwd())
  source("Methods_gym.R")
  source("baseline_gym.R")
  source("posterior_imputation_gym.R")
  dgp <- generate_gym_dgp("CartPole-v1")
  offline_data_dir <- resolve_offline_gym_data_dir(getwd())
  truth_file <- file.path(out_dir, "cartpole_truth.rds")
  if (file.exists(truth_file)) {
    truth <- readRDS(truth_file)
  } else {
    # Matches Gym/simulation_cartpole.R: 10,000 target trajectories, horizon 400.
    truth_out <- estimate_true_value_gym(dgp, N = 10000L, TT = 400L, gamma = gamma,
                                         seed = 10002L)
    truth <- list(value = truth_out$V_value, mc_se = truth_out$mc_se,
                  method = "Target-policy Monte Carlo (seed 10002)",
                  target_x = getOption("lure.cartpole.target_x", -2.4),
                  target_theta = getOption("lure.cartpole.target_theta", .2))
    saveRDS(truth, truth_file)
  }
}
cat("Reference value:", format(truth$value, digits = 10), "; MC SE:",
    format(truth$mc_se, digits = 5), "\n")
cat("Design: N=", N, ", T=", TT, ", gamma=", gamma, ", replications=", n_rep,
    ", imputations M=", M, ", misclassification=", misclassification,
    if (imp_only) ", IMP only" else "", "\n", sep = "")

run_one <- function(eps, rep_id) {
  rep_file <- file.path(rep_dir, sprintf("eps_%.2f_rep_%03d.rds", eps, rep_id))
  if (file.exists(rep_file)) return(invisible(rep_file))
  warnings <- character()
  record_warning <- function(w) {
    warnings <<- c(warnings, conditionMessage(w))
    invokeRestart("muffleWarning")
  }
  t0 <- proc.time()[["elapsed"]]
  failure <- NA_character_
  existing <- setNames(rep(NA_real_, length(existing_methods) + 2L),
                       c(existing_methods, "MR_ci_lo", "MR_ci_hi"))
  imp <- NULL
  res <- tryCatch(withCallingHandlers({
    est <- existing
    if (setting == "Tabular") {
      set.seed(rep_id)
      if (!imp_only) {
        est <- one_rep(dgp, N = N, TT = TT, epsilon = eps, gamma = gamma,
                       misclassification = misclassification, state_strength = state_strength)
      }
      set.seed(rep_id)  # identical data: generation is one_rep()'s first RNG use
      dat <- generate_data(dgp, N, TT, eps, misclassification, state_strength)
      set.seed(1000000L + rep_id)
      imp <- imp_estimator_tabular(dat, dgp, gamma, M = M)
    } else if (setting == "Continuous") {
      set.seed(rep_id)
      if (!imp_only) {
        est <- one_rep_continuous(dgp, N = N, TT = TT, epsilon = eps, gamma = gamma,
                                  misclassification = misclassification,
                                  state_strength = state_strength)
      }
      set.seed(rep_id)
      dat <- generate_data_continuous(dgp, N, TT, eps, misclassification, state_strength)
      set.seed(1000000L + rep_id)
      imp <- imp_estimator_continuous(dat, dgp, gamma, M = M)
    } else {
      path <- offline_gym_dataset_path(dgp, tau = eps, rep = rep_id, data_dir = offline_data_dir)
      if (!file.exists(path)) stop("Missing offline dataset: ", path)
      dat <- load_gym_dataset(path)
      if (!imp_only) est <- evaluate_gym_estimators(dat, dgp, gamma, seed = rep_id)
      set.seed(1000000L + rep_id)
      imp <- imp_estimator_gym(dat, dgp, gamma, M = M)
    }
    existing[names(existing)] <- est[names(existing)]
    TRUE
  }, warning = record_warning), error = function(e) { failure <<- conditionMessage(e); FALSE })

  record <- list(
    setting = setting, misclassification = misclassification, epsilon = eps, rep = rep_id,
    estimates = c(existing[existing_methods],
                  if (is.null(imp)) setNames(rep(NA_real_, 5L),
                                             paste0("IMP-", c("FQE", "SIS", "MIS", "DRL", "LSTD")))
                  else imp$estimates),
    lure_ci = existing[c("MR_ci_lo", "MR_ci_hi")],
    imp_between_sd = if (is.null(imp)) NULL else imp$between_sd,
    imp_diagnostics = if (is.null(imp)) NULL else imp$diagnostics,
    imp_state_mu = if (is.null(imp)) NULL else imp$state_mu,
    M = M, elapsed = proc.time()[["elapsed"]] - t0,
    warnings = unique(warnings), failure = failure
  )
  saveRDS(record, rep_file)
  invisible(rep_file)
}

tasks <- expand.grid(rep = seq_len(n_rep), eps = grid)
tasks <- tasks[order(tasks$rep, tasks$eps), ]
done <- file.exists(file.path(rep_dir, sprintf("eps_%.2f_rep_%03d.rds", tasks$eps, tasks$rep)))
cat("Tasks:", nrow(tasks), "; already done:", sum(done), "; workers:", workers, "\n")

todo <- which(!done)
chunks <- split(todo, ceiling(seq_along(todo) / (4L * workers)))
for (ch in chunks) {
  parallel::mclapply(ch, function(i) run_one(tasks$eps[i], tasks$rep[i]),
                     mc.cores = workers, mc.preschedule = FALSE)
  n_done <- sum(file.exists(file.path(rep_dir, sprintf("eps_%.2f_rep_%03d.rds",
                                                       tasks$eps, tasks$rep))))
  cat(format(Sys.time(), "%H:%M:%S"), " completed ", n_done, "/", nrow(tasks), "\n", sep = "")
}

files <- file.path(rep_dir, sprintf("eps_%.2f_rep_%03d.rds", tasks$eps, tasks$rep))
missing <- !file.exists(files)
if (any(missing)) stop(sum(missing), " replications are missing; rerun to resume.")
records <- lapply(files, readRDS)

estimates <- do.call(rbind, lapply(records, function(r) {
  data.frame(rep = r$rep, epsilon = r$epsilon, method = names(r$estimates),
             estimate = unname(r$estimates), stringsAsFactors = FALSE)
}))
if (imp_only) estimates <- estimates[grepl("^IMP-", estimates$method), ]
lure_ci <- do.call(rbind, lapply(records, function(r) {
  data.frame(rep = r$rep, epsilon = r$epsilon, ci_lo = r$lure_ci[[1]], ci_hi = r$lure_ci[[2]])
}))
lure_ci$covers <- lure_ci$ci_lo <= truth$value & truth$value <= lure_ci$ci_hi
diagnostics <- do.call(rbind, lapply(records, function(r) {
  if (is.null(r$imp_diagnostics)) return(NULL)
  cbind(data.frame(rep = r$rep, epsilon = r$epsilon, elapsed = r$elapsed,
                   n_warnings = length(r$warnings)),
        as.data.frame(as.list(r$imp_diagnostics)))
}))
failures <- do.call(rbind, lapply(records, function(r) {
  if (is.na(r$failure)) return(NULL)
  data.frame(rep = r$rep, epsilon = r$epsilon, failure = r$failure)
}))

out <- list(
  setting = setting, misclassification = misclassification, state_strength = state_strength,
  imp_only = imp_only, truth = truth, estimates = estimates, lure_ci = lure_ci, diagnostics = diagnostics,
  failures = failures, records = records,
  config = list(N = N, TT = TT, gamma = gamma, n_rep = n_rep, grid = grid, M = M,
                seeds = "set.seed(rep) for data and existing methods; set.seed(1e6 + rep) for IMP"),
  source_md5 = source_md5, started = started, finished = Sys.time()
)
saveRDS(out, result_file)
write.csv(estimates, file.path(out_dir, paste0(tag, "_estimates.csv")), row.names = FALSE)

summ <- aggregate(estimate ~ epsilon + method, data = estimates, FUN = function(x) {
  c(bias = mean(x) - truth$value, sd = sd(x), rmse = sqrt(mean((x - truth$value)^2)))
})
summ <- cbind(summ[c("epsilon", "method")], as.data.frame(summ$estimate))
print(summ[order(summ$epsilon, summ$method), ], digits = 3, row.names = FALSE)
if (!imp_only) {
  cat("LURE coverage:\n")
  print(aggregate(covers ~ epsilon, data = lure_ci, FUN = mean))
}
cat("Failures:", if (is.null(failures)) 0 else nrow(failures), "\n")
cat("Finished", format(Sys.time(), tz = "UTC"), "UTC; elapsed",
    round(as.numeric(difftime(Sys.time(), started, units = "mins")), 1), "minutes\n")
