#!/usr/bin/env Rscript

# Oracle benchmark for the posterior-imputation study: the same standard OPE
# estimators (FQE, SIS, MIS, DRL, LSTD, same tuning) given the TRUE action, on
# the same data sets as run_posterior_imputation.R. Comparing ORACLE-* with
# IMP-* separates the cost of latent-action recovery from the estimators' own
# finite-sample error (AE Major Comment 4).
#
# Usage from Code/:
#   Rscript run_oracle_true_action.R {Tabular|Continuous|CartPole} OUTPUT_DIR [N_REP=100] [WORKERS=2]

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2L || !(args[1] %in% c("Tabular", "Continuous", "CartPole"))) {
  stop("Usage: Rscript run_oracle_true_action.R {Tabular|Continuous|CartPole} OUTPUT_DIR [N_REP] [WORKERS]")
}
setting <- args[1]
n_rep <- if (length(args) >= 3L) as.integer(args[3]) else 100L
workers <- if (length(args) >= 4L) as.integer(args[4]) else 2L
misclassification <- Sys.getenv("LURE_MISCLASSIFICATION", "constant")
state_strength <- as.numeric(Sys.getenv("LURE_STATE_STRENGTH", "0.75"))

runner_arg <- grep("^--file=", commandArgs(), value = TRUE)[1]
root <- dirname(normalizePath(sub("^--file=", "", runner_arg)))
out_dir <- normalizePath(args[2])
tag <- paste0(tolower(setting), if (misclassification != "constant") "_state_dependent" else "")
result_file <- file.path(out_dir, paste0(tag, "_oracle.rds"))
if (file.exists(result_file)) stop("Refusing to overwrite existing results: ", result_file)

setwd(file.path(root, switch(setting, Tabular = "Tabular", Continuous = "Continuous", CartPole = "Gym")))
gamma <- 0.7
N <- 50L
TT <- 50L
grid <- c(0.05, 0.10, 0.20, 0.30)

if (setting == "Tabular") {
  source("Methods.R")
  source("posterior_imputation_tabular.R")
  dgp <- generate_dgp()
} else if (setting == "Continuous") {
  source("Methods_continuous.R")
  source("baseline.R")
  source("posterior_imputation_continuous.R")
  dgp <- generate_dgp_continuous(
    s1_a_int = 0.4, s2_a_int = -0.3,
    init_mean = c(0.25, 0.05), init_sd = c(0.75, 0.75),
    pi_func = function(s1, s2) as.numeric(s1 >= 0.25 & s2 >= -0.10)
  )
} else {
  options(lure.gym.dir = getwd())
  source("Methods_gym.R")
  source("baseline_gym.R")
  source("posterior_imputation_gym.R")
  dgp <- generate_gym_dgp("CartPole-v1")
  offline_data_dir <- resolve_offline_gym_data_dir(getwd())
}

run_one <- function(eps, rep_id) {
  if (setting == "Tabular") {
    set.seed(rep_id)  # same data as run_posterior_imputation.R
    dat <- generate_data(dgp, N, TT, eps, misclassification, state_strength)
    set.seed(2000000L + rep_id)
    est <- imp_standard_ope_tabular(dat, as.vector(dat$A), dgp, gamma)
  } else if (setting == "Continuous") {
    set.seed(rep_id)
    dat <- generate_data_continuous(dgp, N, TT, eps, misclassification, state_strength)
    set.seed(2000000L + rep_id)
    est <- imp_standard_ope_continuous(dat, as.vector(dat$A), dgp, gamma)
  } else {
    dat <- load_gym_dataset(offline_gym_dataset_path(dgp, tau = eps, rep = rep_id,
                                                     data_dir = offline_data_dir))
    dgp_rep <- dgp
    dgp_rep$init_states <- dat$init_states
    set.seed(2000000L + rep_id)
    est <- imp_standard_ope_gym(dat, as.vector(dat$A), dgp_rep, gamma)
  }
  data.frame(rep = rep_id, epsilon = eps, method = paste0("ORACLE-", names(est)),
             estimate = unname(est), stringsAsFactors = FALSE)
}

tasks <- expand.grid(rep = seq_len(n_rep), eps = grid)
started <- Sys.time()
rows <- parallel::mclapply(seq_len(nrow(tasks)), function(i) run_one(tasks$eps[i], tasks$rep[i]),
                           mc.cores = workers, mc.preschedule = TRUE)
failed <- vapply(rows, inherits, logical(1), what = "try-error")
if (any(failed)) stop(sum(failed), " oracle tasks failed: ", as.character(rows[[which(failed)[1]]]))
estimates <- do.call(rbind, rows)
saveRDS(list(setting = setting, misclassification = misclassification, estimates = estimates,
             config = list(N = N, TT = TT, gamma = gamma, n_rep = n_rep, grid = grid,
                           seeds = "set.seed(rep) for data; set.seed(2e6 + rep) for estimators"),
             started = started, finished = Sys.time()),
        result_file)
cat(setting, "oracle done in", round(as.numeric(difftime(Sys.time(), started, units = "mins")), 1),
    "minutes\n")
