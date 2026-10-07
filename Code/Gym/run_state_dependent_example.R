## Run from any directory: Rscript Code/Gym/run_state_dependent_example.R OUTPUT [10] [2]
## Fresh data, fixed mechanism, IF CIs; no setting selection or bootstrap.
script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
gym_dir <- if (length(script_arg)) dirname(normalizePath(sub("^--file=", "", script_arg[1]))) else normalizePath(".")
options(lure.gym.dir = gym_dir)
source(file.path(gym_dir, "Methods_gym.R"))
source(file.path(gym_dir, "baseline_gym.R"))
source(file.path(gym_dir, "state_dependent_error.R"))

run_state_dependent_example <- function(output, n_rep = 10L, cores = 2L) {
  stopifnot(n_rep >= 1L, n_rep == as.integer(n_rep), cores >= 1L)
  if (dir.exists(output) && length(list.files(output, all.files = TRUE, no.. = TRUE))) {
    stop("Use a new/empty result directory; existing results will not be overwritten.")
  }
  dir.create(output, recursive = TRUE, showWarnings = FALSE)
  output <- normalizePath(output)
  options(lure.cartpole.target_x = as.numeric(Sys.getenv("LURE_CARTPOLE_TARGET_X", "-2.4")),
          lure.cartpole.target_theta = as.numeric(Sys.getenv("LURE_CARTPOLE_TARGET_THETA", "0.2")))
  stopifnot(is.finite(getOption("lure.cartpole.target_x")), is.finite(getOption("lure.cartpole.target_theta")))
  config <- list(N = 50L, TT = 50L, gamma = .7, n_rep = n_rep, cores = cores,
    rates = c(.05, .10, .20, .30), modes = c("constant", "state_dependent"), strength = .75,
    calibration_N = 1000L, calibration_seed = 7000000L,
    target_N = 10000L, target_T = 400L, target_seed = 8000000L,
    rollout_seeds = 9000000L + 1000L * seq_len(n_rep),
    label_seeds = 10000000L + seq_len(n_rep), fit_seeds = 20000L + seq_len(n_rep),
    target_x = getOption("lure.cartpole.target_x"), target_theta = getOption("lure.cartpole.target_theta"),
    ci = "Unchanged user IF CI: Vhat +/- 1.96*sqrt(var(held-out correction)/(N*T))",
    session_info = capture.output(sessionInfo()))
  saveRDS(config, file.path(output, "config.rds"))
  snapshot <- file.path(output, "source_snapshot")
  dir.create(snapshot)
  sources <- file.path(gym_dir, c("Methods_gym.R", "baseline_gym.R", "gym_data.py",
    "state_dependent_error.R", "run_state_dependent_example.R", "report_state_dependent_example.R"))
  stopifnot(all(file.copy(sources, snapshot)))
  write.csv(data.frame(file = basename(sources), md5 = unname(tools::md5sum(sources))),
            file.path(output, "source_hashes.csv"), row.names = FALSE)
  writeLines(c("# Fixed 10-replication CartPole example", "",
    sprintf("N=50, T=50, gamma=0.7; %d paired replications at rates .05,.10,.20,.30.", n_rep),
    "Constant and state-dependent recording share trajectories and recording uniforms.",
    "p(flip|S) = tau + .75*min(tau,.5-tau)*tanh((x-center)/scale).",
    "Scale is the calibration sample's position IQR; center solves mean(tanh((x-center)/scale))=0.",
    "Calibration uses 1,000 independent behavior trajectories, fixed before the example outcomes.",
    sprintf("The target is x > %g and theta < %g in BOTH R estimation and Python Monte Carlo.", config$target_x, config$target_theta),
    "Fresh target MC: 10,000 trajectories, horizon 400. Its MC SE is reported.",
    "Outer reset-seed ranges are disjoint and separate from calibration and target MC.",
    "Only the action RECORDING mechanism is changed. Dynamics, rewards, nuisance fits, and the user's IF CI are retained.",
    "JSON loading is corrected to preserve trajectory/time/coordinate alignment. Prior saved data and results are not overwritten.",
    "The inherited clean-reset/noisy-initial-state convention remains and is a limitation.",
    "No bootstrap, SE multiplier, trimming of estimates, or coverage-driven selection. All failures and warnings are retained.", ""),
    file.path(output, "PLAN.md"))
  dgp <- generate_gym_dgp("CartPole-v1")
  cat("Generating independent calibration trajectories...\n")
  cal_dat <- generate_offline_data_gym(dgp, config$calibration_N, config$TT, 0,
                                       config$gamma, seed = config$calibration_seed)
  calibration <- cartpole_calibrate_error(gym_flatten_states(cal_dat$S))
  saveRDS(calibration, file.path(output, "calibration.rds"))
  print(calibration)
  cat("Generating fresh target-policy Monte Carlo (10000 x 400)...\n")
  target <- generate_target_data_gym(dgp, config$target_N, config$target_T,
                                     config$gamma, seed = config$target_seed)
  stopifnot(all(is.finite(target$discounted_returns)),
    target$target_policy$x_threshold == config$target_x,
    target$target_policy$theta_threshold == config$target_theta)
  truth <- list(V_true = mean(target$discounted_returns),
    mc_se = sd(target$discounted_returns) / sqrt(config$target_N),
    n = config$target_N, horizon = config$target_T, seed = config$target_seed,
    target_policy = target$target_policy)
  saveRDS(target, file.path(output, "target_mc.rds"))
  saveRDS(truth, file.path(output, "truth.rds"))
  print(truth)
  run_one <- function(rep) {
    dat0 <- generate_offline_data_gym(dgp, config$N, config$TT, 0,
                                     config$gamma, seed = config$rollout_seeds[rep])
    stopifnot(max(abs(dat0$S[, -1, ] - dat0$Sp[, -config$TT, ])) < 1e-12)
    set.seed(config$label_seeds[rep])
    uniforms <- matrix(runif(config$N * config$TT), config$N, config$TT)
    estimates <- intervals <- diagnostics <- warnings <- list()
    j <- 0L
    for (mode in config$modes) for (rate in config$rates) {
      j <- j + 1L
      dat <- cartpole_record_errors(dat0, rate, uniforms, mode, config$strength,
                                    calibration$center, calibration$scale)
      messages <- character()
      started <- proc.time()["elapsed"]
      est <- withCallingHandlers(evaluate_gym_estimators(dat, dgp, config$gamma,
                                                        seed = config$fit_seeds[rep]),
        warning = function(w) { messages <<- c(messages, conditionMessage(w)); invokeRestart("muffleWarning") })
      methods <- c("DIRECT", "FQE", "SIS", "MIS", "DRL", "LSTD", "MR")
      estimates[[j]] <- data.frame(rep = rep, mode = mode, rate = rate, method = methods,
        estimate = unname(est[methods]), V_true = truth$V_true)
      lo <- unname(est["MR_ci_lo"]); hi <- unname(est["MR_ci_hi"])
      valid <- is.finite(lo) && is.finite(hi) && lo <= hi
      intervals[[j]] <- data.frame(rep = rep, mode = mode, rate = rate,
        estimate = unname(est["MR"]), lower = lo, upper = hi, se = (hi-lo)/3.92,
        valid = valid, covered = valid && lo <= truth$V_true && truth$V_true <= hi,
        V_true = truth$V_true)
      diagnostics[[j]] <- data.frame(rep = rep, mode = mode, rate = rate,
        expected_rate = mean(dat$misclassification_prob), actual_rate = mean(dat$A != dat$Atilde),
        min_probability = min(dat$misclassification_prob), max_probability = max(dat$misclassification_prob),
        bridge_index = unname(est["bridge_index"]), warning_count = length(messages),
        nonfinite_estimates = sum(!is.finite(est[methods])), elapsed = unname(proc.time()["elapsed"]-started))
      warnings[[j]] <- data.frame(rep = rep, mode = mode, rate = rate,
                                  messages = paste(unique(messages), collapse = " | "))
      cat(sprintf("rep %02d/%02d %s rate %.2f: LURE %.5f, IF SE %.5f, covered=%s\n",
        rep, n_rep, mode, rate, est["MR"], (hi-lo)/3.92, intervals[[j]]$covered))
    }
    out <- list(estimates = do.call(rbind, estimates), intervals = do.call(rbind, intervals),
      diagnostics = do.call(rbind, diagnostics), warnings = do.call(rbind, warnings),
      oracle = dat0, recording_uniforms = uniforms)
    saveRDS(out, file.path(output, sprintf("rep_%03d.rds", rep)))
    TRUE
  }
  status <- parallel::mclapply(seq_len(n_rep), run_one, mc.cores = min(cores, n_rep),
                               mc.set.seed = FALSE)
  if (!all(vapply(status, isTRUE, logical(1)))) stop("A worker failed; completed datasets remain saved.")
  source(file.path(gym_dir, "report_state_dependent_example.R"))
  report_state_dependent_example(output)
  invisible(output)
}

if (sys.nframe() == 0L) {
  args <- commandArgs(TRUE)
  if (!length(args) %in% 1:3) stop("Pass OUTPUT [n_rep=10] [cores=2].")
  run_state_dependent_example(args[1], if (length(args)>=2) as.integer(args[2]) else 10L,
                              if (length(args)>=3) as.integer(args[3]) else 2L)
}
