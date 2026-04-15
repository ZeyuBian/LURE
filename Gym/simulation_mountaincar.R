################################################################################
## MountainCar Simulation for LURE
################################################################################

get_gym_script_dir <- function() {
  frame_files <- vapply(sys.frames(), function(x) {
    if (!is.null(x$ofile)) x$ofile else ""
  }, character(1))
  frame_files <- frame_files[nzchar(frame_files)]
  if (length(frame_files) > 0L) {
    return(dirname(normalizePath(frame_files[length(frame_files)])))
  }
  normalizePath(getwd())
}

gym_dir <- get_gym_script_dir()
options(lure.gym.dir = gym_dir)

source(file.path(gym_dir, "Methods_gym.R"))
source(file.path(gym_dir, "baseline_gym.R"))

library(dplyr)
library(ggplot2)

gamma <- 0.6
N <- 50
TT <- 50
mc_eval_N <- 10000
mc_eval_T <- 2000
n_rep <- 50
tau_grid <- c(0.05, 0.10, 0.20, 0.30)
offline_data_dir <- resolve_offline_gym_data_dir(gym_dir)
target_eval_seed <- 10001L

summarize_env_results <- function(results, V_true) {
  results %>%
    group_by(tau, method) %>%
    summarize(
      bias = mean(estimate, na.rm = TRUE) - V_true,
      rmse = sqrt(mean((estimate - V_true)^2, na.rm = TRUE)),
      sd = sd(estimate, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(method = factor(method, levels = c("DIRECT", "FQE", "MIS", "DRL", "LSTD", "MR"))) %>%
    arrange(tau, method)
}

run_mountaincar_simulation <- function(dgp, N, TT, mc_eval_N, mc_eval_T,
                                       tau_grid, gamma, n_rep,
                                       offline_data_dir) {
  methods <- c("DIRECT", "FQE", "MIS", "DRL", "LSTD", "MR")
  results <- data.frame(
    rep = integer(0),
    tau = numeric(0),
    method = character(0),
    estimate = numeric(0),
    stringsAsFactors = FALSE
  )
  ci_results <- data.frame(
    rep = integer(0),
    tau = numeric(0),
    covers = logical(0),
    stringsAsFactors = FALSE
  )

  target_path <- target_gym_dataset_path(
    dgp,
    N = mc_eval_N,
    TT = mc_eval_T,
    seed = target_eval_seed,
    gamma = gamma,
    data_dir = gym_dir
  )
  if (file.exists(target_path)) {
    target_dat <- load_gym_dataset(target_path)
    truth_target_path <- target_path
  } else {
    cat("Target-policy MC dataset not found; generating summary-only MC truth on the fly.\n")
    target_dat <- generate_target_data_gym(
      dgp,
      N = mc_eval_N,
      TT = mc_eval_T,
      gamma = gamma,
      seed = target_eval_seed
    )
    truth_target_path <- NA_character_
  }
  truth_out <- estimate_true_value_gym(
    dgp,
    N = mc_eval_N,
    TT = mc_eval_T,
    gamma = gamma,
    seed = target_eval_seed,
    eval_dat = target_dat
  )
  dgp$init_states <- truth_out$eval_dat$init_states
  V_true <- truth_out$V_value

  cat(
    "True V(pi) for MountainCar =", round(V_true, 4),
    "(MC SE =", round(truth_out$mc_se, 4), ")\n\n"
  )

  for (tau in tau_grid) {
    cat("Running MountainCar with tau =", tau, "and", n_rep, "replications\n")
    for (rep in seq_len(n_rep)) {
      if (rep %% 10 == 0 || rep == 1 || rep == n_rep) {
        cat("  rep", rep, "/", n_rep, "\n")
      }

      offline_path <- offline_gym_dataset_path(
        dgp,
        tau = tau,
        rep = rep,
        data_dir = offline_data_dir
      )
      if (!file.exists(offline_path)) {
        stop("Missing offline dataset: ", offline_path)
      }

      dat <- load_gym_dataset(offline_path)
      est <- evaluate_gym_estimators(dat, dgp, gamma, seed = rep)
      est_methods <- est[methods]

      results <- rbind(
        results,
        data.frame(
          rep = rep,
          tau = tau,
          method = names(est_methods),
          estimate = as.numeric(est_methods),
          stringsAsFactors = FALSE
        )
      )

      ci_results <- rbind(
        ci_results,
        data.frame(
          rep = rep,
          tau = tau,
          covers = est["MR_ci_lo"] <= V_true && V_true <= est["MR_ci_hi"],
          stringsAsFactors = FALSE
        )
      )
    }
  }

  summary <- summarize_env_results(results, V_true)
  coverage <- ci_results %>%
    group_by(tau) %>%
    summarize(coverage = mean(covers, na.rm = TRUE), .groups = "drop")

  list(
    env = "MountainCar",
    results = results,
    summary = summary,
    coverage = coverage,
    ci_results = ci_results,
    truth = data.frame(
      env = "MountainCar",
      V_true = V_true,
      mc_se = truth_out$mc_se,
      mc_ci_lo = truth_out$ci_lo,
      mc_ci_hi = truth_out$ci_hi,
      target_path = truth_target_path,
      stringsAsFactors = FALSE
    ),
    N = N,
    TT = TT,
    mc_eval_N = mc_eval_N,
    mc_eval_T = mc_eval_T,
    n_rep = n_rep,
    offline_data_dir = offline_data_dir,
    tau_grid = tau_grid,
    gamma = gamma
  )
}

plot_mountaincar_results <- function(results, V_true) {
  plot_dat <- results
  plot_dat$method[plot_dat$method == "DIRECT"] <- "Direct"
  plot_dat$method[plot_dat$method == "MR"] <- "LURE"
  plot_dat$method <- factor(plot_dat$method,
                            levels = c("Direct", "LURE", "FQE", "MIS", "DRL", "LSTD"))
  plot_dat$tau_lab <- paste0("tau = ", plot_dat$tau)

  method_cols <- c(
    "Direct" = "#2F2F2F",
    "LURE" = "deepskyblue",
    "FQE" = "#4E79A7",
    "MIS" = "#59A14F",
    "DRL" = "#E15759",
    "LSTD" = "#F28E2B"
  )

  ggplot(plot_dat, aes(x = method, y = estimate, fill = method)) +
    geom_hline(yintercept = V_true, linetype = "dashed",
               color = "gray35", linewidth = 0.7) +
    geom_boxplot(outlier.shape = NA, alpha = 0.85, width = 0.62,
                 color = "gray20", linewidth = 0.45) +
    facet_wrap(~tau_lab, scales = "free_y") +
    scale_fill_manual(values = method_cols) +
    labs(y = "Estimated Value", x = NULL) +
    theme_minimal(base_size = 15) +
    theme(
      legend.position = "none",
      strip.background = element_rect(fill = "#F3F4F6", color = NA),
      strip.text = element_text(size = 14, color = "gray15"),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.grid.major.y = element_line(color = "gray85", linewidth = 0.35),
      axis.text.x = element_text(color = "gray15"),
      axis.text.y = element_text(color = "gray20"),
      plot.margin = margin(10, 15, 10, 10)
    )
}

dgp <- generate_gym_dgp("MountainCar-v0", bridge_index = 2L)

sim_out <- run_mountaincar_simulation(
  dgp = dgp,
  N = N,
  TT = TT,
  mc_eval_N = mc_eval_N,
  mc_eval_T = mc_eval_T,
  tau_grid = tau_grid,
  gamma = gamma,
  n_rep = n_rep,
  offline_data_dir = offline_data_dir
)

print(sim_out$summary)
print(sim_out$coverage)

plot_mountaincar_results(sim_out$results, sim_out$truth$V_true[1])

save(sim_out, file = file.path(gym_dir, "res_mountaincar.RData"))