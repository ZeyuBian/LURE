################################################################################
## Continuous-State MDP Simulation — "Learning from the Unseen"
##
## DGP: S = (S1, S2) in R^2, A in {0,1}, hidden actions
## Linear transitions, deterministic reward from next state
## Competing methods (FQE, SIS, MIS, DRL, LSTD) and LURE loaded from
## Methods_continuous.R and baseline.R
################################################################################

source("Methods_continuous.R")
source("baseline.R")

library(dplyr)
library(ggplot2)

gamma <- 0.7
N <- 50
TT <- 50
n_rep <- 100
epsilon_grid <- c(0.05, 0.1, 0.2,0.3)

# ==============================================================================
# Summarise replications
# ==============================================================================
summarize_results <- function(results, V_true) {
  results %>%
    group_by(epsilon, method) %>%
    summarize(
      bias = mean(estimate, na.rm = TRUE) - V_true,
      rmse = mean((estimate - V_true)^2, na.rm = TRUE),
      sd   = sd(estimate, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(method = factor(method,
           levels = c("FQE", "SIS", "MIS", "DRL", "LSTD", "MR"))) %>%
    arrange(epsilon, method)
}

# ==============================================================================
# Full simulation
# ==============================================================================
run_full_simulation <- function(dgp, N, TT, epsilon_grid, gamma, n_rep) {
  methods <- c("FQE", "SIS", "MIS", "DRL", "LSTD", "MR")
  set.seed(2324)
  V_true  <- compute_true_value_continuous(dgp, gamma)$V_value

  cat("True V(pi) =", round(V_true, 4), "\n\n")

  results <- data.frame(rep = integer(0), epsilon = numeric(0),
                        method = character(0), estimate = numeric(0),
                        stringsAsFactors = FALSE)
  ## LURE CI coverage tracking (TRUE/FALSE)
  ci_results <- data.frame(rep = integer(0), epsilon = numeric(0),
                           covers = logical(0),
                           stringsAsFactors = FALSE)
  bridge_results <- data.frame(rep = integer(0), epsilon = numeric(0),
                               bridge_index = integer(0),
                               bridge_score_sp1 = numeric(0),
                               bridge_score_sp2 = numeric(0),
                               stringsAsFactors = FALSE)

  for (eps in epsilon_grid) {
    cat("Running epsilon =", eps, "with", n_rep, "replications\n")
    for (rep in seq_len(n_rep)) {
      set.seed(rep)
      if (rep %% 25 == 0 || rep == 1 || rep == n_rep) {
        cat("  rep", rep, "/", n_rep, "\n")
      }

      est <- one_rep_continuous(dgp, N = N, TT = TT,
                                epsilon = eps, gamma = gamma)

      ## Store point estimates
      est_methods <- est[c("FQE", "SIS", "MIS", "DRL", "LSTD", "MR")]
      results <- rbind(
        results,
        data.frame(rep = rep, epsilon = eps,
                   method = names(est_methods),
                   estimate = as.numeric(est_methods),
                   stringsAsFactors = FALSE)
      )

      ## Store LURE CI coverage
      ci_results <- rbind(
        ci_results,
        data.frame(rep = rep, epsilon = eps,
                   covers = est["MR_ci_lo"] <= V_true & V_true <= est["MR_ci_hi"],
                   stringsAsFactors = FALSE)
      )

      bridge_results <- rbind(
        bridge_results,
        data.frame(
          rep = rep,
          epsilon = eps,
          bridge_index = as.integer(est["bridge_index"]),
          bridge_score_sp1 = as.numeric(est["bridge_score_sp1"]),
          bridge_score_sp2 = as.numeric(est["bridge_score_sp2"]),
          stringsAsFactors = FALSE
        )
      )
    }
  }

  summary <- summarize_results(results, V_true)
  summary <- summary[order(summary$epsilon,
                           match(summary$method, methods)), ]

  ## LURE coverage
  coverage <- ci_results %>%
    group_by(epsilon) %>%
    summarize(coverage = mean(covers, na.rm = TRUE),
              .groups = "drop")
  bridge_summary <- bridge_results %>%
    mutate(bridge_state = ifelse(bridge_index == 1L, "Sp1",
                          ifelse(bridge_index == 2L, "Sp2", NA_character_))) %>%
    group_by(epsilon, bridge_state) %>%
    summarize(
      n_selected = n(),
      mean_score_sp1 = mean(bridge_score_sp1, na.rm = TRUE),
      mean_score_sp2 = mean(bridge_score_sp2, na.rm = TRUE),
      .groups = "drop"
    )
  cat("\n--- LURE 95% CI Coverage ---\n")
  print(as.data.frame(coverage))
  cat("\n--- Selected Bridge Coordinate ---\n")
  print(as.data.frame(bridge_summary))

  list(results  = results,
       summary  = summary,
       coverage = coverage,
       bridge_results = bridge_results,
       bridge_summary = bridge_summary,
       ci_results = ci_results,
       V_true   = V_true,
       N = N, TT = TT,
       n_rep = n_rep,
       epsilon_grid = epsilon_grid)
}


dgp   <- generate_dgp_continuous(
  s1_a_int = 0.4,
  s2_a_int = -0.3,
  init_mean = c(0.25, 0.05),
  init_sd = c(0.75, 0.75),
  pi_func = function(s1, s2) as.numeric(s1 >= 0.25 & s2 >= -0.10)
)


sim_out <- run_full_simulation(dgp = dgp, N = N, TT = TT,
                               epsilon_grid = epsilon_grid,
                               gamma = gamma, n_rep = n_rep)

print(sim_out$summary)

save(sim_out, file = "res_continuous.RData")

library(dplyr)
library(ggplot2)

truncate_remove <- function(x, q) {
  lower <- quantile(x, 1 - q, na.rm = TRUE)
  upper <- quantile(x, q, na.rm = TRUE)
  
  x[x < lower | x > upper] <- NA
  x
}

sim_out$results <- sim_out$results %>%
  dplyr::group_by(method, epsilon) %>%
  dplyr::mutate(
    estimate = ifelse(
      method == "SIS",
      truncate_remove(estimate, q = 0.85),
      estimate
    )
  ) %>%
  dplyr::ungroup()

plot_simulation_results <- function(results, V_true, N, TT, n_rep) {
  
  # Rename MR to LURE
  results$method[results$method == "MR"] <- "LURE"
  
  method_cols <- c(
    "LURE"   = "deepskyblue",
    "FQE"    = "#59A14F",
    "SIS"    = "#4E79A7",
    "MIS"    = "deeppink",
    "DRL"    = "#E15759",
    "LSTD"   = "#F28E2B"
  )
  
  results$method <- factor(results$method, levels = names(method_cols))
  
  eps_vals <- sort(unique(results$epsilon))
  scenario_labels <- setNames(
    paste0("Scenario ", seq_along(eps_vals)),
    eps_vals
  )
  
  ggplot(results, aes(x = method, y = estimate, fill = method)) +
    geom_hline(
      yintercept = V_true,
      linetype = "dashed",
      color = "gray35",
      linewidth = 0.7
    ) +
    geom_boxplot(
      outlier.shape = NA,
      alpha = 0.85,
      width = 0.62,
      color = "gray20",
      linewidth = 0.45,
      na.rm = TRUE
    ) +
    facet_wrap(
      ~epsilon,
      labeller = as_labeller(scenario_labels),
      scales = "free_y"
    ) +
    scale_fill_manual(values = method_cols) +
    labs(y = "Estimated Value", x = NULL) +
    theme_minimal(base_size = 15) +
    theme(
      legend.position    = "none",
      strip.background   = element_rect(fill = "#F3F4F6", color = NA),
      strip.text         = element_text(size = 15, color = "gray15"),
      panel.grid.minor   = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.grid.major.y = element_line(color = "gray85", linewidth = 0.35),
      axis.text.x        = element_text(color = "gray15"),
      axis.text.y        = element_text(color = "gray20"),
      plot.title         = element_text(size = 18, color = "gray10"),
      plot.margin        = margin(10, 15, 10, 10)
    )
}

plot_simulation_results(
  results = sim_out$results,
  V_true  = sim_out$V_true,
  N = sim_out$N, TT = sim_out$TT,
  n_rep = sim_out$n_rep
)


sim_out$ci_results %>%
  group_by(epsilon) %>%
  summarize(mean_cover = round(mean(covers), 2))




