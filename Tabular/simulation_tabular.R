################################################################################
## Tabular MDP Simulation — "Learning from the Unseen"
##
## DGP: |S|=3, A in {0,1}, gamma=0.9, with hidden actions
## Competing methods (FQE, MIS, DRL) loaded from Methods.R
## Our method: multiply robust (MR) estimator with EM-style nuisance estimation
################################################################################

source("Methods.R")
library(rpart)
library(dplyr)

summarize_results <- function(results, V_true) {
  results %>%
    group_by(epsilon, method) %>%
    summarize(
      mean = mean(estimate, na.rm = TRUE),
      bias = mean(estimate, na.rm = TRUE) - V_true,
      rmse = sqrt(mean((estimate - V_true)^2, na.rm = TRUE)),
      sd = sd(estimate, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    # Ensure the output matches your original sorting/method order
    mutate(method = factor(method, levels = c("FQE", "SIS", "MIS", "DRL", "MR"))) %>%
    arrange(epsilon, method)
}
run_full_simulation <- function(dgp, N, TT, epsilon_grid, gamma, n_rep) {
	methods <- c("FQE", "SIS", "MIS", "DRL", "MR")
	V_true <- compute_true_value(dgp, gamma)$V_value

	results <- data.frame(
		rep = integer(0),
		epsilon = numeric(0),
		method = character(0),
		estimate = numeric(0),
		stringsAsFactors = FALSE
	)

	for (eps in epsilon_grid) {
		cat("Running epsilon =", eps, "with", n_rep, "replications\n")
		for (rep in seq_len(n_rep)) {
			set.seed(rep)
			if (rep %% 25 == 0 || rep == 1 || rep == n_rep) {
				cat("  rep", rep, "/", n_rep, "\n")
			}

			est <- one_rep(dgp, N = N, TT = TT, epsilon = eps, gamma = gamma)
			results <- rbind(
				results,
				data.frame(
					rep = rep,
					epsilon = eps,
					method = names(est),
					estimate = as.numeric(est),
					stringsAsFactors = FALSE
				)
			)
		}
	}

	summary <- summarize_results(results, V_true)
	summary <- summary[order(summary$epsilon, match(summary$method, methods)), ]

	list(
		results = results,
		summary = summary,
		V_true = V_true,
		N = N,
		TT = TT,
		n_rep = n_rep,
		epsilon_grid = epsilon_grid
	)
}

dgp <- generate_dgp()
gamma <- 0.6
N <- 40
TT <- 40
n_rep <- 100
epsilon_grid <- c(0.05, 0.15, 0.3)

sim_out <- run_full_simulation(dgp = dgp,N = N,TT = TT,epsilon_grid = epsilon_grid,
                               gamma = gamma,n_rep = n_rep)


library(ggplot2)

plot_simulation_results <- function(results, V_true, N, TT, n_rep) {
  library(ggplot2)
  
  # Rename MR to LURE
  results$method[results$method == "MR"] <- "LURE"
  
  # Color palette with LURE first
  method_cols <- c(
    "LURE" = "deepskyblue",
    "FQE"  = "#4E79A7",
    "SIS"  = "#F28E2B",
    "MIS"  = "#59A14F",
    "DRL"  = "#E15759"
  )
  
  # Fix method order (controls x-axis order)
  results$method <- factor(results$method, levels = names(method_cols))
  
  # Facet labels: epsilon -> Scenario
  eps_vals <- sort(unique(results$epsilon))
  scenario_labels <- setNames(
    paste0("Scenario ", seq_along(eps_vals)),
    eps_vals
  )
  
  # Robust Y-axis scaling
  y_min <- min(c(quantile(results$estimate, 0.01, na.rm = TRUE), V_true))
  y_max <- max(c(quantile(results$estimate, 0.99, na.rm = TRUE), V_true))
  padding <- (y_max - y_min) * 0.1
  
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
      linewidth = 0.45
    ) +
    
    facet_wrap(
      ~epsilon,
      labeller = as_labeller(scenario_labels)
    ) +
    
    scale_fill_manual(values = method_cols) +
    
    coord_cartesian(ylim = c(y_min - padding, y_max + padding)) +
    
    labs(
      y = "Estimated Value",
      x = NULL
    ) +
    
    theme_minimal(base_size = 15) +
    theme(
      legend.position = "none",
      strip.background = element_rect(fill = "#F3F4F6", color = NA),
      strip.text = element_text(size = 15, color = "gray15"),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.grid.major.y = element_line(color = "gray85", linewidth = 0.35),
      axis.text.x = element_text(color = "gray15"),
      axis.text.y = element_text(color = "gray20"),
      plot.title = element_text(size = 18, color = "gray10"),
      plot.margin = margin(10, 15, 10, 10)
    )
}

plot_simulation_results(
  results = sim_out$results,
  V_true = sim_out$V_true,
  N = sim_out$N,
  TT = sim_out$TT,
  n_rep = sim_out$n_rep
)

save(sim_out, file = "res.RData")

