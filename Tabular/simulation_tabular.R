################################################################################
## Tabular MDP Simulation — "Learning from the Unseen"
##
## DGP: |S|=3, A in {0,1}, gamma=0.9, with hidden actions
## Competing methods (FQE, SIS, MIS, DRL, LSTD) loaded from Methods.R
## Our method: multiply robust (MR) estimator with EM-style nuisance estimation
################################################################################

source("Methods.R")
library(rpart)
library(dplyr)
gamma <- 0.7
N <- 50
TT <- 50
n_rep <- 100
epsilon_grid <- c(0.05, 0.1, 0.2,0.3)

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
    mutate(method = factor(method, levels = c("FQE", "SIS", "MIS", "DRL", "LSTD", "MR"))) %>%
    arrange(epsilon, method)
}

run_full_simulation <- function(dgp, N, TT, epsilon_grid, gamma, n_rep) {
	  methods <- c("FQE", "SIS", "MIS", "DRL", "LSTD", "MR")
	V_true <- compute_true_value(dgp, gamma)$V_value

	results <- data.frame(
		rep = integer(0),
		epsilon = numeric(0),
		method = character(0),
		estimate = numeric(0),
		stringsAsFactors = FALSE
	)

	## LURE CI coverage tracking (TRUE/FALSE)
	ci_results <- data.frame(rep = integer(0), epsilon = numeric(0),
	                         covers = logical(0),
	                         stringsAsFactors = FALSE)

	for (eps in epsilon_grid) {
		cat("Running epsilon =", eps, "with", n_rep, "replications\n")
		for (rep in seq_len(n_rep)) {
			set.seed(rep)
			if (rep %% 25 == 0 || rep == 1 || rep == n_rep) {
				cat("  rep", rep, "/", n_rep, "\n")
			}

			est <- one_rep(dgp, N = N, TT = TT, epsilon = eps, gamma = gamma)

      ## Store point estimates
      est_methods <- est[c("FQE", "SIS", "MIS", "DRL", "LSTD", "MR")]
			results <- rbind(
				results,
				data.frame(
					rep = rep,
					epsilon = eps,
					method = names(est_methods),
					estimate = as.numeric(est_methods),
					stringsAsFactors = FALSE
				)
			)

			## Store LURE CI coverage
			ci_results <- rbind(
				ci_results,
				data.frame(rep = rep, epsilon = eps,
				           covers = est["MR_ci_lo"] <= V_true & V_true <= est["MR_ci_hi"],
				           stringsAsFactors = FALSE)
			)
		}
	}

	summary <- summarize_results(results, V_true)
	summary <- summary[order(summary$epsilon, match(summary$method, methods)), ]

	## LURE coverage
	coverage <- ci_results %>%
	  group_by(epsilon) %>%
	  summarize(coverage = mean(covers, na.rm = TRUE),
	            .groups = "drop")
	cat("\n--- LURE 95% CI Coverage ---\n")
	print(as.data.frame(coverage))

	list(
		results = results,
		summary = summary,
		coverage = coverage,
		ci_results = ci_results,
		V_true = V_true,
		N = N,
		TT = TT,
		n_rep = n_rep,
		epsilon_grid = epsilon_grid
	)
}

dgp <- generate_dgp()

sim_out <- run_full_simulation(dgp = dgp,N = N,TT = TT,epsilon_grid = epsilon_grid,
                               gamma = gamma,n_rep = n_rep)


library(ggplot2)

plot_simulation_results <- function(results, V_true, N, TT, n_rep) {
  library(ggplot2)
  
  # Rename MR to LURE
  results$method[results$method == "MR"] <- "LURE"
  
  # Color palette with LURE first
  method_cols <- c(
    "LURE"   = "deepskyblue",
    "FQE"    = "#59A14F",
    "SIS"    = "#4E79A7",
    "MIS"    = "deeppink",
    "DRL"    = "#E15759",
    "LSTD"   = "#F28E2B"
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
      y = "Empirical Value",
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

library(dplyr)

sim_out$ci_results %>%
  group_by(epsilon) %>%
  summarize(mean_cover = round(mean(covers), 2))
