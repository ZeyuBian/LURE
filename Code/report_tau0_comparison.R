#!/usr/bin/env Rscript

# Summarize the two runs produced by run_tau0_comparison.R. No estimates are
# filtered or trimmed.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) {
  stop("Usage: Rscript report_tau0_comparison.R OUTPUT_DIR")
}
out_dir <- normalizePath(args[1])
runs <- lapply(c("tabular", "continuous"), function(setting) {
  readRDS(file.path(out_dir, paste0(setting, ".rds")))
})
if (!all(vapply(runs, function(run) isTRUE(run$complete), logical(1)))) {
  stop("Both Tabular and Continuous runs must be complete before reporting.")
}

methods <- c("FQE", "SIS", "MIS", "DRL", "LSTD", "MR")
display_methods <- c("FQE", "SIS", "MIS", "DRL", "LSTD", "LURE")
raw_rows <- list()
ci_rows <- list()
diagnostic_rows <- list()

for (run in runs) {
  if (length(run$records) != run$config$n_rep) {
    stop("Incomplete record count for ", run$config$setting)
  }
  for (record in run$records) {
    est <- record$estimates
    raw_rows[[length(raw_rows) + 1L]] <- data.frame(
      setting = run$config$setting,
      rep = record$rep,
      method = display_methods,
      estimate = as.numeric(est[methods]),
      truth = run$truth$value,
      stringsAsFactors = FALSE
    )
    valid_ci <- all(is.finite(est[c("MR_ci_lo", "MR_ci_hi")])) &&
      est["MR_ci_lo"] <= est["MR_ci_hi"]
    ci_rows[[length(ci_rows) + 1L]] <- data.frame(
      setting = run$config$setting,
      rep = record$rep,
      estimate = unname(est["MR"]),
      lo = unname(est["MR_ci_lo"]),
      hi = unname(est["MR_ci_hi"]),
      valid = valid_ci,
      covers = valid_ci && est["MR_ci_lo"] <= run$truth$value &&
        est["MR_ci_hi"] >= run$truth$value
    )
    meta <- record$misclassification
    diagnostic_rows[[length(diagnostic_rows) + 1L]] <- data.frame(
      setting = run$config$setting,
      rep = record$rep,
      expected_rate = if (is.null(meta)) NA_real_ else meta$expected_rate,
      realized_rate = if (is.null(meta)) NA_real_ else meta$realized_rate,
      min_probability = if (is.null(meta)) NA_real_ else meta$min_probability,
      max_probability = if (is.null(meta)) NA_real_ else meta$max_probability,
      warning_count = length(record$warnings),
      wrapper_error = !is.na(record$error),
      elapsed_seconds = record$elapsed_seconds
    )
  }
}

raw <- do.call(rbind, raw_rows)
intervals <- do.call(rbind, ci_rows)
diagnostics <- do.call(rbind, diagnostic_rows)
stopifnot(
  nrow(raw) == 2L * 20L * length(methods),
  nrow(intervals) == 40L,
  !anyDuplicated(raw[c("setting", "rep", "method")]),
  all(table(raw$setting, raw$method) == 20L)
)

summary_rows <- lapply(
  split(raw, interaction(raw$setting, raw$method, drop = TRUE)),
  function(d) {
    finite <- is.finite(d$estimate)
    all_finite <- all(finite)
    data.frame(
      setting = d$setting[1],
      method = d$method[1],
      n = nrow(d),
      n_finite = sum(finite),
      mean = if (all_finite) mean(d$estimate) else NA_real_,
      bias = if (all_finite) mean(d$estimate - d$truth) else NA_real_,
      rmse = if (all_finite) sqrt(mean((d$estimate - d$truth)^2)) else NA_real_,
      empirical_sd = if (all_finite) sd(d$estimate) else NA_real_
    )
  }
)
summary <- do.call(rbind, summary_rows)
method_order <- c("LURE", "FQE", "SIS", "MIS", "DRL", "LSTD")
summary <- summary[order(summary$setting, match(summary$method, method_order)), ]

coverage <- do.call(rbind, lapply(split(intervals, intervals$setting), function(d) {
  covered <- sum(d$covers)
  binom_ci <- binom.test(covered, nrow(d))$conf.int
  data.frame(
    setting = d$setting[1],
    n = nrow(d),
    n_valid = sum(d$valid),
    n_covered = covered,
    coverage = mean(d$covers),
    coverage_lo = binom_ci[1],
    coverage_hi = binom_ci[2],
    mean_if_se = if (all(d$valid)) mean((d$hi - d$lo) / (2 * 1.96)) else NA_real_,
    mean_width = if (all(d$valid)) mean(d$hi - d$lo) else NA_real_
  )
}))

comparison <- do.call(rbind, lapply(split(summary, summary$setting), function(s) {
  lure <- s[s$method == "LURE", ]
  baselines <- s[s$method != "LURE" & is.finite(s$rmse), ]
  best <- baselines[which.min(baselines$rmse), ]
  data.frame(
    setting = s$setting[1],
    lure_rmse = lure$rmse,
    best_baseline = best$method,
    best_baseline_rmse = best$rmse,
    lure_lower_rmse = lure$rmse < best$rmse
  )
}))

diagnostic_summary <- do.call(rbind, lapply(split(diagnostics, diagnostics$setting), function(d) {
  data.frame(
    setting = d$setting[1],
    mean_expected_rate = mean(d$expected_rate),
    mean_realized_rate = mean(d$realized_rate),
    min_probability = min(d$min_probability),
    max_probability = max(d$max_probability),
    warnings = sum(d$warning_count),
    wrapper_errors = sum(d$wrapper_error),
    elapsed_seconds = sum(d$elapsed_seconds)
  )
}))

write.csv(raw, file.path(out_dir, "estimates.csv"), row.names = FALSE)
write.csv(summary, file.path(out_dir, "summary.csv"), row.names = FALSE)
write.csv(intervals, file.path(out_dir, "lure_intervals.csv"), row.names = FALSE)
write.csv(coverage, file.path(out_dir, "lure_coverage.csv"), row.names = FALSE)
write.csv(comparison, file.path(out_dir, "comparison.csv"), row.names = FALSE)
write.csv(diagnostics, file.path(out_dir, "diagnostics.csv"), row.names = FALSE)
saveRDS(
  list(
    summary = summary,
    comparison = comparison,
    coverage = coverage,
    estimates = raw,
    intervals = intervals,
    diagnostics = diagnostics
  ),
  file.path(out_dir, "performance_summary.rds")
)

suppressPackageStartupMessages(library(ggplot2))
colors <- c(
  LURE = "#087F8C", FQE = "#59A14F", SIS = "#4E79A7",
  MIS = "#B0448C", DRL = "#D55E50", LSTD = "#C78220"
)
summary$method <- factor(summary$method, levels = names(colors))
rmse_plot <- ggplot(summary, aes(method, rmse, color = method)) +
  geom_point(size = 3.2) +
  facet_wrap(~setting, scales = "free_y") +
  scale_color_manual(values = colors) +
  theme_minimal(base_size = 13) +
  labs(
    title = "No action misclassification: estimation accuracy",
    subtitle = "20 replications per environment; lower RMSE is better",
    x = NULL,
    y = "RMSE",
    color = NULL,
    caption = "N=50 trajectories, T=50, gamma=0.7, tau=0. All outcomes retained."
  ) +
  theme(legend.position = "none", panel.grid.minor = element_blank())
ggsave(file.path(out_dir, "rmse_comparison.png"), rmse_plot, width = 9, height = 4.8, dpi = 180)

raw$method <- factor(raw$method, levels = names(colors))
estimate_plot <- ggplot(raw, aes(method, estimate, color = method)) +
  geom_hline(aes(yintercept = truth), linetype = "dashed", color = "#555555") +
  geom_boxplot(outlier.shape = NA, width = 0.55, linewidth = 0.45) +
  geom_jitter(width = 0.12, height = 0, alpha = 0.7, size = 1.6) +
  facet_wrap(~setting, scales = "free_y") +
  scale_color_manual(values = colors) +
  theme_minimal(base_size = 13) +
  labs(
    title = "Policy-value estimates when recorded actions are exact",
    subtitle = "Dashed line is the reference policy value",
    x = NULL,
    y = "Estimated policy value",
    color = NULL
  ) +
  theme(legend.position = "none", panel.grid.minor = element_blank())
ggsave(file.path(out_dir, "estimate_distributions.png"), estimate_plot, width = 9, height = 4.8, dpi = 180)

fmt <- function(x, digits = 4L) {
  ifelse(is.finite(x), formatC(x, digits = digits, format = "f"), "NA")
}
lines <- c(
  "# LURE versus baselines with no action misclassification",
  "",
  "## Design",
  "",
  "Twenty replications were run for each of the Tabular and Continuous environments with tau=0, N=50 trajectories, T=50 steps, gamma=0.7, and seeds 1-20. At tau=0 the generated surrogate action equals the true action at every transition. The current LURE estimator and all five baseline implementations were run unchanged. No outcomes were filtered or trimmed.",
  "",
  "The baselines are FQE, sequential importance sampling (SIS), marginalized importance sampling (MIS), double reinforcement learning (DRL), and LSTD. LURE appears as MR in the source implementation and is renamed only in this report.",
  "",
  "## Reference values",
  "",
  "| Environment | Value | MC standard error | Reference |",
  "|---|---:|---:|---|"
)
for (run in runs) {
  lines <- c(lines, sprintf(
    "| %s | %s | %s | %s |",
    run$config$setting, fmt(run$truth$value, 6), fmt(run$truth$mc_se, 6),
    run$truth$method
  ))
}
lines <- c(
  lines,
  "",
  "## Headline comparison",
  "",
  "The best baseline is the method with the smallest observed RMSE in these 20 replications; this is a descriptive comparison, not a formal ranking test.",
  "",
  "| Environment | LURE RMSE | Best baseline | Baseline RMSE | LURE 95% IF coverage |",
  "|---|---:|---|---:|---:|"
)
for (i in seq_len(nrow(comparison))) {
  row <- comparison[i, ]
  cov <- coverage[coverage$setting == row$setting, ]
  lines <- c(lines, sprintf(
    "| %s | %s | %s | %s | %d/20 (%.0f%%) |",
    row$setting, fmt(row$lure_rmse), row$best_baseline,
    fmt(row$best_baseline_rmse), cov$n_covered, 100 * cov$coverage
  ))
}
lines <- c(
  lines,
  "",
  "![RMSE comparison](rmse_comparison.png)",
  "",
  "## All methods",
  "",
  "| Environment | Method | Finite/attempted | Mean | Bias | RMSE | Empirical SD |",
  "|---|---|---:|---:|---:|---:|---:|"
)
for (i in seq_len(nrow(summary))) {
  row <- summary[i, ]
  lines <- c(lines, sprintf(
    "| %s | %s | %d/%d | %s | %s | %s | %s |",
    row$setting, as.character(row$method), row$n_finite, row$n,
    fmt(row$mean), fmt(row$bias), fmt(row$rmse), fmt(row$empirical_sd)
  ))
}
lines <- c(
  lines,
  "",
  "![Estimate distributions](estimate_distributions.png)",
  "",
  "## LURE interval diagnostics",
  "",
  "These are the current nominal 95% influence-function intervals. With 20 replications, observed coverage changes in 5-percentage-point increments.",
  "",
  "| Environment | Valid intervals | Coverage | Exact binomial 95% interval | Mean IF SE | Mean width |",
  "|---|---:|---:|---|---:|---:|"
)
for (i in seq_len(nrow(coverage))) {
  row <- coverage[i, ]
  lines <- c(lines, sprintf(
    "| %s | %d/20 | %d/20 (%.0f%%) | [%.1f%%, %.1f%%] | %s | %s |",
    row$setting, row$n_valid, row$n_covered, 100 * row$coverage,
    100 * row$coverage_lo, 100 * row$coverage_hi,
    fmt(row$mean_if_se), fmt(row$mean_width)
  ))
}
lines <- c(
  lines,
  "",
  "## Run validation",
  "",
  "| Environment | Mean expected error | Mean realized error | Error-probability range | Warnings | Wrapper errors | Runtime |",
  "|---|---:|---:|---|---:|---:|---:|"
)
for (i in seq_len(nrow(diagnostic_summary))) {
  row <- diagnostic_summary[i, ]
  lines <- c(lines, sprintf(
    "| %s | %.2f%% | %.2f%% | [%.2f%%, %.2f%%] | %d | %d | %.1f s |",
    row$setting, 100 * row$mean_expected_rate, 100 * row$mean_realized_rate,
    100 * row$min_probability, 100 * row$max_probability,
    row$warnings, row$wrapper_errors, row$elapsed_seconds
  ))
}
lines <- c(
  lines,
  "",
  "Each run was checkpointed after every replication. The RDS files contain the complete named estimates, interval endpoints, seeds, timing, warnings, error metadata, source hashes, and R session information. Exact source snapshots are stored under source_snapshot/.",
  "",
  "## Reproduce",
  "",
  "From the Code directory, choose a new output directory:",
  "",
  "```sh",
  "env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 Rscript run_tau0_comparison.R Tabular NEW_OUTPUT_DIR",
  "env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 Rscript run_tau0_comparison.R Continuous NEW_OUTPUT_DIR",
  "Rscript report_tau0_comparison.R NEW_OUTPUT_DIR",
  "```"
)
writeLines(lines, file.path(out_dir, "REPORT.md"))

cat("\nLURE comparison:\n")
print(comparison, row.names = FALSE)
cat("\nLURE coverage:\n")
print(coverage, row.names = FALSE)
cat("\nRun diagnostics:\n")
print(diagnostic_summary, row.names = FALSE)
cat("\nReport:", file.path(out_dir, "REPORT.md"), "\n")

