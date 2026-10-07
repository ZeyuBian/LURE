#!/usr/bin/env Rscript
# Report all raw outcomes from run_state_dependent_check.R, without trimming.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1L)
out_dir <- normalizePath(args[1])
runs <- lapply(c("tabular", "continuous"), function(s)
  readRDS(file.path(out_dir, paste0(s, ".rds"))))
stopifnot(all(vapply(runs, function(x) isTRUE(x$complete), logical(1))))
reference_file <- file.path(out_dir, "continuous_reference_100000.rds")
if (file.exists(reference_file)) {
  reference <- readRDS(reference_file)
  stopifnot(isTRUE(reference$original_returns_verified), reference$n_mc == 100000L)
  runs[[2]]$truth <- reference
}
methods <- c("FQE", "SIS", "MIS", "DRL", "LSTD", "MR")
raw <- intervals <- diagnostics <- list()
for (run in runs) {
  stopifnot(length(run$records) == 80L)
  for (r in run$records) {
    est <- r$estimates
    id <- data.frame(setting = run$config$setting, epsilon = r$epsilon, rep = r$rep)
    raw[[length(raw) + 1L]] <- cbind(id, data.frame(
      method = ifelse(methods == "MR", "LURE", methods),
      estimate = as.numeric(est[methods]), truth = run$truth$value))
    valid_ci <- all(is.finite(est[c("MR_ci_lo", "MR_ci_hi")])) &&
      est["MR_ci_lo"] <= est["MR_ci_hi"]
    truth_lower <- run$truth$value - 1.96 * run$truth$mc_se
    truth_upper <- run$truth$value + 1.96 * run$truth$mc_se
    intervals[[length(intervals) + 1L]] <- cbind(id, data.frame(
      estimate = unname(est["MR"]), lo = unname(est["MR_ci_lo"]),
      hi = unname(est["MR_ci_hi"]), valid = valid_ci,
      se = unname((est["MR_ci_hi"] - est["MR_ci_lo"]) / (2 * 1.96)),
      covers = valid_ci && est["MR_ci_lo"] <= run$truth$value && est["MR_ci_hi"] >= run$truth$value,
      covers_full_truth_band = valid_ci && est["MR_ci_lo"] <= truth_lower && est["MR_ci_hi"] >= truth_upper,
      intersects_truth_band = valid_ci && est["MR_ci_lo"] <= truth_upper && est["MR_ci_hi"] >= truth_lower))
    meta <- r$misclassification
    diagnostics[[length(diagnostics) + 1L]] <- cbind(id, data.frame(
      expected_rate = if (is.null(meta)) NA_real_ else meta$expected_rate,
      realized_rate = if (is.null(meta)) NA_real_ else meta$realized_rate,
      min_probability = if (is.null(meta)) NA_real_ else meta$min_probability,
      max_probability = if (is.null(meta)) NA_real_ else meta$max_probability,
      warning_count = length(r$warnings), error = r$error,
      elapsed_seconds = r$elapsed_seconds))
  }
}
raw <- do.call(rbind, raw)
intervals <- do.call(rbind, intervals)
diagnostics <- do.call(rbind, diagnostics)
stopifnot(nrow(raw) == 960L, nrow(intervals) == 160L,
          !anyDuplicated(raw[c("setting", "epsilon", "rep", "method")]),
          all(table(raw$setting, raw$epsilon, raw$method) == 20L))

summary_rows <- lapply(split(raw, interaction(raw$setting, raw$epsilon, raw$method, drop = TRUE)), function(d) {
  ok <- is.finite(d$estimate)
  # No attractive-looking primary summaries based on silently dropping failed fits.
  data.frame(setting = d$setting[1], epsilon = d$epsilon[1], method = d$method[1],
    n = nrow(d), n_finite = sum(ok), mean = if (all(ok)) mean(d$estimate) else NA_real_,
    bias = if (all(ok)) mean(d$estimate - d$truth) else NA_real_,
    rmse = if (all(ok)) sqrt(mean((d$estimate - d$truth)^2)) else NA_real_,
    empirical_sd = if (all(ok)) sd(d$estimate) else NA_real_)
})
summary <- do.call(rbind, summary_rows)
summary <- summary[order(summary$setting, summary$epsilon,
                         match(summary$method, c("LURE", "FQE", "SIS", "MIS", "DRL", "LSTD"))), ]
coverage <- do.call(rbind, lapply(split(intervals, interaction(intervals$setting, intervals$epsilon, drop = TRUE)), function(d) {
  n_covered <- sum(d$covers)
  binomial_ci <- binom.test(n_covered, nrow(d))$conf.int
  data.frame(setting = d$setting[1], epsilon = d$epsilon[1], n = nrow(d),
    n_valid = sum(d$valid), n_covered = n_covered, coverage = mean(d$covers),
    coverage_lo = binomial_ci[1], coverage_hi = binomial_ci[2],
    mean_if_se = if (all(d$valid)) mean(d$se) else NA_real_,
    mean_width = if (all(d$valid)) mean(d$hi - d$lo) else NA_real_,
    cover_full_truth_band = sum(d$covers_full_truth_band),
    intersect_truth_band = sum(d$intersects_truth_band))
}))
coverage <- coverage[order(coverage$setting, coverage$epsilon), ]
comparison <- do.call(rbind, lapply(seq_len(nrow(coverage)), function(i) {
  cov <- coverage[i, ]
  s <- subset(summary, setting == cov$setting & epsilon == cov$epsilon)
  lure <- subset(s, method == "LURE")
  baseline <- subset(s, method != "LURE" & is.finite(rmse))
  best <- baseline[which.min(baseline$rmse), ]
  data.frame(setting = cov$setting, epsilon = cov$epsilon, lure_rmse = lure$rmse,
    best_baseline = best$method, best_baseline_rmse = best$rmse,
    lure_lower_rmse = lure$rmse < best$rmse, n_covered = cov$n_covered,
    n_valid_ci = cov$n_valid, coverage = cov$coverage)
}))
error_summary <- do.call(rbind, lapply(split(diagnostics, interaction(diagnostics$setting, diagnostics$epsilon, drop = TRUE)), function(d)
  data.frame(setting = d$setting[1], epsilon = d$epsilon[1],
    expected_rate = mean(d$expected_rate), realized_rate = mean(d$realized_rate),
    min_probability = min(d$min_probability), max_probability = max(d$max_probability),
    warnings = sum(d$warning_count), wrapper_errors = sum(!is.na(d$error)))))
saveRDS(list(summary = summary, coverage = coverage, comparison = comparison,
             misclassification = error_summary, estimates = raw, intervals = intervals,
             diagnostics = diagnostics), file.path(out_dir, "performance_summary.rds"))

fmt <- function(x, digits = 4L) ifelse(is.finite(x), formatC(x, digits = digits, format = "f"), "NA")
lines <- c("# State-dependent error: 20-replication performance check", "",
  "Only Tabular and Continuous were run. Gym and MIMIC were neither run nor modified in this check.", "",
  "## Fixed design", "",
  "20 replications **per error level per environment** (160 fits of each method in total); seeds 1–20, N=50 trajectories, T=50, gamma=0.7. The same seeds are reused across error levels. Current source estimators, nuisance fits, and influence-function (IF) CIs are unchanged.", "",
  "Flip probability is epsilon + 0.75 * min(epsilon, 0.5-epsilon) * score(state). Tabular scores are -1, 0, 1; Continuous uses tanh((s1+s2)/2), with center (0,0) and scale (1,1). The four epsilon values are reference levels, not necessarily marginal error rates. DGP parameters match the current simulation drivers.", "",
  "All raw outcomes are retained; no outlier removal, seed/setting search, CI inflation, or post-hoc estimator tuning. RMSE below is sqrt(mean(squared error)). The existing Continuous driver labels MSE as RMSE; this standalone report computes the square root correctly without editing that driver.", "",
  "## Reference values", "",
  "| Environment | Value | MC standard error | Reference |",
  "|---|---:|---:|---|")
for (run in runs) lines <- c(lines, sprintf("| %s | %s | %s | %s |", run$config$setting,
  fmt(run$truth$value, 6), fmt(run$truth$mc_se, 6), run$truth$type))
lines <- c(lines, "", sprintf("Continuous truth uses %s target-policy trajectories of length 200, seed 2324, through the existing truth function. Individual returns are saved to quantify reference Monte Carlo uncertainty. Batching was verified to preserve the original function's RNG stream and mean.",
  format(length(runs[[2]]$truth$returns), big.mark = ",")))
if (file.exists(reference_file)) lines <- c(lines, "", sprintf(
  "The initial driver-sized reference used 10,000 rollouts (value %.6f, MC SE %.6f). Only the reference was extended to 100,000 because that uncertainty was material relative to estimator error; the 20 datasets, point estimates, and IF CIs are untouched. The first 10,000 returns agree exactly with the initial calculation. All tables use the refined reference; the original remains in continuous.rds.",
  reference$original_value, reference$original_mc_se))
lines <- c(lines, "",
  "## LURE versus the best observed baseline", "",
  "The best baseline is the method with the lowest RMSE in these 20 replications, not a statistically established winner.", "",
  "| Environment | Epsilon | LURE RMSE | Best baseline | Baseline RMSE | LURE IF coverage |",
  "|---|---:|---:|---|---:|---:|")
for (i in seq_len(nrow(comparison))) {
  r <- comparison[i, ]
  lines <- c(lines, sprintf("| %s | %.2f | %s | %s | %s | %d/20 (%.0f%%) |",
    r$setting, r$epsilon, fmt(r$lure_rmse), r$best_baseline, fmt(r$best_baseline_rmse),
    r$n_covered, 100 * r$coverage))
}
lines <- c(lines, "", sprintf("LURE has lower observed RMSE than every baseline in %d of %d configurations.",
  sum(comparison$lure_lower_rmse), nrow(comparison)))
extreme_file <- file.path(out_dir, "tabular_extreme_diagnostic.rds")
if (file.exists(extreme_file)) {
  extreme <- readRDS(extreme_file)
  stopifnot(isTRUE(extreme$reproduced))
  lines <- c(lines, "", "### Important Tabular instability at epsilon=0.30", "",
    sprintf("Replication %d produced LURE %.5f (true value %.5f), IF SE %.5f and CI [%.5f, %.5f]. This run is retained and drives the large RMSE. Thus 20/20 coverage here must not be interpreted as stable or precise inference.",
      extreme$rep, extreme$estimate$V_hat, runs[[1]]$truth$value,
      extreme$estimate$se, extreme$estimate$ci_lo, extreme$estimate$ci_hi), "",
    sprintf("A deterministic rerun reproduced the estimate exactly. The fitted surrogate-mean contrast in state 3 was only %.8f; the estimator divides by this contrast in its bridge terms. This provides a concrete mechanism for amplification of the estimate and IF variance. No estimator change or replacement replication was made. The diagnostic is saved in tabular_extreme_diagnostic.rds.",
      extreme$bridge_gaps$proxy[extreme$bridge_gaps$state == 3]))
}
lines <- c(lines, "",
  sprintf("![RMSE comparison](%s)", file.path(out_dir, "performance_rmse.png")), "", "## All methods", "",
  "| Environment | Epsilon | Method | Finite/attempted | Bias | RMSE | Empirical SD |",
  "|---|---:|---|---:|---:|---:|---:|")
for (i in seq_len(nrow(summary))) {
  r <- summary[i, ]
  lines <- c(lines, sprintf("| %s | %.2f | %s | %d/%d | %s | %s | %s |",
    r$setting, r$epsilon, r$method, r$n_finite, r$n, fmt(r$bias), fmt(r$rmse), fmt(r$empirical_sd)))
}
lines <- c(lines, "", "## Influence-function CI diagnostics", "",
  "These are the estimators' current nominal 95% CIs, estimate +/- 1.96 * IF SE. No bootstrap CIs were substituted. Both current implementations compute SE as sqrt(var(phi)/(N*T)); this check does not revise their variance formulas.", "",
  "| Environment | Epsilon | Valid CIs | Coverage | Exact 95% binomial interval | Mean IF SE | Mean CI width | Truth-band coverage range* |",
  "|---|---:|---:|---:|---|---:|---:|---|")
for (i in seq_len(nrow(coverage))) {
  r <- coverage[i, ]
  lines <- c(lines, sprintf("| %s | %.2f | %d/20 | %d/20 | [%.1f%%, %.1f%%] | %s | %s | %d–%d /20 |",
    r$setting, r$epsilon, r$n_valid, r$n_covered, 100*r$coverage_lo, 100*r$coverage_hi,
    fmt(r$mean_if_se), fmt(r$mean_width), r$cover_full_truth_band, r$intersect_truth_band))
}
lines <- c(lines, "", "*Conservative sensitivity bounds over the reference value +/- 1.96 MC SE: lower counts CIs containing the whole reference band; upper counts CIs intersecting it. These are not new estimator CIs. Tabular reference uncertainty is zero.", "",
  "With only 20 replications, observed coverage moves in 5-percentage-point steps and cannot precisely establish 95% coverage. Binomial intervals describe replication uncertainty; they are not replacements for the estimator's IF CIs. Invalid CIs, if any, count as noncoverage, and primary bias/RMSE/SD are withheld if any estimate is nonfinite.", "",
  sprintf("![Coverage uncertainty](%s)", file.path(out_dir, "performance_coverage.png")), "", "## Error-generation and run diagnostics", "",
  "| Environment | Epsilon | Mean expected rate | Mean realized rate | Probability range | Warnings | Wrapper errors |",
  "|---|---:|---:|---:|---|---:|---:|")
for (i in seq_len(nrow(error_summary))) {
  r <- error_summary[i, ]
  lines <- c(lines, sprintf("| %s | %.2f | %.2f%% | %.2f%% | [%.2f%%, %.2f%%] | %d | %d |",
    r$setting, r$epsilon, 100*r$expected_rate, 100*r$realized_rate,
    100*r$min_probability, 100*r$max_probability, r$warnings, r$wrapper_errors))
}
lines <- c(lines, "", "Warnings and wrapper errors are retained per replication. Existing one-replication functions catch some estimator errors internally and return NA; the finite-count columns expose those failures even if their messages are suppressed upstream.", "",
  "## Reproduce and inspect", "", "From the Code directory, choose a new output directory and run:", "", "```sh",
  "env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 Rscript run_state_dependent_check.R Tabular NEW_OUTPUT_DIR",
  "env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 Rscript run_state_dependent_check.R Continuous NEW_OUTPUT_DIR",
  "Rscript refine_continuous_reference.R NEW_OUTPUT_DIR",
  "Rscript report_state_dependent_check.R NEW_OUTPUT_DIR", "```", "",
  "`tabular.rds` and `continuous.rds` contain all named estimates, CI endpoints, metadata, warning/error records, initial truth, configuration, source hashes, and session information. `continuous_reference_100000.rds` stores the refined reference used in this report. `performance_summary.rds` contains summary and raw long-format tables. `source_snapshot/` preserves the exact source used; source hashes were checked again at completion. Read any RDS using `readRDS(path)`.")
writeLines(lines, file.path(out_dir, "REPORT.md"))

suppressPackageStartupMessages(library(ggplot2))
cols <- c(LURE = "#087F8C", FQE = "#59A14F", SIS = "#4E79A7", MIS = "#B0448C",
          DRL = "#D55E50", LSTD = "#C78220")
summary$method <- factor(summary$method, levels = names(cols))
p1 <- ggplot(summary, aes(epsilon, rmse, color = method, group = method)) +
  geom_line(linewidth = .8) + geom_point(size = 2.5) +
  facet_wrap(~setting, scales = "free_y") + scale_color_manual(values = cols) +
  scale_x_continuous(breaks = c(.05, .1, .2, .3), labels = c("5%", "10%", "20%", "30%")) +
  scale_y_log10() + theme_minimal(base_size = 13) +
  labs(title = "State-dependent errors: estimation accuracy", subtitle = "20 replications per error level; all outcomes retained",
       x = "Reference error level (epsilon)", y = "RMSE (log scale; lower is better)", color = NULL,
       caption = "N=50, T=50, gamma=0.7, state strength=0.75. Panel y-scales differ.") +
  theme(legend.position = "bottom", panel.grid.minor = element_blank())
ggsave(file.path(out_dir, "performance_rmse.png"), p1, width = 10, height = 5.4, dpi = 180)
p2 <- ggplot(coverage, aes(epsilon, coverage)) +
  geom_hline(yintercept = .95, linetype = "dashed", color = "#777777") +
  geom_errorbar(aes(ymin = coverage_lo, ymax = coverage_hi), width = .012, color = "#087F8C") +
  geom_point(size = 3, color = "#087F8C") +
  facet_wrap(~setting) +
  scale_x_continuous(breaks = c(.05, .1, .2, .3), labels = c("5%", "10%", "20%", "30%")) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, .2), labels = function(x) paste0(100*x, "%")) +
  theme_minimal(base_size = 13) +
  labs(title = "LURE: coverage of the current 95% IF confidence intervals",
       subtitle = "Points: observed coverage; bars: exact binomial 95% uncertainty intervals",
       x = "Reference error level (epsilon)", y = "Empirical coverage",
       caption = "Dashed line: nominal 95%. These bars describe simulation uncertainty, not new estimator CIs.") +
  theme(panel.grid.minor = element_blank())
ggsave(file.path(out_dir, "performance_coverage.png"), p2, width = 10, height = 5.4, dpi = 180)
cat("\nLURE comparison:\n"); print(comparison, row.names = FALSE)
cat("\nIF diagnostics:\n"); print(coverage, row.names = FALSE)
cat("\nRun diagnostics:\n"); print(error_summary, row.names = FALSE)
cat("\nReport:", file.path(out_dir, "REPORT.md"), "\n")
