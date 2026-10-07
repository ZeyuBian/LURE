#!/usr/bin/env Rscript

# Summaries and figures for run_posterior_imputation.R.
#
# Usage from Code/:
#   Rscript report_posterior_imputation.R OUTPUT_DIR [BASE_DIR]
# Reads every <setting>.rds in OUTPUT_DIR and writes OUTPUT_DIR/report/. If
# BASE_DIR is given (a full run on the same seeds), LURE, the surrogate-as-truth
# baselines, the oracle, and LURE intervals are taken from it; IMP comes from
# OUTPUT_DIR (e.g. an LURE_IMP_ONLY=1 run).

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
})

args <- commandArgs(trailingOnly = TRUE)
if (!(length(args) %in% 1:2)) stop("Usage: Rscript report_posterior_imputation.R OUTPUT_DIR [BASE_DIR]")
out_dir <- normalizePath(args[1])
base_dir <- if (length(args) == 2L) normalizePath(args[2]) else NULL
report_dir <- file.path(out_dir, "report")
dir.create(report_dir, showWarnings = FALSE)

result_files <- list.files(out_dir, pattern = "^(tabular|continuous|cartpole)(_state_dependent)?[.]rds$",
                           full.names = TRUE)
if (!length(result_files)) stop("No result files found in ", out_dir)

naive <- c("FQE", "SIS", "MIS", "DRL", "LSTD")
imp <- paste0("IMP-", naive)
method_levels <- c("LURE", imp, naive)
oracle <- paste0("ORACLE-", naive)  # true-action benchmark: summary tables only
base_cols <- c(FQE = "#59A14F", SIS = "#4E79A7", MIS = "deeppink", DRL = "#E15759",
               LSTD = "#F28E2B", LURE = "deepskyblue")
paper_cols <- c(LURE = "deepskyblue", FQE = "#59A14F", SIS = "#4E79A7", MIS = "deeppink",
                DRL = "#E15759", LSTD = "#F28E2B", `IMP-FQE` = "#B07AA1", `IMP-DRL` = "#6B4C9A")

theme_paper <- function() {
  theme_minimal(base_size = 15) +
    theme(
      strip.background = element_rect(fill = "#F3F4F6", color = NA),
      strip.text = element_text(size = 15, color = "gray15"),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.grid.major.y = element_line(color = "gray85", linewidth = 0.35),
      axis.text.x = element_text(color = "gray15", angle = 35, hjust = 1),
      axis.text.y = element_text(color = "gray20"),
      plot.margin = margin(10, 15, 10, 10)
    )
}

## SIS (surrogate or imputed) can be far more variable than every other
## estimator; its boxes are clipped rather than allowed to set the axis.
robust_limits <- function(x, method, truth) {
  x <- x[!grepl("SIS", method)]
  lo <- min(quantile(x, 0.01, na.rm = TRUE), truth)
  hi <- max(quantile(x, 0.99, na.rm = TRUE), truth)
  pad <- (hi - lo) * 0.08
  c(lo - pad, hi + pad)
}

fmt <- function(x, d = 3) formatC(x, format = "f", digits = d)
md_lines <- c("# Posterior-imputation (IMP) baseline", "",
              "IMP imputes the latent action from the posterior eta(a | O) built from",
              "mu (continuous-outcome eigendecomposition with outcome R and binary next-state proxy,",
              "Zhou and Tchetgen Tchetgen 2024, Supplement S2), the closed-form behavior",
              "policy from the proof of Theorem 1, and EM estimates of h and q; it then applies",
              "FQE, SIS, MIS, DRL, and LSTD to the completed data and averages over M imputations.",
              "Naive methods use the surrogate as the true action. Monte Carlo SE of the bias is SD/sqrt(n).", "")

for (f in result_files) {
  res <- readRDS(f)
  tag <- sub("[.]rds$", "", basename(f))
  truth <- res$truth$value
  est <- res$estimates
  lure_ci <- res$lure_ci
  oracle_dir <- out_dir
  if (!is.null(base_dir) && file.exists(file.path(base_dir, basename(f)))) {
    base <- readRDS(file.path(base_dir, basename(f)))
    if (!isTRUE(all.equal(base$truth$value, truth))) stop("BASE_DIR truth differs for ", tag)
    est <- rbind(est[grepl("^IMP-", est$method), ],
                 base$estimates[!grepl("^IMP-", base$estimates$method), ])
    lure_ci <- base$lure_ci
    oracle_dir <- base_dir
  }
  est <- est[!is.na(est$estimate) | grepl("^IMP-", est$method), ]
  oracle_file <- file.path(oracle_dir, paste0(tag, "_oracle.rds"))
  if (file.exists(oracle_file)) est <- rbind(est, readRDS(oracle_file)$estimates)
  est$method[est$method == "MR"] <- "LURE"
  est$method <- factor(est$method, levels = c(method_levels, oracle))
  scen <- sort(unique(est$epsilon))
  est$scenario <- factor(paste0("Scenario ", match(est$epsilon, scen)),
                         levels = paste0("Scenario ", seq_along(scen)))

  summ <- est %>%
    group_by(epsilon, method) %>%
    summarize(n = sum(is.finite(estimate)),
              mean = mean(estimate, na.rm = TRUE),
              bias = mean - truth,
              sd = sd(estimate, na.rm = TRUE),
              rmse = sqrt(mean((estimate - truth)^2, na.rm = TRUE)),
              mcse_bias = sd / sqrt(n),
              .groups = "drop") %>%
    arrange(epsilon, method)
  write.csv(summ, file.path(report_dir, paste0(tag, "_summary.csv")), row.names = FALSE)

  cov <- lure_ci %>% group_by(epsilon) %>%
    summarize(coverage = mean(covers, na.rm = TRUE), .groups = "drop")
  diag_cols <- intersect(c("expected_accuracy", "imputation_accuracy", "mu0_mean", "mu1_mean",
                           "thetaR0_mean", "thetaR1_mean", "b1_mean", "eig_complex_frac",
                           "eig_degenerate_frac", "eig_out_of_range_frac", "gap_floored_frac"),
                         names(res$diagnostics))
  diag <- res$diagnostics %>% group_by(epsilon) %>%
    summarize(across(all_of(diag_cols), ~ mean(.x, na.rm = TRUE)), .groups = "drop")
  write.csv(diag, file.path(report_dir, paste0(tag, "_imp_diagnostics.csv")), row.names = FALSE)

  ## Figure 1: every estimator; IMP boxes share the hue of the OPE method they
  ## complete, drawn lighter, so each surrogate/imputed pair reads together.
  est <- est %>% filter(!method %in% oracle) %>% droplevels()
  est$base <- factor(sub("^IMP-", "", as.character(est$method)), levels = names(base_cols))
  est$actions <- factor(ifelse(grepl("^IMP-", est$method), "Imputed A",
                               ifelse(est$method == "LURE", "LURE", "Surrogate as A")),
                        levels = c("LURE", "Imputed A", "Surrogate as A"))
  p_all <- ggplot(est, aes(x = method, y = estimate, fill = base, alpha = actions)) +
    geom_hline(yintercept = truth, linetype = "dashed", color = "gray35", linewidth = 0.7) +
    geom_boxplot(outlier.shape = NA, width = 0.62, color = "gray20", linewidth = 0.45,
                 na.rm = TRUE) +
    facet_wrap(~scenario) +
    scale_fill_manual(values = base_cols, guide = "none") +
    scale_alpha_manual(values = c(LURE = 0.9, `Imputed A` = 0.45, `Surrogate as A` = 0.9),
                       guide = "none") +
    coord_cartesian(ylim = robust_limits(est$estimate, est$method, truth)) +
    labs(y = "Estimated Value", x = NULL) +
    theme_paper()
  ggsave(file.path(report_dir, paste0(tag, "_all_methods.pdf")), p_all, width = 12, height = 8.5)
  ggsave(file.path(report_dir, paste0(tag, "_all_methods.png")), p_all, width = 12, height = 8.5,
         dpi = 150)

  ## Figure 2: the manuscript figure plus the two IMP estimators most comparable
  ## to LURE's direct (FQE) and doubly robust (DRL) components.
  paper <- est %>% filter(method %in% names(paper_cols)) %>%
    mutate(method = factor(as.character(method), levels = names(paper_cols)))
  p_paper <- ggplot(paper, aes(x = method, y = estimate, fill = method)) +
    geom_hline(yintercept = truth, linetype = "dashed", color = "gray35", linewidth = 0.7) +
    geom_boxplot(outlier.shape = NA, alpha = 0.85, width = 0.62, color = "gray20",
                 linewidth = 0.45, na.rm = TRUE) +
    facet_wrap(~scenario) +
    scale_fill_manual(values = paper_cols) +
    coord_cartesian(ylim = robust_limits(paper$estimate, paper$method, truth)) +
    labs(y = "Estimated Value", x = NULL) +
    theme_paper() + theme(legend.position = "none")
  ggsave(file.path(report_dir, paste0(tag, "_paper.pdf")), p_paper, width = 11, height = 8)
  ggsave(file.path(report_dir, paste0(tag, "_paper.png")), p_paper, width = 11, height = 8,
         dpi = 150)

  md_lines <- c(md_lines,
    paste0("## ", tag), "",
    paste0("Truth ", fmt(truth, 4), " (", res$truth$method, "); ", res$config$n_rep,
           " replications; M = ", res$config$M, " imputations; failures: ",
           if (is.null(res$failures)) 0 else nrow(res$failures), "."), "",
    "| Scenario | Method | Bias | MCSE | SD | RMSE |", "|---|---|---:|---:|---:|---:|",
    with(summ, paste0("| ", match(epsilon, scen), " (", epsilon, ") | ", method, " | ",
                      fmt(bias), " | ", fmt(mcse_bias), " | ", fmt(sd), " | ", fmt(rmse), " |")),
    "", "LURE 95% CI coverage: ",
    paste0(paste0("Scenario ", match(cov$epsilon, scen), " = ", fmt(cov$coverage, 2)), collapse = "; "),
    "", "IMP diagnostics (means over replications):", "",
    paste0("| epsilon | ", paste(diag_cols, collapse = " | "), " |"),
    paste0("|", paste(rep("---:", length(diag_cols) + 1L), collapse = "|"), "|"),
    apply(diag, 1, function(r) paste0("| ", paste(fmt(as.numeric(r), 3), collapse = " | "), " |")),
    "", paste0("![", tag, "](", tag, "_all_methods.png)"), "")
}
writeLines(md_lines, file.path(report_dir, "REPORT.md"))
cat("Wrote", report_dir, "\n")
