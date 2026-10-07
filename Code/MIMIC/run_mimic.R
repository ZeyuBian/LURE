#!/usr/bin/env Rscript

# Production MIMIC-III sepsis analysis (Section 6 / Table 1 of the paper).
#
# All definitions live in MIMIC.R, which has no top-level side effects. This script
# is the only place that runs an analysis or writes results, so MIMIC.R can be
# source()d freely by sensitivity studies and tests.
#
# Usage from Code/MIMIC:
#   Rscript run_mimic.R                       # reward_timing = "next", -> res.RData
#   Rscript run_mimic.R OUT.RData             # choose the output file
#   Rscript run_mimic.R OUT.RData current     # reproduce the published Table 1

args <- commandArgs(trailingOnly = TRUE)

script_path <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])
script_dir <- if (is.na(script_path)) normalizePath(getwd()) else dirname(normalizePath(script_path))

source(file.path(script_dir, "MIMIC.R"))

out_file <- if (length(args) >= 1L && nzchar(args[1])) {
  args[1]
} else {
  file.path(script_dir, "res.RData")
}
# Always resolve relative to this script, never to getwd().
if (!grepl("^(/|~)", out_file)) {
  out_file <- file.path(script_dir, out_file)
}

reward_timing <- if (length(args) >= 2L && nzchar(args[2])) args[2] else "next"
if (!reward_timing %in% c("next", "current")) {
  stop("reward_timing must be 'next' or 'current', got: ", reward_timing)
}

config <- list(
  seed = 23L,
  gamma = 0.9,
  horizon = 20L,
  max_stays = 500L,
  reward_timing = reward_timing,
  drop_constant_states = TRUE,
  policies = c("high_dose", "low_dose", "sofa_11"),
  policy_labels = c(
    high_dose = "Always high dose",
    low_dose = "Always low dose",
    sofa_11 = "SOFA-tailored rule"
  )
)

cat("MIMIC analysis: reward_timing =", config$reward_timing,
    "| gamma =", config$gamma,
    "| max_stays =", config$max_stays, "\n")
cat("Output:", out_file, "\n\n")

set.seed(config$seed)

results_high <- run_mimic_lure(
  policy_type = "high_dose",
  gamma = config$gamma,
  horizon = config$horizon,
  max_stays = config$max_stays,
  reward_timing = config$reward_timing,
  drop_constant_states = config$drop_constant_states
)
results_low <- run_mimic_lure(
  policy_type = "low_dose",
  gamma = config$gamma,
  horizon = config$horizon,
  max_stays = config$max_stays,
  reward_timing = config$reward_timing,
  drop_constant_states = config$drop_constant_states
)
results_sofa <- run_mimic_lure(
  policy_type = "sofa_11",
  gamma = config$gamma,
  horizon = config$horizon,
  max_stays = config$max_stays,
  reward_timing = config$reward_timing,
  drop_constant_states = config$drop_constant_states
)

results_by_policy <- list(
  high_dose = results_high,
  low_dose = results_low,
  sofa_11 = results_sofa
)

policy_value_table <- mimic_policy_value_table(
  results_by_policy[config$policies],
  policy_labels = unname(config$policy_labels[config$policies]),
  digits = 2
)

cat("\nN =", results_high$n_stays,
    "| T =", results_high$n_transitions,
    "| d =", length(results_high$state_cols), "\n")
if (length(results_high$dropped_states) > 0L) {
  cat("Dropped constant state variables:",
      paste(results_high$dropped_states, collapse = ", "), "\n")
}
cat("Proxy (bridge) variable:", results_high$bridge_state, "\n\n")
print(policy_value_table, row.names = FALSE)

cat("\nDirect / correction decomposition (LURE):\n")
for (nm in config$policies) {
  r <- results_by_policy[[nm]]
  cat(sprintf(
    "  %-18s direct = %8.3f   correction = %8.3f   V = %8.3f\n",
    unname(config$policy_labels[nm]),
    r$mr_out$direct, r$mr_out$correction, r$estimate
  ))
}

cat("\n")
policy_value_table_latex <- mimic_policy_value_latex(policy_value_table)
cat(policy_value_table_latex, sep = "\n")
cat("\n")

save.image(out_file)
cat("\nSaved workspace to", out_file, "\n")
