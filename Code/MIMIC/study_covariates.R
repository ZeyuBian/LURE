#!/usr/bin/env Rscript

# Covariate reduction: evidence for trimming the 45-variable state.
#
#   1. Per variable: share of transitions where the value is unchanged from t to t+1
#      (static, or a lab carried forward between draws) and its variance inflation
#      factor (VIF) within the full linear design.
#   2. Strongly correlated pairs (|r| >= 0.7).
#   3. For the full, reduced and compact candidate sets: size, max VIF, R^2 for
#      next-step SOFA, and -- the confounding check -- the state-adjusted coefficient
#      on the action. If a smaller set lets that coefficient drift toward the raw
#      difference (+0.60 at cut-off 1), it has lost confounding control.
#
# The reduced and compact sets replace cumulative urine output (output_total) with
# the per-bloc urine output (output_4hourly), read from the CSV.
#
# Usage from Code/MIMIC:
#   Rscript study_covariates.R            (~1 min, mostly reading the CSV)

script_path <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])
script_dir <- if (is.na(script_path)) normalizePath(getwd()) else dirname(normalizePath(script_path))
source(file.path(script_dir, "MIMIC.R"))

baseline <- new.env()
load(file.path(script_dir, "res.RData"), envir = baseline)
dat <- baseline$results_high$dat
TT <- dim(dat$S)[2]
S <- gym_flatten_states(dat$S); Sp <- gym_flatten_states(dat$Sp)
nm <- dat$state_names; colnames(S) <- nm
R <- as.vector(dat$R)

raw <- mimic_read_csv(file.path(script_dir, "sepsis_processed_state_action.csv"))
raw <- raw[raw$icustayid %in% as.numeric(dat$stay_ids), c("icustayid", "bloc", "iv_input", "output_4hourly")]
raw <- raw[order(raw$icustayid, raw$bloc), ]
get_mat <- function(col) do.call(rbind, lapply(dat$stay_ids, function(id) {
  raw[[col]][raw$icustayid == as.numeric(id)][seq_len(TT)]
}))
iv <- as.vector(get_mat("iv_input"))
out4 <- as.vector(get_mat("output_4hourly"))
stopifnot(all(as.integer(iv >= 1) == as.vector(dat$Atilde)), !anyNA(out4))
X_all <- cbind(S, output_4hourly = out4)

REMOVED <- list(
  "near-duplicate (|r| >= 0.85) or derived" = c("PT", "CO2_mEqL", "SGOT", "paO2", "DiaBP", "SysBP"),
  "composite of other state variables"      = c("SIRS"),
  "rarely re-measured, low priority for fluid response" =
    c("SGPT", "Albumin", "Magnesium", "Calcium", "Ionised_Ca", "PTT", "Chloride"),
  "acid-base redundant with pH/HCO3"        = c("Arterial_BE", "paCO2"),
  "ventilator setting (captured by P/F ratio)" = c("FiO2_1"),
  "cumulative, non-stationary (replaced by output_4hourly)" = c("output_total")
)
SETS <- list(
  full    = nm,
  reduced = c(setdiff(nm, unlist(REMOVED)), "output_4hourly"),
  compact = c("SOFA", "MeanBP", "HR", "Shock_Index", "RR", "SpO2", "Temp_C", "PaO2_FiO2",
              "mechvent", "Arterial_lactate", "output_4hourly", "age", "gender", "elixhauser")
)

# The canonical definitions live in MIMIC.R; this script's lists must match them.
stopifnot(setequal(SETS$full, mimic_covariate_set("full")),
          setequal(SETS$reduced, mimic_covariate_set("reduced")),
          setequal(SETS$compact, mimic_covariate_set("compact")))

unch <- colMeans(abs(Sp - S) < 1e-8)
vif_of <- function(X) vapply(seq_len(ncol(X)), function(j) {
  1 / (1 - summary(lm(X[, j] ~ X[, -j]))$r.squared)
}, numeric(1))
vif_full <- setNames(vif_of(S), nm)

cat("=== 1. Full state: unchanged t -> t+1, and VIF ===\n\n")
cat(sprintf("%-18s %10s %8s  %s\n", "variable", "unchanged", "VIF", "in reduced / compact"))
for (j in order(-unch)) {
  cat(sprintf("%-18s %9.1f%% %8.1f  %s / %s\n", nm[j], 100 * unch[j], vif_full[j],
              if (nm[j] %in% SETS$reduced) "keep" else "drop",
              if (nm[j] %in% SETS$compact) "keep" else "drop"))
}

cat("\n=== 2. Pairs with |r| >= 0.7 (full state) ===\n\n")
C <- cor(S); diag(C) <- 0
idx <- which(abs(C) >= 0.7 & upper.tri(C), arr.ind = TRUE)
for (k in order(-abs(C[idx]))) cat(sprintf("  %-16s %-16s %+.3f\n", nm[idx[k, 1]], nm[idx[k, 2]], C[idx][k]))

cat("\n=== 3. Candidate sets ===\n\n")
cat(sprintf("%-8s %4s %8s %12s %22s %22s\n", "set", "d", "max VIF", "R2 SOFA_t+1",
            "adj. A coef, k>=1 (t)", "adj. A coef, k>=3 (t)"))
for (s in names(SETS)) {
  X <- X_all[, SETS[[s]], drop = FALSE]
  r2 <- summary(lm(R ~ X))$r.squared
  co <- vapply(c(1, 3), function(k) {
    A <- as.integer(iv >= k)
    summary(lm(R ~ X + A))$coefficients["A", c(1, 3)]
  }, numeric(2))
  cat(sprintf("%-8s %4d %8.1f %12.4f %13s (%5.2f) %13s (%5.2f)\n", s, ncol(X), max(vif_of(X)), r2,
              sprintf("%+.3f", co[1, 1]), co[2, 1], sprintf("%+.3f", co[1, 2]), co[2, 2]))
}
cat(sprintf("\nraw (unadjusted) differences: k>=1 %+.3f, k>=3 %+.3f\n",
            mean(R[iv >= 1]) - mean(R[iv < 1]), mean(R[iv >= 3]) - mean(R[iv < 3])))

cat("\nRemoved in the reduced set, by reason:\n")
for (r in names(REMOVED)) cat(sprintf("  - %s: %s\n", r, paste(REMOVED[[r]], collapse = ", ")))
