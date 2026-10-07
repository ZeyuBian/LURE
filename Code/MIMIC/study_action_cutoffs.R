#!/usr/bin/env Rscript

# What the binary action means at each cut-off, and whether the recorded action has
# any association with next-step SOFA once the state is accounted for.
#
#   1. Volume range of each iv_input dose code (from input_4hourly, mL per 4 h).
#   2. lm(SOFA_{t+1} ~ state + A) for A = 1{iv_input >= k}, k = 1..4.
#   3. Dose-response: each dose level vs none, adjusted for the state.
#
# Usage from Code/MIMIC:
#   Rscript study_action_cutoffs.R          (~1 min, mostly reading the CSV)

script_path <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])
script_dir <- if (is.na(script_path)) normalizePath(getwd()) else dirname(normalizePath(script_path))
source(file.path(script_dir, "MIMIC.R"))

baseline <- new.env()
load(file.path(script_dir, "res.RData"), envir = baseline)
dat <- baseline$results_high$dat
TT <- dim(dat$S)[2]

raw <- mimic_read_csv(file.path(script_dir, "sepsis_processed_state_action.csv"))
raw <- raw[raw$icustayid %in% as.numeric(dat$stay_ids), c("icustayid", "bloc", "iv_input", "input_4hourly")]
raw <- raw[order(raw$icustayid, raw$bloc), ]
get_mat <- function(col) do.call(rbind, lapply(dat$stay_ids, function(id) {
  raw[[col]][raw$icustayid == as.numeric(id)][seq_len(TT)]
}))
iv  <- as.vector(get_mat("iv_input"))
vol <- as.vector(get_mat("input_4hourly"))
stopifnot(all(as.integer(iv >= 1) == as.vector(dat$Atilde)))   # cut-off 1 = production action

cat("=== 1. What the dose codes mean (analysis transitions, n =", length(iv), ") ===\n\n")
cat(sprintf("%8s %7s %8s %12s %14s %12s\n", "iv_input", "share", "cum>=k", "min mL/4h", "median mL/4h", "max mL/4h"))
for (k in 0:4) {
  v <- vol[iv == k]
  cat(sprintf("%8d %6.1f%% %7.1f%% %12.1f %14.1f %12.1f\n", k, 100 * mean(iv == k),
              100 * mean(iv >= k), min(v), median(v), max(v)))
}

S <- gym_flatten_states(dat$S); R <- as.vector(dat$R)
X <- as.data.frame(S); names(X) <- paste0("f", seq_len(ncol(S)))

cat("\n=== 2. Conditional action effect by cut-off: lm(SOFA_{t+1} ~ state + A) ===\n\n")
cat(sprintf("%-14s %8s | %10s %8s %7s | %10s\n", "cut-off", "P(A=1)", "adj. coef", "se", "t", "raw diff"))
for (k in 1:4) {
  A <- as.integer(iv >= k)
  co <- summary(lm(R ~ ., data = cbind(R = R, X, A = A)))$coefficients["A", ]
  cat(sprintf("iv_input >= %d  %7.1f%% | %+10.4f %8.4f %7.2f | %+10.4f\n", k, 100 * mean(A),
              co[1], co[2], co[3], mean(R[A == 1]) - mean(R[A == 0])))
}

cat("\n=== 3. Dose-response: each dose level vs none, adjusted for the state ===\n\n")
cf <- summary(lm(R ~ ., data = cbind(R = R, X, dose = factor(iv))))$coefficients
cf <- cf[grep("^dose", rownames(cf)), , drop = FALSE]
for (i in seq_len(nrow(cf))) {
  cat(sprintf("  %-6s vs 0:  %+8.4f  (se %.4f, t = %6.2f)\n", rownames(cf)[i], cf[i, 1], cf[i, 2], cf[i, 3]))
}
