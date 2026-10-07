#!/usr/bin/env Rscript

# DRL: discount factor sweep, and side-by-side comparison against the cached LURE
# sweep from study_lure_gamma.R.
#
# DRL never cross-fits -- mimic_drl_estimate() fits FQE and MIS on all trajectories
# and evaluates on the same data -- so the "no cross-fitting" condition already holds
# and only gamma is swept. No EM is involved, so this runs in seconds.
#
# Usage from Code/MIMIC:
#   Rscript study_drl_gamma.R        (requires lure_gamma_sweep.rds for the comparison)

script_path <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])
script_dir <- if (is.na(script_path)) normalizePath(getwd()) else dirname(normalizePath(script_path))
source(file.path(script_dir, "MIMIC.R"))

GAMMAS <- c(0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
POLICIES <- c(high_dose = "Always IV", low_dose = "No IV", sofa_11 = "SOFA-tailored")
out_file <- file.path(script_dir, "drl_gamma_sweep.rds")
lure_file <- file.path(script_dir, "lure_gamma_sweep.rds")

baseline <- new.env()
load(file.path(script_dir, "res.RData"), envir = baseline)
dat <- baseline$results_high$dat
bridge_index <- as.integer(baseline$results_high$mr_out$bridge_index)
n_traj <- dim(dat$S)[1]

cat("N =", n_traj, " T =", dim(dat$S)[2], " d =", length(dat$state_names), "\n")
cat("Reward timing:", dat$reward_timing, "\n\n")

rows <- list()
psi_store <- list()

for (pol in names(POLICIES)) {
  policy <- mimic_fit_target_policy(dat, policy_type = pol)
  dgp <- generate_mimic_dgp(dat, pi_func = policy$pi_func, bridge_index = bridge_index)

  for (gam in GAMMAS) {
    set.seed(23)   # naive_fqe_gym draws initial states; fix it for reproducibility
    t0 <- proc.time()[["elapsed"]]
    drl <- mimic_drl_estimate(dat, dgp, gam)

    psi <- drl$psi
    V <- drl$V_hat; se <- drl$se
    psi_store[[sprintf("%.1f", gam)]][[pol]] <- psi

    rows[[length(rows) + 1L]] <- data.frame(
      policy = pol, policy_label = unname(POLICIES[[pol]]), gamma = gam,
      V = V, se = se, ci_lo = drl$ci_lo, ci_hi = drl$ci_hi, width = drl$ci_hi - drl$ci_lo,
      direct = drl$direct, correction = drl$correction,
      V_fqe = drl$V_fqe, V_mis = drl$V_mis,
      sd_direct = stats::sd(drl$direct_terms),
      sd_correction = stats::sd(drl$correction_terms),
      per_step_V = (1 - gam) * V, rel_se = se / abs(V),
      elapsed = proc.time()[["elapsed"]] - t0, stringsAsFactors = FALSE
    )
    cat(sprintf("  %-14s gamma=%.1f  V=%8.3f  se=%7.3f  width=%7.2f  (%.0fs)\n",
                POLICIES[[pol]], gam, V, se, drl$ci_hi - drl$ci_lo,
                proc.time()[["elapsed"]] - t0))
    flush.console()
  }
  cat("\n")
}

res <- do.call(rbind, rows)
saveRDS(list(summary = res, psi = psi_store), out_file)

fmt <- function(x, d = 2) formatC(x, format = "f", digits = d)

cat("\n=== DRL by discount factor ===\n\n")
cat(sprintf("%-14s %6s %9s %8s %9s %9s %9s\n",
            "policy", "gamma", "V", "se", "CI width", "rel se", "(1-g)*V"))
for (i in seq_len(nrow(res))) {
  r <- res[i, ]
  cat(sprintf("%-14s %6.1f %9.3f %8.3f %9.2f %9.4f %9.3f\n",
              r$policy_label, r$gamma, r$V, r$se, r$width, r$rel_se, r$per_step_V))
}

cat("\n=== DRL variance sources (always-IV) ===\n\n")
cat(sprintf("%6s %11s %14s %11s\n", "gamma", "sd(direct)", "sd(correction)", "correction"))
h <- res[res$policy == "high_dose", ]
for (i in seq_len(nrow(h))) {
  cat(sprintf("%6.1f %11.3f %14.3f %11.3f\n",
              h$gamma[i], h$sd_direct[i], h$sd_correction[i], h$correction[i]))
}

# ------------------------------------------------------------------ comparison

if (!file.exists(lure_file)) {
  cat("\n[no lure_gamma_sweep.rds found; skipping comparison]\n")
  quit(save = "no")
}
lure <- readRDS(lure_file)
lure_res <- lure$summary
lure_psi <- lure$psi

cat("\n\n=== LURE vs DRL: standard error by gamma ===\n\n")
cat(sprintf("%-14s %6s | %9s %9s %8s | %9s %9s\n",
            "policy", "gamma", "LURE se", "DRL se", "ratio", "LURE width", "DRL width"))
for (pol in names(POLICIES)) {
  for (gam in GAMMAS) {
    l <- lure_res[lure_res$policy == pol & lure_res$gamma == gam, ]
    d <- res[res$policy == pol & res$gamma == gam, ]
    cat(sprintf("%-14s %6.1f | %9.3f %9.3f %8.2f | %9.2f %9.2f\n",
                POLICIES[[pol]], gam, l$se, d$se, l$se / d$se, l$width, d$width))
  }
  cat("\n")
}

cat("=== LURE vs DRL: point estimates (per-step scale, (1-gamma)*V) ===\n\n")
cat(sprintf("%6s | %-28s | %-28s\n", "gamma", "LURE  (IV / SOFA / NoIV)", "DRL   (IV / SOFA / NoIV)"))
for (gam in GAMMAS) {
  lv <- setNames(lure_res$per_step_V[lure_res$gamma == gam], lure_res$policy[lure_res$gamma == gam])
  dv <- setNames(res$per_step_V[res$gamma == gam], res$policy[res$gamma == gam])
  cat(sprintf("%6.1f | %8.3f %9.3f %9.3f | %8.3f %9.3f %9.3f\n", gam,
              lv[["high_dose"]], lv[["sofa_11"]], lv[["low_dose"]],
              dv[["high_dose"]], dv[["sofa_11"]], dv[["low_dose"]]))
}

pairs <- list(c("high_dose", "low_dose"), c("high_dose", "sofa_11"), c("sofa_11", "low_dose"))
paired_z <- function(store, key, a, b) {
  d <- store[[key]][[b]] - store[[key]][[a]]
  mean(d) / (stats::sd(d) / sqrt(n_traj))
}

cat("\n=== LURE vs DRL: paired contrast z-statistics ===\n")
cat("    (positive => first policy has the lower / better SOFA value)\n\n")
cat(sprintf("%6s %-28s | %9s %9s\n", "gamma", "contrast", "LURE z", "DRL z"))
for (gam in GAMMAS) {
  key <- sprintf("%.1f", gam)
  for (pr in pairs) {
    cat(sprintf("%6.1f %-28s | %9.2f %9.2f\n", gam,
                paste0(POLICIES[[pr[1]]], " vs ", POLICIES[[pr[2]]]),
                paired_z(lure_psi, key, pr[1], pr[2]),
                paired_z(psi_store, key, pr[1], pr[2])))
  }
  cat("\n")
}

cat("=== Contrasts significant at |z| > 1.96, out of 3 per gamma ===\n\n")
cat(sprintf("%6s %10s %10s\n", "gamma", "LURE", "DRL"))
for (gam in GAMMAS) {
  key <- sprintf("%.1f", gam)
  lz <- vapply(pairs, function(pr) paired_z(lure_psi, key, pr[1], pr[2]), numeric(1))
  dz <- vapply(pairs, function(pr) paired_z(psi_store, key, pr[1], pr[2]), numeric(1))
  cat(sprintf("%6.1f %10d %10d\n", gam, sum(abs(lz) > 1.96), sum(abs(dz) > 1.96)))
}

cat("\nSaved to", out_file, "\n")
