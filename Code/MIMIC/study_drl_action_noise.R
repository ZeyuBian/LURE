#!/usr/bin/env Rscript

# DRL only: sensitivity of the policy value and, more importantly, of the
# POLICY CONTRAST to extra misclassification of the observed action.
#
# Motivation. In the gamma sweep DRL assigned all three policies nearly the same
# value (per-step spread ~0.02) while LURE separated them (~0.63). The proposed
# explanation was attenuation: DRL treats the recorded action A~ as the true action,
# and misclassification biases an estimated contrast toward zero. This script tests
# that explanation by injecting known extra noise and watching the contrast decay.
#
# For symmetric binary misclassification at rate tau, classical attenuation predicts
# the contrast shrinks by (1 - 2*tau). Comparing the measured decay against that
# benchmark says whether DRL behaves like an attenuated estimator, and the implied
# tau needed to explain the LURE/DRL gap says whether attenuation alone is a
# plausible account of it.
#
# Noise model: independent symmetric flips. One fixed Uniform(0,1) draw per action
# per replication, so perturbations are nested in tau (anything flipped at a lower
# rate stays flipped at a higher one) and the curves are smooth in tau.
#
# LURE is not re-run. States, rewards, patients and target policies are unchanged.
#
# Usage from Code/MIMIC:
#   Rscript study_drl_action_noise.R

script_path <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])
script_dir <- if (is.na(script_path)) normalizePath(getwd()) else dirname(normalizePath(script_path))
source(file.path(script_dir, "MIMIC.R"))

TAUS   <- c(0, 0.05, 0.10, 0.20, 0.30, 0.40)
GAMMAS <- c(0.4, 0.5, 0.7, 0.9)
N_REP  <- 5L            # replications per tau > 0; tau = 0 is deterministic
POLICIES <- c(high_dose = "Always IV", low_dose = "No IV", sofa_11 = "SOFA-tailored")
out_file <- file.path(script_dir, "drl_action_noise.rds")

baseline <- new.env()
load(file.path(script_dir, "res.RData"), envir = baseline)
dat <- baseline$results_high$dat
bridge_index <- as.integer(baseline$results_high$mr_out$bridge_index)
n_traj <- dim(dat$S)[1]
original_action <- dat$Atilde
stopifnot(all(original_action %in% c(0L, 1L)))

cat("N =", n_traj, " T =", dim(dat$S)[2], " d =", length(dat$state_names), "\n")
cat("Observed P(A=1) =", formatC(mean(original_action), digits = 4, format = "f"), "\n")
cat("tau grid:", paste0(100 * TAUS, "%", collapse = ", "), "| replications per tau>0:", N_REP, "\n\n")

pairs <- list(c("high_dose", "low_dose"), c("high_dose", "sofa_11"), c("sofa_11", "low_dose"))

rows <- list(); contrast_rows <- list()

for (rep_id in seq_len(N_REP)) {
  set.seed(20260921L + rep_id)
  noise_u <- matrix(runif(length(original_action)),
                    nrow = nrow(original_action), ncol = ncol(original_action))

  for (tau in TAUS) {
    if (tau == 0 && rep_id > 1L) next    # tau = 0 does not depend on the draw

    flip <- noise_u < tau
    dat_tau <- dat
    dat_tau$Atilde <- original_action
    dat_tau$Atilde[flip] <- 1L - dat_tau$Atilde[flip]

    for (gam in GAMMAS) {
      psi_by_pol <- list()
      for (pol in names(POLICIES)) {
        policy <- mimic_fit_target_policy(dat_tau, policy_type = pol)
        dgp <- generate_mimic_dgp(dat_tau, pi_func = policy$pi_func, bridge_index = bridge_index)
        set.seed(23)
        drl <- mimic_drl_estimate(dat_tau, dgp, gam)
        psi_by_pol[[pol]] <- drl$psi

        rows[[length(rows) + 1L]] <- data.frame(
          rep = rep_id, tau = tau, realized_flip = mean(flip), gamma = gam,
          policy = pol, policy_label = unname(POLICIES[[pol]]),
          V = drl$V_hat, se = drl$se, ci_lo = drl$ci_lo, ci_hi = drl$ci_hi,
          per_step_V = (1 - gam) * drl$V_hat,
          direct = drl$direct, correction = drl$correction,
          stringsAsFactors = FALSE
        )
      }
      for (pr in pairs) {
        d <- psi_by_pol[[pr[2]]] - psi_by_pol[[pr[1]]]
        contrast_rows[[length(contrast_rows) + 1L]] <- data.frame(
          rep = rep_id, tau = tau, gamma = gam,
          contrast = paste0(POLICIES[[pr[1]]], " vs ", POLICIES[[pr[2]]]),
          diff = mean(d), se = stats::sd(d) / sqrt(n_traj),
          z = mean(d) / (stats::sd(d) / sqrt(n_traj)),
          per_step_diff = (1 - gam) * mean(d),
          stringsAsFactors = FALSE
        )
      }
    }
    cat(sprintf("  rep %d, tau = %.2f (realized %.3f): done\n", rep_id, tau, mean(flip)))
    flush.console()
  }
}

res <- do.call(rbind, rows)
con <- do.call(rbind, contrast_rows)
saveRDS(list(values = res, contrasts = con, taus = TAUS, gammas = GAMMAS), out_file)

agg <- function(df, keys, val) {
  a <- aggregate(df[[val]], by = df[keys], FUN = mean)
  names(a)[ncol(a)] <- "mean"
  s <- aggregate(df[[val]], by = df[keys], FUN = function(x) if (length(x) > 1) sd(x) else 0)
  a$sd <- s$x
  a
}

cat("\n\n=== DRL policy values and 95% CIs by action-noise rate (gamma = 0.5) ===\n")
cat("    (averaged over replications)\n\n")
sub <- res[res$gamma == 0.5, ]
cat(sprintf("%6s %-15s %9s %-22s\n", "tau", "policy", "V", "95% CI"))
for (tau in TAUS) {
  for (pol in c("high_dose", "sofa_11", "low_dose")) {
    s <- sub[sub$tau == tau & sub$policy == pol, ]
    cat(sprintf("%5.0f%% %-15s %9.2f [%7.2f, %7.2f]\n", 100 * tau,
                POLICIES[[pol]], mean(s$V), mean(s$ci_lo), mean(s$ci_hi)))
  }
  cat("\n")
}

cat("=== Policy contrast vs action noise: does DRL attenuate? ===\n")
cat("    Contrast = Always IV vs No IV, per-step scale. Ratio is contrast(tau)/contrast(0).\n")
cat("    Classical symmetric misclassification predicts ratio = 1 - 2*tau.\n\n")
for (gam in GAMMAS) {
  cat(sprintf("gamma = %.1f\n", gam))
  cat(sprintf("%6s %12s %10s %10s %10s %10s\n",
              "tau", "contrast", "sd(rep)", "ratio", "1-2*tau", "z"))
  base <- mean(con$per_step_diff[con$gamma == gam & con$tau == 0 &
                                   con$contrast == "Always IV vs No IV"])
  for (tau in TAUS) {
    s <- con[con$gamma == gam & con$tau == tau & con$contrast == "Always IV vs No IV", ]
    cat(sprintf("%5.0f%% %12.4f %10.4f %10.3f %10.3f %10.2f\n",
                100 * tau, mean(s$per_step_diff),
                if (nrow(s) > 1) sd(s$per_step_diff) else 0,
                mean(s$per_step_diff) / base, 1 - 2 * tau, mean(s$z)))
  }
  cat("\n")
}

cat("=== All three contrasts, z-statistics by tau (gamma = 0.5) ===\n\n")
cat(sprintf("%6s %-28s %10s %8s\n", "tau", "contrast", "diff", "z"))
for (tau in TAUS) {
  for (cn in unique(con$contrast)) {
    s <- con[con$gamma == 0.5 & con$tau == tau & con$contrast == cn, ]
    cat(sprintf("%5.0f%% %-28s %10.4f %8.2f\n", 100 * tau, cn,
                mean(s$per_step_diff), mean(s$z)))
  }
  cat("\n")
}

# How much extra misclassification would be needed for DRL at tau = 0 to be an
# attenuated version of LURE's contrast? Solve LURE_contrast * (1 - 2*tau) = DRL_contrast.
lure_file <- file.path(script_dir, "lure_gamma_sweep.rds")
if (file.exists(lure_file)) {
  lure <- readRDS(lure_file)
  cat("=== Implied misclassification rate: what tau explains the LURE/DRL gap? ===\n\n")
  cat(sprintf("%6s %12s %12s %10s %12s\n",
              "gamma", "LURE diff", "DRL diff", "ratio", "implied tau"))
  for (gam in GAMMAS) {
    lp <- lure$psi[[sprintf("%.1f", gam)]]
    ld <- (1 - gam) * mean(lp[["low_dose"]] - lp[["high_dose"]])
    dd <- mean(con$per_step_diff[con$gamma == gam & con$tau == 0 &
                                    con$contrast == "Always IV vs No IV"])
    ratio <- dd / ld
    cat(sprintf("%6.1f %12.4f %12.4f %10.3f %12s\n", gam, ld, dd, ratio,
                if (ratio > 0 && ratio < 1) formatC((1 - ratio) / 2, format = "f", digits = 3)
                else "> 0.5 (n/a)"))
  }
  cat("\n    An implied tau near or above 0.5 means attenuation of A~ alone cannot\n")
  cat("    account for the gap: at tau = 0.5 the action is pure noise.\n")
}

cat("\nSaved to", out_file, "\n")
