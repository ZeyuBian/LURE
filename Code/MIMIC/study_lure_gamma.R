#!/usr/bin/env Rscript

# LURE: discount factor sweep, without cross-fitting.
#
# Two variance levers are examined together:
#   1. cross_fit = FALSE  -- nuisances fit once on all 500 trajectories and evaluated
#      on the same data. Removes the single-split variability, which was measured at
#      ~15 points for the No-IV policy.
#   2. gamma < 0.9 -- the influence function carries 1/(1-gamma) on T1 and
#      gamma/(1-gamma) on T2. At gamma = 0.9 those are 10 and 9; at gamma = 0.5 they
#      are 2 and 1. The value scale V ~ E[R]/(1-gamma) shrinks by the same factor, so
#      the residual (V(S') - MV) shrinks too.
#
# em_gym() does not use its gamma argument -- the EM nuisances are conditional data
# distributions, not value functions -- so ONE EM fit on the full sample serves every
# gamma. That is what makes this sweep cheap.
#
# Because the estimand itself depends on gamma, the report also gives the per-step
# normalisation (1 - gamma) * V, which is comparable across gamma, and the ratio
# se / V, which measures relative precision.
#
# Usage from Code/MIMIC:
#   Rscript study_lure_gamma.R

script_path <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])
script_dir <- if (is.na(script_path)) normalizePath(getwd()) else dirname(normalizePath(script_path))
source(file.path(script_dir, "MIMIC.R"))

GAMMAS <- c(0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
POLICIES <- c(high_dose = "Always IV", low_dose = "No IV", sofa_11 = "SOFA-tailored")
out_file <- file.path(script_dir, "lure_gamma_sweep.rds")

baseline <- new.env()
load(file.path(script_dir, "res.RData"), envir = baseline)
dat <- baseline$results_high$dat
bridge_index <- as.integer(baseline$results_high$mr_out$bridge_index)
n_traj <- dim(dat$S)[1]
TT <- dim(dat$S)[2]

cat("N =", n_traj, " T =", TT, " d =", length(dat$state_names), "\n")
cat("Proxy:", dat$state_names[bridge_index], "\n")
cat("Reward timing:", dat$reward_timing, "\n\n")

# ---- one EM fit on the full sample, reused for every gamma -------------------
em_cache <- file.path(script_dir, "lure_em_full.rds")
if (file.exists(em_cache)) {
  cat("Reusing cached full-sample EM fit\n\n")
  em_full <- readRDS(em_cache)
} else {
  cat("Fitting EM once on all", n_traj, "trajectories ...\n"); flush.console()
  set.seed(23)
  t0 <- proc.time()[["elapsed"]]
  em_full <- em_gym(dat, gamma = NA_real_)   # gamma is unused by em_gym
  cat("  done in", round(proc.time()[["elapsed"]] - t0, 1), "s\n\n")
  saveRDS(em_full, em_cache)
}

S_all  <- gym_flatten_states(dat$S)
At_all <- as.vector(dat$Atilde)
R_all  <- as.vector(dat$R)
Sp_all <- gym_flatten_states(dat$Sp)
n_all  <- nrow(S_all)

# Nuisances that do not depend on gamma: evaluate once.
tR0 <- em_full$predict_theta_R(S_all, 0); tR1 <- em_full$predict_theta_R(S_all, 1)
tAt0 <- em_full$predict_mu(S_all, 0);     tAt1 <- em_full$predict_mu(S_all, 1)
tSp0 <- em_full$predict_theta_Sp(S_all, 0)[, bridge_index]
tSp1 <- em_full$predict_theta_Sp(S_all, 1)[, bridge_index]
Sp_bridge <- Sp_all[, bridge_index]

bridge_clip_q <- 0.98; bridge_abs_cap <- 10
omega_clip_q  <- 0.98; omega_abs_cap  <- 10

br_At_0 <- gym_clip_abs_quantile(gym_safe_ratio(At_all - tAt1, tAt0 - tAt1), bridge_clip_q, bridge_abs_cap)
br_At_1 <- gym_clip_abs_quantile(gym_safe_ratio(At_all - tAt0, tAt1 - tAt0), bridge_clip_q, bridge_abs_cap)
br_R_0  <- gym_clip_abs_quantile(gym_safe_ratio(R_all - tR1, tR0 - tR1),     bridge_clip_q, bridge_abs_cap)
br_R_1  <- gym_clip_abs_quantile(gym_safe_ratio(R_all - tR0, tR1 - tR0),     bridge_clip_q, bridge_abs_cap)
br_Sp_0 <- gym_clip_abs_quantile(gym_safe_ratio(Sp_bridge - tSp1, tSp0 - tSp1), bridge_clip_q, bridge_abs_cap)
br_Sp_1 <- gym_clip_abs_quantile(gym_safe_ratio(Sp_bridge - tSp0, tSp1 - tSp0), bridge_clip_q, bridge_abs_cap)

g0  <- gym_clip_abs_quantile(br_At_0 * br_R_0,  bridge_clip_q, bridge_abs_cap)
g1  <- gym_clip_abs_quantile(br_At_1 * br_R_1,  bridge_clip_q, bridge_abs_cap)
gp0 <- gym_clip_abs_quantile(br_At_0 * br_Sp_0, bridge_clip_q, bridge_abs_cap)
gp1 <- gym_clip_abs_quantile(br_At_1 * br_Sp_1, bridge_clip_q, bridge_abs_cap)

rows <- list()
psi_store <- list()   # psi_store[[gamma]][[policy]] = per-trajectory psi
for (pol in names(POLICIES)) {
  policy <- mimic_fit_target_policy(dat, policy_type = pol)
  dgp <- generate_mimic_dgp(dat, pi_func = policy$pi_func, bridge_index = bridge_index)

  for (gam in GAMMAS) {
    t0 <- proc.time()[["elapsed"]]
    omega_out <- mimic_solve_omega(em_full, dat, dgp, gam)
    fqe <- weighted_fqe_gym(em_full, dat, dgp, gam)

    om0 <- omega_out$predict_omega(S_all, rep(0L, n_all))
    om1 <- omega_out$predict_omega(S_all, rep(1L, n_all))
    om_cap <- min(as.numeric(quantile(c(abs(om0), abs(om1)), omega_clip_q, names = FALSE)),
                  omega_abs_cap)
    om0c <- pmin(pmax(om0, -om_cap), om_cap); om1c <- pmin(pmax(om1, -om_cap), om_cap)

    V_sp <- fqe$predict_V(Sp_all)
    Q0 <- fqe$predict_Q(S_all, 0); Q1 <- fqe$predict_Q(S_all, 1)
    MV0 <- (Q0 - tR0) / gam; MV1 <- (Q1 - tR1) / gam

    T1 <- gp0 * om0c * (R_all - tR0) + gp1 * om1c * (R_all - tR1)
    T2 <- g0  * om0c * (V_sp - MV0)  + g1  * om1c * (V_sp - MV1)
    phi <- T1 / (1 - gam) + T2 * gam / (1 - gam)

    direct_terms <- fqe$predict_V(dat$init_states)
    t1_terms <- rowMeans(matrix(T1 / (1 - gam), nrow = n_traj, ncol = TT))
    t2_terms <- rowMeans(matrix(T2 * gam / (1 - gam), nrow = n_traj, ncol = TT))
    corr_terms <- rowMeans(matrix(phi, nrow = n_traj, ncol = TT))

    psi <- direct_terms + corr_terms
    V <- mean(psi); se <- stats::sd(psi) / sqrt(n_traj)
    psi_store[[sprintf("%.1f", gam)]][[pol]] <- psi

    rows[[length(rows) + 1L]] <- data.frame(
      policy = pol, policy_label = unname(POLICIES[[pol]]), gamma = gam,
      V = V, se = se, ci_lo = V - 1.96 * se, ci_hi = V + 1.96 * se,
      width = 2 * 1.96 * se,
      direct = mean(direct_terms), correction = mean(corr_terms),
      sd_T1 = stats::sd(t1_terms), sd_T2 = stats::sd(t2_terms),
      sd_direct = stats::sd(direct_terms),
      per_step_V = (1 - gam) * V, per_step_se = (1 - gam) * se,
      rel_se = se / abs(V), elapsed = proc.time()[["elapsed"]] - t0,
      stringsAsFactors = FALSE
    )
    cat(sprintf("  %-14s gamma=%.1f  V=%8.3f  se=%7.3f  width=%7.2f  (%.0fs)\n",
                POLICIES[[pol]], gam, V, se, 2 * 1.96 * se,
                proc.time()[["elapsed"]] - t0))
    flush.console()
  }
  cat("\n")
}

res <- do.call(rbind, rows)
saveRDS(list(summary = res, psi = psi_store), out_file)

fmt <- function(x, d = 2) formatC(x, format = "f", digits = d)

cat("\n=== LURE without cross-fitting, by discount factor ===\n\n")
cat(sprintf("%-14s %6s %9s %8s %9s %9s %9s\n",
            "policy", "gamma", "V", "se", "CI width", "rel se", "(1-g)*V"))
for (i in seq_len(nrow(res))) {
  r <- res[i, ]
  cat(sprintf("%-14s %6.1f %9.3f %8.3f %9.2f %9.4f %9.3f\n",
              r$policy_label, r$gamma, r$V, r$se, r$width, r$rel_se, r$per_step_V))
}

cat("\n=== Variance sources by gamma (always-IV) ===\n\n")
cat(sprintf("%6s %11s %11s %11s %11s\n", "gamma", "sd(direct)", "sd(T1)", "sd(T2)", "correction"))
h <- res[res$policy == "high_dose", ]
for (i in seq_len(nrow(h))) {
  cat(sprintf("%6.1f %11.3f %11.3f %11.3f %11.3f\n",
              h$gamma[i], h$sd_direct[i], h$sd_T1[i], h$sd_T2[i], h$correction[i]))
}

cat("\n=== Policy values at each gamma (per-step scale, (1-gamma)*V) ===\n\n")
cat(sprintf("%6s %12s %12s %12s   %s\n", "gamma", "AlwaysIV", "NoIV", "SOFA", "ranking (best first)"))
for (gam in GAMMAS) {
  sub <- res[res$gamma == gam, ]
  v <- setNames(sub$per_step_V, sub$policy)
  ord <- names(sort(v))
  cat(sprintf("%6.1f %12.4f %12.4f %12.4f   %s\n", gam,
              v[["high_dose"]], v[["low_dose"]], v[["sofa_11"]],
              paste(unname(POLICIES[ord]), collapse = " < ")))
}

# Pairwise policy contrasts. The two policy values are estimated from the SAME
# trajectories with the SAME nuisances, so psi_i(pi_a) and psi_i(pi_b) are strongly
# correlated. A paired SE on the per-trajectory differences is the correct standard
# error for a contrast; treating the two estimates as independent badly overstates it.
cat("\n=== Pairwise contrasts: paired vs independent SE ===\n\n")
cat(sprintf("%6s %-26s %9s %9s %8s %9s %8s\n",
            "gamma", "contrast", "diff", "paired SE", "z", "indep SE", "z"))
pairs <- list(c("high_dose", "low_dose"), c("high_dose", "sofa_11"), c("sofa_11", "low_dose"))
for (gam in GAMMAS) {
  key <- sprintf("%.1f", gam)
  sub <- res[res$gamma == gam, ]
  for (pr in pairs) {
    a <- pr[1]; b <- pr[2]
    d <- psi_store[[key]][[b]] - psi_store[[key]][[a]]   # positive => a is better
    diff <- mean(d)
    se_paired <- stats::sd(d) / sqrt(n_traj)
    se_indep <- sqrt(sub$se[sub$policy == a]^2 + sub$se[sub$policy == b]^2)
    cat(sprintf("%6.1f %-26s %9.3f %9.3f %8.2f %9.3f %8.2f\n",
                gam, paste0(POLICIES[[a]], " vs ", POLICIES[[b]]),
                diff, se_paired, diff / se_paired, se_indep, diff / se_indep))
  }
  cat("\n")
}

cat("Correlation of per-trajectory psi across policies (gamma = 0.5 / 0.9):\n")
for (key in c("0.5", "0.9")) {
  m <- do.call(cbind, psi_store[[key]])
  cat("  gamma =", key, " cor(AlwaysIV, NoIV) =",
      fmt(cor(m[, "high_dose"], m[, "low_dose"]), 3), "\n")
}

cat("\nSaved to", out_file, "\n")
