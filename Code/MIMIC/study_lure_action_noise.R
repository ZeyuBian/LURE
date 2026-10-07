#!/usr/bin/env Rscript

# LURE: sensitivity to extra misclassification of the observed action.
#
# Mirrors study_drl_action_noise.R exactly: same tau grid, same gammas, and the SAME
# noise draws (seed 20260921 + rep), so every LURE replication is paired with the DRL
# replication that saw identical flipped actions. LURE is run without cross-fitting,
# with the production clipping constants, as in study_lure_gamma.R.
#
# What this tests. The E-step diagnostic showed the EM latent class is driven almost
# entirely by the next-state likelihood (lab carry-forward), with the observed action
# contributing ~0.002% of the classification signal. If so, flipping A~ should barely
# move the latent classes. The one channel through which A~ matters a lot is the
# class-labelling rule in em_gym(): the class with the higher mean A~ is called "IV",
# and that gap is only ~0.016. Symmetric flips shrink it by (1 - 2*tau), so labels may
# swap at higher tau, flipping the sign of every policy contrast. Each fit therefore
# records whether its latent classes match, swap, or re-cluster relative to tau = 0.
#
# Cost. Each noise draw needs a fresh EM fit (~4 min, ~1 GB). Fits run in parallel
# (LURE_WORKERS, default 4) and are checkpointed to lure_noise_jobs/, so an
# interrupted run resumes where it stopped.
#
# Usage from Code/MIMIC:
#   env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
#     Rscript study_lure_action_noise.R

script_path <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])
script_dir <- if (is.na(script_path)) normalizePath(getwd()) else dirname(normalizePath(script_path))
source(file.path(script_dir, "MIMIC.R"))
suppressPackageStartupMessages(library(parallel))

TAUS   <- c(0, 0.05, 0.10, 0.20, 0.30, 0.40)
GAMMAS <- c(0.4, 0.5, 0.7, 0.9)
N_REP  <- 5L
N_WORKERS <- as.integer(Sys.getenv("LURE_WORKERS", "4"))
POLICIES <- c(high_dose = "Always IV", low_dose = "No IV", sofa_11 = "SOFA-tailored")
PAIRS <- list(c("high_dose", "low_dose"), c("high_dose", "sofa_11"), c("sofa_11", "low_dose"))

job_dir  <- file.path(script_dir, "lure_noise_jobs")
out_file <- file.path(script_dir, "lure_action_noise.rds")
dir.create(job_dir, showWarnings = FALSE)

baseline <- new.env()
load(file.path(script_dir, "res.RData"), envir = baseline)
dat <- baseline$results_high$dat
bridge_index <- as.integer(baseline$results_high$mr_out$bridge_index)
n_traj <- dim(dat$S)[1]
TT <- dim(dat$S)[2]
original_action <- dat$Atilde

S_all  <- gym_flatten_states(dat$S)
R_all  <- as.vector(dat$R)
Sp_all <- gym_flatten_states(dat$Sp)
n_all  <- nrow(S_all)
nm <- dat$state_names
unchanged <- function(v) abs(Sp_all[, match(v, nm)] - S_all[, match(v, nm)]) < 1e-8
carried_forward <- unchanged("SGPT") & unchanged("SGOT")

# ---- LURE without cross-fitting, given an EM fit (identical to study_lure_gamma.R)
lure_given_em <- function(em, dat_tau) {
  At_all <- as.vector(dat_tau$Atilde)
  q <- 0.98; cap <- 10
  tR0 <- em$predict_theta_R(S_all, 0); tR1 <- em$predict_theta_R(S_all, 1)
  tAt0 <- em$predict_mu(S_all, 0);     tAt1 <- em$predict_mu(S_all, 1)
  tSp0 <- em$predict_theta_Sp(S_all, 0)[, bridge_index]
  tSp1 <- em$predict_theta_Sp(S_all, 1)[, bridge_index]
  Sp_b <- Sp_all[, bridge_index]
  clip <- function(x) gym_clip_abs_quantile(x, q, cap)
  bA0 <- clip(gym_safe_ratio(At_all - tAt1, tAt0 - tAt1))
  bA1 <- clip(gym_safe_ratio(At_all - tAt0, tAt1 - tAt0))
  bR0 <- clip(gym_safe_ratio(R_all - tR1, tR0 - tR1))
  bR1 <- clip(gym_safe_ratio(R_all - tR0, tR1 - tR0))
  bS0 <- clip(gym_safe_ratio(Sp_b - tSp1, tSp0 - tSp1))
  bS1 <- clip(gym_safe_ratio(Sp_b - tSp0, tSp1 - tSp0))
  g0 <- clip(bA0 * bR0); g1 <- clip(bA1 * bR1)
  gp0 <- clip(bA0 * bS0); gp1 <- clip(bA1 * bS1)

  rows <- list(); psi <- list()
  for (pol in names(POLICIES)) {
    policy <- mimic_fit_target_policy(dat_tau, policy_type = pol)
    dgp <- generate_mimic_dgp(dat_tau, pi_func = policy$pi_func, bridge_index = bridge_index)
    for (gam in GAMMAS) {
      omega_out <- mimic_solve_omega(em, dat_tau, dgp, gam)
      fqe <- weighted_fqe_gym(em, dat_tau, dgp, gam)
      om0 <- omega_out$predict_omega(S_all, rep(0L, n_all))
      om1 <- omega_out$predict_omega(S_all, rep(1L, n_all))
      oc <- min(as.numeric(quantile(c(abs(om0), abs(om1)), q, names = FALSE)), cap)
      om0 <- pmin(pmax(om0, -oc), oc); om1 <- pmin(pmax(om1, -oc), oc)
      V_sp <- fqe$predict_V(Sp_all)
      MV0 <- (fqe$predict_Q(S_all, 0) - tR0) / gam
      MV1 <- (fqe$predict_Q(S_all, 1) - tR1) / gam
      T1 <- gp0 * om0 * (R_all - tR0) + gp1 * om1 * (R_all - tR1)
      T2 <- g0 * om0 * (V_sp - MV0) + g1 * om1 * (V_sp - MV1)
      phi <- T1 / (1 - gam) + T2 * gam / (1 - gam)
      direct_terms <- fqe$predict_V(dat_tau$init_states)
      corr_terms <- rowMeans(matrix(phi, nrow = n_traj, ncol = TT))
      p <- direct_terms + corr_terms
      V <- mean(p); se <- stats::sd(p) / sqrt(n_traj)
      psi[[sprintf("%.1f", gam)]][[pol]] <- p
      rows[[length(rows) + 1L]] <- data.frame(
        gamma = gam, policy = pol, policy_label = unname(POLICIES[[pol]]),
        V = V, se = se, ci_lo = V - 1.96 * se, ci_hi = V + 1.96 * se,
        per_step_V = (1 - gam) * V, direct = mean(direct_terms),
        correction = mean(corr_terms), stringsAsFactors = FALSE
      )
    }
  }
  list(rows = do.call(rbind, rows), psi = psi)
}

em_diagnostics <- function(em, dat_tau, cls0) {
  cls <- as.integer(em$eta[, 2] > 0.5)
  A_tau <- as.vector(dat_tau$Atilde)
  agree0 <- mean(cls == cls0)
  data.frame(
    p_latent1 = mean(em$eta[, 2]),
    mu_gap = mean(em$predict_mu(S_all, 1) - em$predict_mu(S_all, 0)),
    agree_tau0 = agree0,
    labels = if (agree0 > 0.9) "same" else if (agree0 < 0.1) "SWAPPED" else "re-clustered",
    agree_carry_fwd = mean(cls == carried_forward),
    agree_A_perturbed = mean(cls == A_tau),
    agree_A_original = mean(cls == as.vector(original_action)),
    p_A1 = mean(A_tau),
    em_iter = em$n_iter,
    stringsAsFactors = FALSE
  )
}

perturb <- function(rep_id, tau) {
  set.seed(20260921L + rep_id)          # identical to study_drl_action_noise.R
  u <- matrix(runif(length(original_action)),
              nrow = nrow(original_action), ncol = ncol(original_action))
  dat_tau <- dat
  dat_tau$Atilde <- original_action
  flip <- u < tau
  dat_tau$Atilde[flip] <- 1L - dat_tau$Atilde[flip]
  list(dat = dat_tau, realized = mean(flip))
}

# ---- tau = 0 from the cached full-sample EM, and a reproduction check ----------
job0 <- file.path(job_dir, "tau0.rds")
if (!file.exists(job0)) {
  cat("tau = 0: using cached full-sample EM\n"); flush.console()
  em0 <- readRDS(file.path(script_dir, "lure_em_full.rds"))
  cls0 <- as.integer(em0$eta[, 2] > 0.5)
  r0 <- lure_given_em(em0, dat)
  d0 <- em_diagnostics(em0, dat, cls0)
  d0$bridge_selected <- nm[bridge_index]
  saveRDS(list(rep = 0L, tau = 0, realized = 0, rows = r0$rows, psi = r0$psi,
               diag = d0, cls0 = cls0), job0)
  rm(em0); invisible(gc())
}
j0 <- readRDS(job0)
cls0 <- j0$cls0

sweep <- readRDS(file.path(script_dir, "lure_gamma_sweep.rds"))$summary
chk <- merge(j0$rows[, c("gamma", "policy", "V")], sweep[, c("gamma", "policy", "V")],
             by = c("gamma", "policy"))
max_dev <- max(abs(chk$V.x - chk$V.y))
cat(sprintf("Reproduction check vs lure_gamma_sweep.rds: max |dV| = %.2e\n", max_dev))
if (max_dev > 1e-6) stop("tau = 0 does not reproduce the cached LURE sweep.")

# ---- tau > 0: one EM fit per (replication, tau), in parallel -------------------
jobs <- expand.grid(rep = seq_len(N_REP), tau = TAUS[TAUS > 0])
jobs$file <- file.path(job_dir, sprintf("rep%d_tau%03d.rds", jobs$rep, round(1000 * jobs$tau)))
todo <- which(!file.exists(jobs$file))
cat(sprintf("%d of %d noisy fits already done; running %d on %d workers\n\n",
            nrow(jobs) - length(todo), nrow(jobs), length(todo), N_WORKERS))
flush.console()

run_job <- function(i) {
  tryCatch({
    t0 <- proc.time()[["elapsed"]]
    pj <- perturb(jobs$rep[i], jobs$tau[i])
    set.seed(23)                           # same EM restart RNG as the tau = 0 fit
    em <- em_gym(pj$dat, gamma = NA_real_)
    t_em <- proc.time()[["elapsed"]] - t0
    r <- lure_given_em(em, pj$dat)
    d <- em_diagnostics(em, pj$dat, cls0)
    rm(em); invisible(gc())
    d$bridge_selected <- nm[select_bridge_index_gym(pj$dat)$bridge_index]
    saveRDS(list(rep = jobs$rep[i], tau = jobs$tau[i], realized = pj$realized,
                 rows = r$rows, psi = r$psi, diag = d), jobs$file[i])
    cat(sprintf("[done] rep=%d tau=%.2f  EM %.0fs total %.0fs  mu_gap=%+.4f  labels=%s\n",
                jobs$rep[i], jobs$tau[i], t_em, proc.time()[["elapsed"]] - t0,
                d$mu_gap, d$labels))
    "ok"
  }, error = function(e) {
    cat(sprintf("[FAILED] rep=%d tau=%.2f: %s\n", jobs$rep[i], jobs$tau[i], conditionMessage(e)))
    conditionMessage(e)
  })
}

if (length(todo) > 0L) {
  status <- mclapply(todo, run_job, mc.cores = N_WORKERS, mc.preschedule = FALSE)
  failed <- todo[vapply(status, function(s) !identical(s, "ok"), logical(1))]
  if (length(failed) > 0L) {
    stop(length(failed), " fits failed; rerun to retry them (completed fits are kept).")
  }
}

# ---- assemble ------------------------------------------------------------------
all_jobs <- c(list(j0), lapply(jobs$file, readRDS))
vals <- do.call(rbind, lapply(all_jobs, function(j) cbind(rep = j$rep, tau = j$tau, j$rows)))
diag <- do.call(rbind, lapply(all_jobs, function(j) cbind(rep = j$rep, tau = j$tau,
                                                         realized = j$realized, j$diag)))
con <- do.call(rbind, lapply(all_jobs, function(j) {
  do.call(rbind, lapply(GAMMAS, function(gam) {
    k <- sprintf("%.1f", gam)
    do.call(rbind, lapply(PAIRS, function(pr) {
      d <- j$psi[[k]][[pr[2]]] - j$psi[[k]][[pr[1]]]
      data.frame(rep = j$rep, tau = j$tau, gamma = gam,
                 contrast = paste0(POLICIES[[pr[1]]], " vs ", POLICIES[[pr[2]]]),
                 diff = mean(d), per_step_diff = (1 - gam) * mean(d),
                 z = mean(d) / (stats::sd(d) / sqrt(n_traj)), stringsAsFactors = FALSE)
    }))
  }))
}))
saveRDS(list(values = vals, contrasts = con, diagnostics = diag), out_file)

# ---- report --------------------------------------------------------------------
cat("\n=== EM diagnostics by replication: does flipping A~ move the latent classes? ===\n\n")
cat(sprintf("%5s %4s %8s %9s %9s %-13s %9s %9s  %s\n", "tau", "rep", "P(A~=1)",
            "mu gap", "agree t0", "labels", "carry-fwd", "agree A~", "bridge"))
for (i in order(diag$tau, diag$rep)) {
  r <- diag[i, ]
  cat(sprintf("%4.0f%% %4d %8.3f %+9.4f %9.3f %-13s %9.3f %9.3f  %s\n",
              100 * r$tau, r$rep, r$p_A1, r$mu_gap, r$agree_tau0, r$labels,
              r$agree_carry_fwd, r$agree_A_perturbed, r$bridge_selected))
}

cat("\n=== LURE values and 95% CIs by action-noise rate (gamma = 0.5) ===\n")
cat("    mean over replications; [min, max] of V across replications in the last column\n\n")
cat(sprintf("%6s %-15s %8s %-20s %-20s\n", "tau", "policy", "V", "95% CI", "V range (reps)"))
for (tau in TAUS) {
  for (pol in c("high_dose", "sofa_11", "low_dose")) {
    s <- vals[vals$gamma == 0.5 & vals$tau == tau & vals$policy == pol, ]
    cat(sprintf("%5.0f%% %-15s %8.2f [%7.2f, %7.2f]  [%6.2f, %6.2f]\n", 100 * tau,
                POLICIES[[pol]], mean(s$V), mean(s$ci_lo), mean(s$ci_hi), min(s$V), max(s$V)))
  }
  cat("\n")
}

drl_file <- file.path(script_dir, "drl_action_noise.rds")
drl <- if (file.exists(drl_file)) readRDS(drl_file) else NULL

cat("=== Always IV vs No IV contrast by tau: LURE vs DRL (per-step scale) ===\n")
cat("    sign flips = replications whose contrast has the opposite sign to tau = 0\n\n")
for (gam in GAMMAS) {
  cat(sprintf("gamma = %.1f\n", gam))
  cat(sprintf("%6s | %9s %8s %7s %6s | %9s %7s\n", "tau",
              "LURE", "sd(rep)", "z", "flips", "DRL", "z"))
  base <- con$per_step_diff[con$gamma == gam & con$tau == 0 & con$contrast == "Always IV vs No IV"]
  for (tau in TAUS) {
    s <- con[con$gamma == gam & con$tau == tau & con$contrast == "Always IV vs No IV", ]
    dd <- if (!is.null(drl)) drl$contrasts[drl$contrasts$gamma == gam & drl$contrasts$tau == tau &
                                            drl$contrasts$contrast == "Always IV vs No IV", ] else NULL
    cat(sprintf("%5.0f%% | %9.4f %8.4f %7.2f %4d/%-1d | %9s %7s\n", 100 * tau,
                mean(s$per_step_diff), if (nrow(s) > 1) sd(s$per_step_diff) else 0,
                mean(s$z), sum(sign(s$per_step_diff) != sign(base)), nrow(s),
                if (is.null(dd)) "NA" else formatC(mean(dd$per_step_diff), format = "f", digits = 4),
                if (is.null(dd)) "NA" else formatC(mean(dd$z), format = "f", digits = 2)))
  }
  cat("\n")
}

cat("=== All three contrasts, LURE z by tau (gamma = 0.5), mean over reps ===\n\n")
cat(sprintf("%6s %-28s %10s %8s %8s\n", "tau", "contrast", "diff", "z", "min z"))
for (tau in TAUS) {
  for (cn in unique(con$contrast)) {
    s <- con[con$gamma == 0.5 & con$tau == tau & con$contrast == cn, ]
    cat(sprintf("%5.0f%% %-28s %10.4f %8.2f %8.2f\n", 100 * tau, cn,
                mean(s$per_step_diff), mean(s$z), min(s$z)))
  }
  cat("\n")
}

cat("Saved to", out_file, "\n")
