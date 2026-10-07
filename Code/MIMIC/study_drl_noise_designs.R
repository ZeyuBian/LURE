#!/usr/bin/env Rscript

# DRL at gamma = 0.6: sensitivity of genuinely doubly robust designs to extra
# misclassification of the observed action.
#
# Production DRL coincides with FQE (matched linear features, so the correction
# cancels) and was insensitive to action noise. This study re-runs the noise grid
# for DRL designs whose correction does NOT cancel, fixed in advance:
#   - DRL = FQE (reference)   linear FQE, linear omega
#   - FQE ridge 1000          FQE shrunk; omega must repair it
#   - FQE ridge 10000         FQE shrunk harder
#   - omega quadratic         omega richer than FQE
#   - FQE SOFA-only           misspecified outcome model; omega must repair it
# The omega-ridge variants are excluded: they do not break the cancellation and
# they deflate the standard error.
#
# Noise: independent symmetric flips of the binary action, one Uniform draw per
# action per replication with seed 20260921 + rep -- identical to the earlier DRL
# and LURE noise studies, so replications are paired across all of them.
# FQE is iterated to convergence (gamma^n_iter < 1e-8).
#
# Usage from Code/MIMIC:
#   env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
#     Rscript study_drl_noise_designs.R

script_path <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])
script_dir <- if (is.na(script_path)) normalizePath(getwd()) else dirname(normalizePath(script_path))
source(file.path(script_dir, "MIMIC.R"))
suppressPackageStartupMessages(library(parallel))

GAMMA <- 0.6
TAUS  <- c(0, 0.05, 0.10, 0.20, 0.30, 0.40)
N_REP <- 5L
N_WORKERS <- as.integer(Sys.getenv("DRL_WORKERS", "6"))
POLICIES <- c(high_dose = "Always IV", sofa_11 = "SOFA-tailored", low_dose = "No IV")
out_file <- file.path(script_dir, "drl_noise_designs.rds")

DESIGNS <- list(
  list(id = "DRL = FQE (ref)", fqe = c("linear", "0.001"),    mis = c("linear", "0.001")),
  list(id = "FQE ridge 1000",  fqe = c("linear", "1000"),     mis = c("linear", "0.001")),
  list(id = "FQE ridge 10000", fqe = c("linear", "10000"),    mis = c("linear", "0.001")),
  list(id = "omega quadratic", fqe = c("linear", "0.001"),    mis = c("quadratic", "0.001")),
  list(id = "FQE SOFA-only",   fqe = c("sofa_only", "0.001"), mis = c("linear", "0.001"))
)
design_ids <- vapply(DESIGNS, `[[`, character(1), "id")

baseline <- new.env()
load(file.path(script_dir, "res.RData"), envir = baseline)
dat <- baseline$results_high$dat
n_traj <- dim(dat$S)[1]; TT <- dim(dat$S)[2]
S <- gym_flatten_states(dat$S); Sp <- gym_flatten_states(dat$Sp)
R <- as.vector(dat$R); S0 <- dat$init_states; nm <- dat$state_names
A0_mat <- dat$Atilde

n_unique <- apply(S, 2, function(x) length(unique(x)))
quad_cols <- which(n_unique > 2)
FEATURES <- list(
  linear    = function(X) X,
  quadratic = function(X) cbind(X, X[, quad_cols, drop = FALSE]^2),
  sofa_only = function(X) X[, match("SOFA", nm), drop = FALSE]
)
PI <- lapply(names(POLICIES), function(p) mimic_fit_target_policy(dat, policy_type = p)$pi_func)
names(PI) <- names(POLICIES)

fqe_fit <- function(pi_func, phi, lambda, A) {
  n_iter <- ceiling(log(1e-8) / log(GAMMA))
  FS <- phi(S); FSp <- phi(Sp); pi_sp <- pi_func(Sp)
  i0 <- which(A == 0L); i1 <- which(A == 1L)
  q0 <- q1 <- rep(0, nrow(S))
  for (it in seq_len(n_iter)) {
    y <- R + GAMMA * ((1 - pi_sp) * q0 + pi_sp * q1)
    f0 <- gym_weighted_ridge_fit(FS[i0, , drop = FALSE], y[i0], ridge = lambda)
    f1 <- gym_weighted_ridge_fit(FS[i1, , drop = FALSE], y[i1], ridge = lambda)
    q0 <- f0$predict(FSp); q1 <- f1$predict(FSp)
  }
  Q <- function(X, a) if (a == 0) f0$predict(phi(X)) else f1$predict(phi(X))
  V <- function(X) { p <- pi_func(X); (1 - p) * Q(X, 0) + p * Q(X, 1) }
  list(Q = Q, V = V)
}

mis_fit <- function(pi_func, phi, lambda, A) {
  FS <- phi(S); mu <- colMeans(FS)
  sdv <- sqrt(colMeans(sweep(FS, 2, mu)^2)); sdv[!is.finite(sdv) | sdv < 1e-8] <- 1
  z <- function(X) cbind(1, sweep(sweep(phi(X), 2, mu), 2, sdv, "/"))
  X <- z(S); Xp <- z(Sp); X0 <- z(S0)
  phi_obs <- gym_action_feature_matrix(X, A)
  phi_pi_sp <- gym_policy_feature_matrix(Xp, pi_func(Sp))
  a_mat <- crossprod(phi_obs - GAMMA * phi_pi_sp, phi_obs) / nrow(S)
  b_vec <- (1 - GAMMA) * colMeans(gym_policy_feature_matrix(X0, pi_func(S0)))
  beta <- solve(a_mat + lambda * diag(ncol(phi_obs)), b_vec)
  list(omega = drop(phi_obs %*% beta))
}

perturb <- function(rep_id, tau) {
  if (tau == 0) return(as.vector(A0_mat))
  set.seed(20260921L + rep_id)          # identical to the earlier noise studies
  u <- matrix(runif(length(A0_mat)), nrow = nrow(A0_mat), ncol = ncol(A0_mat))
  A <- A0_mat; flip <- u < tau; A[flip] <- 1L - A[flip]
  as.vector(A)
}

run_one <- function(job) {
  A <- perturb(job$rep, job$tau)
  rows <- list(); psi <- list()
  for (pol in names(POLICIES)) {
    fq <- list(); ms <- list()
    for (d in DESIGNS) {
      fk <- paste(d$fqe, collapse = "|"); mk <- paste(d$mis, collapse = "|")
      if (is.null(fq[[fk]])) fq[[fk]] <- fqe_fit(PI[[pol]], FEATURES[[d$fqe[1]]], as.numeric(d$fqe[2]), A)
      if (is.null(ms[[mk]])) ms[[mk]] <- mis_fit(PI[[pol]], FEATURES[[d$mis[1]]], as.numeric(d$mis[2]), A)
      f <- fq[[fk]]; m <- ms[[mk]]
      q_obs <- ifelse(A == 0L, f$Q(S, 0), f$Q(S, 1))
      delta <- R + GAMMA * f$V(Sp) - q_obs
      direct_terms <- f$V(S0)
      corr_terms <- rowMeans(matrix(m$omega * delta / (1 - GAMMA), nrow = n_traj, ncol = TT))
      p <- direct_terms + corr_terms
      V <- mean(p); se <- stats::sd(p) / sqrt(n_traj)
      psi[[d$id]][[pol]] <- p
      rows[[length(rows) + 1L]] <- data.frame(
        rep = job$rep, tau = job$tau, realized = mean(A != as.vector(A0_mat)),
        design = d$id, policy = pol, V = V, se = se,
        ci_lo = V - 1.96 * se, ci_hi = V + 1.96 * se,
        direct = mean(direct_terms), correction = mean(corr_terms),
        stringsAsFactors = FALSE
      )
    }
  }
  list(rows = do.call(rbind, rows), psi = psi, rep = job$rep, tau = job$tau)
}

jobs <- c(list(list(rep = 0L, tau = 0)),
          lapply(seq_len(N_REP * (length(TAUS) - 1L)), function(i) {
            g <- expand.grid(rep = seq_len(N_REP), tau = TAUS[TAUS > 0])
            list(rep = g$rep[i], tau = g$tau[i])
          }))
cat(sprintf("gamma = %.1f | %d designs | %d fits (%d workers)\n\n",
            GAMMA, length(DESIGNS), length(jobs), N_WORKERS))
flush.console()
t0 <- proc.time()[["elapsed"]]
out <- mclapply(jobs, run_one, mc.cores = N_WORKERS, mc.preschedule = FALSE)
bad <- vapply(out, function(o) inherits(o, "try-error") || is.null(o$rows), logical(1))
if (any(bad)) stop(sum(bad), " jobs failed: ", paste(unique(unlist(out[bad])), collapse = "; "))
cat(sprintf("All fits done in %.0fs\n", proc.time()[["elapsed"]] - t0))

vals <- do.call(rbind, lapply(out, `[[`, "rows"))
con <- do.call(rbind, lapply(out, function(o) {
  do.call(rbind, lapply(design_ids, function(id) {
    d <- o$psi[[id]][["low_dose"]] - o$psi[[id]][["high_dose"]]
    data.frame(rep = o$rep, tau = o$tau, design = id, diff = mean(d),
               z = mean(d) / (stats::sd(d) / sqrt(n_traj)), stringsAsFactors = FALSE)
  }))
}))
saveRDS(list(values = vals, contrasts = con, designs = DESIGNS, gamma = GAMMA), out_file)

m <- function(x) mean(x)
for (id in design_ids) {
  cat(sprintf("\n=== %s  (gamma = %.1f) — values and 95%% CIs, mean over replications ===\n\n", id, GAMMA))
  cat(sprintf("%6s | %-24s | %-24s | %-24s\n", "tau", "Always IV", "SOFA-tailored", "No IV"))
  for (tau in TAUS) {
    cells <- vapply(names(POLICIES), function(pol) {
      s <- vals[vals$design == id & vals$tau == tau & vals$policy == pol, ]
      sprintf("%6.2f [%6.2f, %6.2f]", m(s$V), m(s$ci_lo), m(s$ci_hi))
    }, character(1))
    cat(sprintf("%5.0f%% | %-24s | %-24s | %-24s\n", 100 * tau, cells[1], cells[2], cells[3]))
  }
}

cat("\n\n=== Sensitivity summary: shift of the value from tau = 0, in units of the tau = 0 SE ===\n")
cat("    shift = mean over replications of V(tau) - V(0);  rep sd = spread across noise draws\n\n")
cat(sprintf("%-16s %-14s %8s %6s | %7s %7s %7s %7s %7s | %9s %9s\n",
            "design", "policy", "V(0)", "SE(0)", "5%", "10%", "20%", "30%", "40%",
            "max|sh|/SE", "rep sd/SE"))
for (id in design_ids) {
  for (pol in names(POLICIES)) {
    s0 <- vals[vals$design == id & vals$tau == 0 & vals$policy == pol, ]
    sh <- vapply(TAUS[-1], function(tau) {
      m(vals$V[vals$design == id & vals$tau == tau & vals$policy == pol]) - s0$V
    }, numeric(1))
    rsd <- max(vapply(TAUS[-1], function(tau) {
      stats::sd(vals$V[vals$design == id & vals$tau == tau & vals$policy == pol])
    }, numeric(1)))
    cat(sprintf("%-16s %-14s %8.3f %6.3f | %+7.3f %+7.3f %+7.3f %+7.3f %+7.3f | %9.2f %9.2f\n",
                id, POLICIES[[pol]], s0$V, s0$se, sh[1], sh[2], sh[3], sh[4], sh[5],
                max(abs(sh)) / s0$se, rsd / s0$se))
  }
}

cat("\n\n=== Where the shift comes from: direct vs correction, change from tau = 0 ===\n\n")
cat(sprintf("%-16s %-14s %6s %10s %12s %10s\n", "design", "policy", "tau", "d(direct)", "d(correction)", "d(V)"))
for (id in design_ids) {
  for (pol in c("high_dose", "low_dose")) {
    s0 <- vals[vals$design == id & vals$tau == 0 & vals$policy == pol, ]
    for (tau in c(0.2, 0.4)) {
      s <- vals[vals$design == id & vals$tau == tau & vals$policy == pol, ]
      cat(sprintf("%-16s %-14s %5.0f%% %+10.3f %+12.3f %+10.3f\n", id, POLICIES[[pol]], 100 * tau,
                  m(s$direct) - s0$direct, m(s$correction) - s0$correction, m(s$V) - s0$V))
    }
  }
}

cat("\n\n=== Always IV vs No IV contrast (No IV - Always IV, raw scale) by tau ===\n\n")
cat(sprintf("%-16s | %s\n", "design", paste(sprintf("%12s", paste0(100 * TAUS, "%")), collapse = "")))
for (id in design_ids) {
  cells <- vapply(TAUS, function(tau) {
    s <- con[con$design == id & con$tau == tau, ]
    sprintf("%6.3f(%4.1f)", m(s$diff), m(s$z))
  }, character(1))
  cat(sprintf("%-16s | %s\n", id, paste(sprintf("%12s", cells), collapse = "")))
}
cat("    entries: mean contrast (mean z)\n")
cat("\nSaved to", out_file, "\n")
