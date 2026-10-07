#!/usr/bin/env Rscript

# DRL with different nuisance designs for FQE and omega, so that it no longer
# collapses to FQE.
#
# Why production DRL equals FQE. The correction is mean(omega * delta) / (1 - gamma)
# with delta = R + gamma * V(S') - Q(S, A~). At FQE's fixed point the least-squares
# normal equations make delta orthogonal to FQE's features, and omega is a linear
# combination of those same features, so the correction is identically zero. The
# cancellation needs BOTH (i) delta orthogonal to FQE's features and (ii) omega in
# their span. Breaking either gives a genuine doubly robust correction:
#   - FQE ridge up         -> breaks (i)
#   - omega features richer-> breaks (ii)
#   - FQE features poorer  -> breaks (ii)
#   - omega ridge alone    -> breaks neither; predicted to leave correction ~ 0
#
# FQE is iterated to convergence (residual share gamma^n_iter < 1e-8) in every
# variant, so a nonzero correction reflects the design, not the 20-iteration
# truncation used in production. Features are standardised inside every fit.
#
# Usage from Code/MIMIC:
#   Rscript study_drl_features.R

script_path <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])
script_dir <- if (is.na(script_path)) normalizePath(getwd()) else dirname(normalizePath(script_path))
source(file.path(script_dir, "MIMIC.R"))

GAMMAS <- c(0.5, 0.9)
POLICIES <- c(high_dose = "Always IV", sofa_11 = "SOFA-tailored", low_dose = "No IV")
PAIRS <- list(c("high_dose", "low_dose"), c("high_dose", "sofa_11"), c("sofa_11", "low_dose"))
out_file <- file.path(script_dir, "drl_features.rds")

baseline <- new.env()
load(file.path(script_dir, "res.RData"), envir = baseline)
dat <- baseline$results_high$dat
bridge_index <- as.integer(baseline$results_high$mr_out$bridge_index)
n_traj <- dim(dat$S)[1]; TT <- dim(dat$S)[2]
S <- gym_flatten_states(dat$S); Sp <- gym_flatten_states(dat$Sp)
A <- as.vector(dat$Atilde); R <- as.vector(dat$R); S0 <- dat$init_states
nm <- dat$state_names

# ---- feature maps ----------------------------------------------------------------
n_unique <- apply(S, 2, function(x) length(unique(x)))
quad_cols <- which(n_unique > 2)          # squaring a binary column only duplicates it
FEATURES <- list(
  linear    = function(X) X,
  quadratic = function(X) cbind(X, X[, quad_cols, drop = FALSE]^2),
  sofa_only = function(X) X[, match("SOFA", nm), drop = FALSE]
)

# ---- FQE with an arbitrary feature map (uses the production ridge fitter) --------
fqe_fit <- function(pi_func, gamma, phi, lambda) {
  n_iter <- ceiling(log(1e-8) / log(gamma))
  FS <- phi(S); FSp <- phi(Sp); pi_sp <- pi_func(Sp)
  i0 <- which(A == 0L); i1 <- which(A == 1L)
  q0 <- q1 <- rep(0, nrow(S))
  for (it in seq_len(n_iter)) {
    y <- R + gamma * ((1 - pi_sp) * q0 + pi_sp * q1)
    f0 <- gym_weighted_ridge_fit(FS[i0, , drop = FALSE], y[i0], ridge = lambda)
    f1 <- gym_weighted_ridge_fit(FS[i1, , drop = FALSE], y[i1], ridge = lambda)
    q0 <- f0$predict(FSp); q1 <- f1$predict(FSp)
  }
  Q <- function(X, a) if (a == 0) f0$predict(phi(X)) else f1$predict(phi(X))
  V <- function(X) { p <- pi_func(X); (1 - p) * Q(X, 0) + p * Q(X, 1) }
  list(Q = Q, V = V, n_iter = n_iter)
}

# ---- MIS (marginalised density ratio) with an arbitrary, standardised feature map --
mis_fit <- function(pi_func, gamma, phi, lambda) {
  FS <- phi(S); mu <- colMeans(FS)
  sdv <- sqrt(colMeans(sweep(FS, 2, mu)^2)); sdv[!is.finite(sdv) | sdv < 1e-8] <- 1
  z <- function(X) cbind(1, sweep(sweep(phi(X), 2, mu), 2, sdv, "/"))
  X <- z(S); Xp <- z(Sp); X0 <- z(S0)
  phi_obs <- gym_action_feature_matrix(X, A)
  phi_pi_sp <- gym_policy_feature_matrix(Xp, pi_func(Sp))
  a_mat <- crossprod(phi_obs - gamma * phi_pi_sp, phi_obs) / nrow(S)
  b_vec <- (1 - gamma) * colMeans(gym_policy_feature_matrix(X0, pi_func(S0)))
  beta <- solve(a_mat + lambda * diag(ncol(phi_obs)), b_vec)
  omega <- drop(phi_obs %*% beta)
  list(omega = omega, V_mis = mean(omega * R) / (1 - gamma))
}

# ---- variants: (FQE features, FQE ridge) x (omega features, omega ridge) ----------
VARIANTS <- list(
  list(id = "production-like", fqe = c("linear", 0.001), mis = c("linear", 0.001),
       note = "same features, tiny ridge: expect correction ~ 0"),
  list(id = "omega ridge 0.01", fqe = c("linear", 0.001), mis = c("linear", 0.01),
       note = "omega ridge only: expect correction ~ 0"),
  list(id = "omega ridge 0.1", fqe = c("linear", 0.001), mis = c("linear", 0.1),
       note = "omega ridge only: expect correction ~ 0"),
  list(id = "omega ridge 1", fqe = c("linear", 0.001), mis = c("linear", 1),
       note = "omega ridge only: expect correction ~ 0"),
  list(id = "FQE ridge 100", fqe = c("linear", 100), mis = c("linear", 0.001),
       note = "FQE shrunk: breaks orthogonality"),
  list(id = "FQE ridge 1000", fqe = c("linear", 1000), mis = c("linear", 0.001),
       note = "FQE shrunk: breaks orthogonality"),
  list(id = "FQE ridge 10000", fqe = c("linear", 10000), mis = c("linear", 0.001),
       note = "FQE shrunk: breaks orthogonality"),
  list(id = "omega quadratic", fqe = c("linear", 0.001), mis = c("quadratic", 0.001),
       note = "omega richer than FQE: leaves FQE's span"),
  list(id = "FQE SOFA-only", fqe = c("sofa_only", 0.001), mis = c("linear", 0.001),
       note = "FQE poorer than omega: misspecified outcome model")
)

fqe_cache <- list(); mis_cache <- list()
get_fqe <- function(pol, pf, gam, spec) {
  key <- paste(pol, gam, spec[1], spec[2])
  if (is.null(fqe_cache[[key]])) {
    fqe_cache[[key]] <<- fqe_fit(pf, gam, FEATURES[[spec[1]]], as.numeric(spec[2]))
  }
  fqe_cache[[key]]
}
get_mis <- function(pol, pf, gam, spec) {
  key <- paste(pol, gam, spec[1], spec[2])
  if (is.null(mis_cache[[key]])) {
    mis_cache[[key]] <<- mis_fit(pf, gam, FEATURES[[spec[1]]], as.numeric(spec[2]))
  }
  mis_cache[[key]]
}

rows <- list(); psi_store <- list()
for (v in VARIANTS) {
  for (gam in GAMMAS) {
    t0 <- proc.time()[["elapsed"]]
    for (pol in names(POLICIES)) {
      pf <- mimic_fit_target_policy(dat, policy_type = pol)$pi_func
      fq <- get_fqe(pol, pf, gam, v$fqe)
      ms <- get_mis(pol, pf, gam, v$mis)
      q_obs <- ifelse(A == 0L, fq$Q(S, 0), fq$Q(S, 1))
      delta <- R + gam * fq$V(Sp) - q_obs
      direct_terms <- fq$V(S0)
      corr_terms <- rowMeans(matrix(ms$omega * delta / (1 - gam), nrow = n_traj, ncol = TT))
      psi <- direct_terms + corr_terms
      V <- mean(psi); se <- stats::sd(psi) / sqrt(n_traj)
      psi_store[[v$id]][[sprintf("%.1f", gam)]][[pol]] <- psi
      rows[[length(rows) + 1L]] <- data.frame(
        variant = v$id, gamma = gam, policy = pol, policy_label = unname(POLICIES[[pol]]),
        V = V, se = se, ci_lo = V - 1.96 * se, ci_hi = V + 1.96 * se,
        direct = mean(direct_terms), correction = mean(corr_terms), V_mis = ms$V_mis,
        sd_direct = stats::sd(direct_terms), sd_correction = stats::sd(corr_terms),
        stringsAsFactors = FALSE
      )
    }
    cat(sprintf("  %-17s gamma=%.1f done (%.0fs)\n", v$id, gam, proc.time()[["elapsed"]] - t0))
    flush.console()
  }
}
res <- do.call(rbind, rows)
saveRDS(list(summary = res, psi = psi_store, variants = VARIANTS), out_file)

pz <- function(id, gam, a, b) {
  d <- psi_store[[id]][[sprintf("%.1f", gam)]][[b]] - psi_store[[id]][[sprintf("%.1f", gam)]][[a]]
  c(diff = mean(d), z = mean(d) / (stats::sd(d) / sqrt(n_traj)))
}

for (gam in GAMMAS) {
  cat(sprintf("\n\n================ gamma = %.1f ================\n", gam))
  cat("\n--- Does the correction survive?  (Always IV) ---\n\n")
  cat(sprintf("%-17s %9s %11s %9s %9s %9s  %s\n",
              "variant", "direct", "correction", "DRL", "V_mis", "sd(corr)", "note"))
  for (v in VARIANTS) {
    r <- res[res$variant == v$id & res$gamma == gam & res$policy == "high_dose", ]
    cat(sprintf("%-17s %9.3f %+11.4f %9.3f %9.3f %9.3f  %s\n", v$id,
                r$direct, r$correction, r$V, r$V_mis, r$sd_correction, v$note))
  }

  cat("\n--- DRL values and 95% CIs ---\n\n")
  cat(sprintf("%-17s | %-24s | %-24s | %-24s\n", "variant",
              "Always IV", "SOFA-tailored", "No IV"))
  for (v in VARIANTS) {
    cells <- vapply(names(POLICIES), function(pol) {
      r <- res[res$variant == v$id & res$gamma == gam & res$policy == pol, ]
      sprintf("%6.2f [%6.2f, %6.2f]", r$V, r$ci_lo, r$ci_hi)
    }, character(1))
    cat(sprintf("%-17s | %-24s | %-24s | %-24s\n", v$id, cells[1], cells[2], cells[3]))
  }

  cat("\n--- Paired policy contrasts (positive => first policy better), z ---\n\n")
  cat(sprintf("%-17s %14s %14s %14s\n", "variant", "IV vs NoIV", "IV vs SOFA", "SOFA vs NoIV"))
  for (v in VARIANTS) {
    zs <- vapply(PAIRS, function(pr) pz(v$id, gam, pr[1], pr[2])[["z"]], numeric(1))
    ds <- vapply(PAIRS, function(pr) pz(v$id, gam, pr[1], pr[2])[["diff"]], numeric(1))
    cat(sprintf("%-17s %6.3f (%5.2f) %6.3f (%5.2f) %6.3f (%5.2f)\n", v$id,
                ds[1], zs[1], ds[2], zs[2], ds[3], zs[3]))
  }
}
cat("\nSaved to", out_file, "\n")
