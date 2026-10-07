################################################################################
## Posterior-imputation (IMP) baseline — Tabular MDP
##
## Requires Methods.R (EM, FQE, MIS, DRL, SIS, LSTD) and ../posterior_imputation.R.
##   mu(Atilde | s, a): continuous-outcome eigendecomposition within each state
##                      (Zhou and Tchetgen Tchetgen, 2024, Supplement S2) with
##                      outcome R and proxy Z = 1{S' > mean(S' | s)}, i.e.
##                      l(s) = s as in LURE; cell probabilities and cell means
##                      of R are empirical frequencies and averages.
##   b(a | s):          closed form from the proof of Theorem 1.
##   h, q:              Gaussian reward model and transition matrix from EM.
################################################################################

.imp_helper_candidates <- c("../posterior_imputation.R", "posterior_imputation.R",
                            "Code/posterior_imputation.R")
.imp_helper_path <- .imp_helper_candidates[file.exists(.imp_helper_candidates)][1]
if (is.na(.imp_helper_path)) stop("Cannot locate Code/posterior_imputation.R.")
source(.imp_helper_path)
rm(.imp_helper_candidates, .imp_helper_path)

imp_measurement_tabular <- function(S, At, R, Sp, nS) {
  p_az <- g_az <- matrix(NA_real_, nS, 4L)
  p_at1 <- numeric(nS)
  for (s in seq_len(nS)) {
    idx <- which(S == s)
    if (length(idx) == 0L) stop("State ", s, " is not observed.")
    Z <- as.integer(Sp[idx] > mean(Sp[idx]))
    cell <- imp_az_cell(At[idx], Z)
    for (k in 1:4) {
      in_cell <- cell == k
      p_az[s, k] <- mean(in_cell)
      g_az[s, k] <- if (any(in_cell)) mean(R[idx][in_cell]) * p_az[s, k] else 0
    }
    p_at1[s] <- mean(At[idx])
  }
  eig <- imp_eigen_measurement(p_az, g_az)
  beh <- imp_behavior_closed_form(p_at1, eig$mu0, eig$mu1)
  list(eig = eig, beh = beh, p_at1 = p_at1, p_az = p_az, g_az = g_az)
}

imp_posterior_tabular <- function(dat, nS, gamma, em = NULL) {
  S  <- as.vector(dat$S)
  At <- as.vector(dat$Atilde)
  R  <- as.vector(dat$R)
  Sp <- as.vector(dat$Sprime)

  meas <- imp_measurement_tabular(S, At, R, Sp, nS)
  if (is.null(em)) em <- em_tabular(dat, nS, gamma)

  mu0 <- meas$eig$mu0[S]; mu1 <- meas$eig$mu1[S]
  b1 <- meas$beh$b1[S]
  eta1 <- imp_posterior(
    log_b1 = log(b1), log_b0 = log(1 - b1),
    log_mu1 = stats::dbinom(At, 1, mu1, log = TRUE),
    log_mu0 = stats::dbinom(At, 1, mu0, log = TRUE),
    log_h1 = stats::dnorm(R, em$theta_R_hat[cbind(S, 2L)], em$sigma_R_hat[cbind(S, 2L)], log = TRUE),
    log_h0 = stats::dnorm(R, em$theta_R_hat[cbind(S, 1L)], em$sigma_R_hat[cbind(S, 1L)], log = TRUE),
    log_q1 = log(pmax(em$P_hat[cbind(S, Sp, 2L)], 1e-10)),
    log_q0 = log(pmax(em$P_hat[cbind(S, Sp, 1L)], 1e-10))
  )

  list(eta1 = eta1, meas = meas, em = em,
       state_mu = cbind(mu0 = meas$eig$mu0, mu1 = meas$eig$mu1,
                        thetaR0 = meas$eig$theta0, thetaR1 = meas$eig$theta1),
       state_b1 = meas$beh$b1)
}

## Standard OPE estimators from Methods.R applied to one completed data set,
## with the same tuning as the surrogate-as-truth baselines in one_rep().
imp_standard_ope_tabular <- function(dat, A_imp_vec, dgp, gamma) {
  N <- nrow(dat$S); TT <- ncol(dat$S); nS <- dgp$nS; pi <- dgp$pi_policy
  pi1_func <- function(S_df) {
    s <- if (is.data.frame(S_df)) S_df[[1]] else S_df
    pi[s]
  }
  phi_tab <- function(S_df, A) {
    s <- if (is.data.frame(S_df)) S_df[[1]] else S_df
    Phi <- matrix(0, length(s), nS * 2)
    Phi[cbind(seq_along(s), A * nS + s)] <- 1
    Phi
  }
  ## as.vector() of an N x T matrix is the time-major stacking used by Methods.R.
  S_df <- data.frame(S = as.vector(dat$S))
  R_vec <- as.vector(dat$R)
  A_mat <- matrix(A_imp_vec, N, TT)

  V_sis <- tryCatch(SIS(S_mat = dat$S, A_mat = A_mat, R_mat = dat$R,
                        pi1 = pi1_func, gamma = gamma),
                    error = function(e) NA_real_)
  drl <- tryCatch(DRL(S = S_df, A = A_imp_vec, R = R_vec, H = TT,
                      pi1 = pi1_func, phi = phi_tab, gamma = gamma,
                      fqe_iter = 30, mis_ridge = 0.001),
                  error = function(e) list(Vhat_FQE = NA_real_, Vhat_MIS = NA_real_,
                                           Vhat_DRL = NA_real_))
  V_lstd <- tryCatch(LSTD(S = S_df, A = A_imp_vec, R = R_vec, H = TT,
                          pi1 = pi1_func, gamma = gamma)$V_hat,
                     error = function(e) NA_real_)
  c(FQE = drl$Vhat_FQE, SIS = V_sis, MIS = drl$Vhat_MIS,
    DRL = drl$Vhat_DRL, LSTD = V_lstd)
}

imp_estimator_tabular <- function(dat, dgp, gamma, M = 20L, em = NULL) {
  post <- imp_posterior_tabular(dat, dgp$nS, gamma, em = em)
  A_draws <- imp_draw_actions(post$eta1, M)
  est_list <- lapply(seq_len(M), function(m) {
    imp_standard_ope_tabular(dat, A_draws[, m], dgp, gamma)
  })
  avg <- imp_average_estimates(est_list)

  diag <- imp_posterior_diagnostics(post$eta1, dat$A, post$meas$eig, post$meas$beh)
  diag <- c(diag,
            imputation_accuracy = mean(A_draws == as.vector(dat$A)),
            em_converged = as.numeric(isTRUE(post$em$converged)),
            em_iterations = post$em$n_iter)

  list(estimates = stats::setNames(avg$mean, paste0("IMP-", names(avg$mean))),
       between_sd = stats::setNames(avg$between_sd, paste0("IMP-", names(avg$between_sd))),
       diagnostics = diag,
       state_mu = post$state_mu, state_b1 = post$state_b1,
       M = M)
}
