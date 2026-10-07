################################################################################
## Posterior-imputation (IMP) baseline — Continuous-state MDP
##
## Requires Methods_continuous.R, baseline.R, and ../posterior_imputation.R.
##   mu(Atilde | s, a): pointwise continuous-outcome eigendecomposition of
##                      {E(R | Atilde, Z, s) (.) P(Atilde, Z | s)} P(Atilde, Z | s)^{-1}
##                      (Zhou and Tchetgen Tchetgen, 2024, Supplement S2), where
##                      Z = 1{S'_j > E[S'_j | S]} and j is the LURE proxy
##                      coordinate; P(Atilde, Z | s) is a multinomial logit and
##                      E(R | Atilde, Z, s) a cell-specific regression, both linear in S.
##   b(a | s):          closed form from the proof of Theorem 1, with
##                      P(Atilde = 1 | s) from a logistic regression on S.
##   h, q:              Gaussian reward and transition models from EM.
################################################################################

.imp_helper_candidates <- c("../posterior_imputation.R", "posterior_imputation.R",
                            "Code/posterior_imputation.R")
.imp_helper_path <- .imp_helper_candidates[file.exists(.imp_helper_candidates)][1]
if (is.na(.imp_helper_path)) stop("Cannot locate Code/posterior_imputation.R.")
source(.imp_helper_path)
rm(.imp_helper_candidates, .imp_helper_path)

imp_measurement_continuous <- function(dat, bridge_index) {
  S1 <- as.vector(dat$S1); S2 <- as.vector(dat$S2)
  At <- as.vector(dat$Atilde); R <- as.vector(dat$R)
  Sp <- continuous_select_sp(as.vector(dat$Sp1), as.vector(dat$Sp2), bridge_index)
  df_s <- data.frame(s1 = S1, s2 = S2)

  Z <- as.integer(Sp > stats::fitted(stats::lm(Sp ~ s1 + s2, data = cbind(Sp = Sp, df_s))))

  cells <- imp_fit_cell_model(df_s, At, Z, R)
  cp <- cells$predict_cells(df_s)
  eig <- imp_eigen_measurement(cp$p_az, cp$g_az)

  fit_at <- stats::glm(at ~ s1 + s2, family = stats::binomial(),
                       data = cbind(at = At, df_s))
  p_at1 <- stats::fitted(fit_at)
  beh <- imp_behavior_closed_form(p_at1, eig$mu0, eig$mu1)

  list(eig = eig, beh = beh, p_at1 = p_at1,
       cell_model_converged = cells$converged)
}

imp_posterior_continuous <- function(dat, dgp, gamma, em = NULL) {
  S1 <- as.vector(dat$S1); S2 <- as.vector(dat$S2)
  At <- as.vector(dat$Atilde); R <- as.vector(dat$R)
  Sp1 <- as.vector(dat$Sp1); Sp2 <- as.vector(dat$Sp2)

  bridge <- select_bridge_index_continuous(dat)
  bridge_index <- if (!is.null(dgp$bridge_index)) as.integer(dgp$bridge_index)[1] else
    bridge$bridge_index

  meas <- imp_measurement_continuous(dat, bridge_index)
  if (is.null(em)) em <- em_continuous(dat, gamma)

  log_h <- log_q <- vector("list", 2L)
  for (a in 0:1) {
    log_h[[a + 1]] <- stats::dnorm(R, em$predict_theta_R(S1, S2, a),
                                   em$sigma_R[a + 1], log = TRUE)
    log_q[[a + 1]] <- stats::dnorm(Sp1, em$predict_theta_Sp1(S1, S2, a),
                                   em$sigma_Sp1[a + 1], log = TRUE) +
      stats::dnorm(Sp2, em$predict_theta_Sp2(S1, S2, a),
                   em$sigma_Sp2[a + 1], log = TRUE)
  }
  b1 <- meas$beh$b1
  eta1 <- imp_posterior(
    log_b1 = log(b1), log_b0 = log(1 - b1),
    log_mu1 = stats::dbinom(At, 1, meas$eig$mu1, log = TRUE),
    log_mu0 = stats::dbinom(At, 1, meas$eig$mu0, log = TRUE),
    log_h1 = log_h[[2]], log_h0 = log_h[[1]],
    log_q1 = log_q[[2]], log_q0 = log_q[[1]]
  )

  list(eta1 = eta1, meas = meas, em = em, bridge_index = bridge_index)
}

## Standard OPE estimators from baseline.R applied to one completed data set,
## with the same tuning as the surrogate-as-truth baselines.
imp_standard_ope_continuous <- function(dat, A_imp_vec, dgp, gamma) {
  dat_imp <- dat
  dat_imp$Atilde <- matrix(A_imp_vec, nrow(dat$S1), ncol(dat$S1))
  drl <- tryCatch(naive_drl_continuous(dat_imp, dgp, gamma),
                  error = function(e) list(V_hat = NA_real_, V_fqe = NA_real_,
                                           V_mis = NA_real_))
  V_sis <- tryCatch(naive_sis_continuous(dat_imp, dgp, gamma),
                    error = function(e) NA_real_)
  V_lstd <- tryCatch(naive_lstd_continuous(dat_imp, dgp, gamma)$V_hat,
                     error = function(e) NA_real_)
  c(FQE = drl$V_fqe, SIS = V_sis, MIS = drl$V_mis, DRL = drl$V_hat, LSTD = V_lstd)
}

imp_estimator_continuous <- function(dat, dgp, gamma, M = 20L, em = NULL) {
  post <- imp_posterior_continuous(dat, dgp, gamma, em = em)
  A_draws <- imp_draw_actions(post$eta1, M)
  est_list <- lapply(seq_len(M), function(m) {
    imp_standard_ope_continuous(dat, A_draws[, m], dgp, gamma)
  })
  avg <- imp_average_estimates(est_list)

  diag <- imp_posterior_diagnostics(post$eta1, dat$A, post$meas$eig, post$meas$beh)
  diag <- c(diag,
            imputation_accuracy = mean(A_draws == as.vector(dat$A)),
            cell_model_converged = as.numeric(post$meas$cell_model_converged),
            em_iterations = post$em$n_iter,
            bridge_index = post$bridge_index)

  list(estimates = stats::setNames(avg$mean, paste0("IMP-", names(avg$mean))),
       between_sd = stats::setNames(avg$between_sd, paste0("IMP-", names(avg$between_sd))),
       diagnostics = diag,
       M = M)
}
