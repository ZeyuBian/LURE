################################################################################
## Posterior-imputation (IMP) baseline — Gym (CartPole)
##
## Requires Methods_gym.R, baseline_gym.R, and ../posterior_imputation.R.
## Same construction as the Continuous version, with the additive cubic
## B-spline state features used by every Gym nuisance model:
##   mu(Atilde | s, a): pointwise continuous-outcome eigendecomposition of
##                      {E(R | Atilde, Z, s) (.) P(Atilde, Z | s)} P(Atilde, Z | s)^{-1},
##                      Z = 1{S'_j > E[S'_j | S]}, j the LURE proxy coordinate.
##   b(a | s):          closed form from the proof of Theorem 1.
##   h, q:              Gaussian reward and transition models from EM.
################################################################################

.imp_helper_candidates <- c(
  if (!is.null(getOption("lure.gym.dir"))) file.path(getOption("lure.gym.dir"), "..", "posterior_imputation.R"),
  "../posterior_imputation.R", "posterior_imputation.R", "Code/posterior_imputation.R")
.imp_helper_path <- .imp_helper_candidates[file.exists(.imp_helper_candidates)][1]
if (is.na(.imp_helper_path)) stop("Cannot locate Code/posterior_imputation.R.")
source(.imp_helper_path)
rm(.imp_helper_candidates, .imp_helper_path)

imp_measurement_gym <- function(S_mat, At, R, Sp_mat, bridge_index) {
  feature_df <- gym_model_feature_df(S_mat)
  n_feat <- ncol(feature_df)
  Sp <- Sp_mat[, bridge_index]

  fit_z <- stats::lm(gym_model_formula("sp", n_feat), data = cbind(data.frame(sp = Sp), feature_df))
  Z <- as.integer(Sp > stats::fitted(fit_z))

  cells <- imp_fit_cell_model(feature_df, At, Z, R)
  cp <- cells$predict_cells(feature_df)
  eig <- imp_eigen_measurement(cp$p_az, cp$g_az)

  fit_at <- stats::glm(gym_model_formula("at", n_feat), family = stats::binomial(),
                       data = cbind(data.frame(at = At), feature_df))
  p_at1 <- stats::fitted(fit_at)
  beh <- imp_behavior_closed_form(p_at1, eig$mu0, eig$mu1)

  list(eig = eig, beh = beh, p_at1 = p_at1,
       cell_model_converged = cells$converged)
}

imp_posterior_gym <- function(dat, dgp, gamma, em = NULL) {
  S_mat <- gym_flatten_states(dat$S)
  Sp_mat <- gym_flatten_states(dat$Sp)
  At <- as.vector(dat$Atilde)
  R <- as.vector(dat$R)

  bridge_index <- if (!is.null(dgp$bridge_index)) as.integer(dgp$bridge_index)[1] else
    select_bridge_index_gym(dat)$bridge_index

  meas <- imp_measurement_gym(S_mat, At, R, Sp_mat, bridge_index)
  if (is.null(em)) em <- em_gym(dat, gamma)

  log_h <- log_q <- vector("list", 2L)
  for (a in 0:1) {
    log_h[[a + 1]] <- stats::dnorm(R, em$predict_theta_R(S_mat, a), em$sigma_R[a + 1], log = TRUE)
    tSp <- em$predict_theta_Sp(S_mat, a)
    log_q[[a + 1]] <- rowSums(vapply(seq_len(ncol(Sp_mat)), function(j) {
      stats::dnorm(Sp_mat[, j], tSp[, j], em$sigma_Sp[a + 1, j], log = TRUE)
    }, numeric(nrow(Sp_mat))))
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

## Standard OPE estimators from baseline_gym.R applied to one completed data
## set, with the same tuning as the surrogate-as-truth baselines.
imp_standard_ope_gym <- function(dat, A_imp_vec, dgp, gamma) {
  dat_imp <- dat
  dat_imp$Atilde <- matrix(as.integer(A_imp_vec), nrow(dat$Atilde), ncol(dat$Atilde))
  drl <- tryCatch(naive_drl_gym(dat_imp, dgp, gamma),
                  error = function(e) list(V_hat = NA_real_, V_fqe = NA_real_,
                                           V_mis = NA_real_))
  V_sis <- tryCatch(naive_sis_gym(dat_imp, dgp, gamma)$V_hat, error = function(e) NA_real_)
  V_lstd <- tryCatch(naive_lstd_gym(dat_imp, dgp, gamma)$V_hat, error = function(e) NA_real_)
  c(FQE = drl$V_fqe, SIS = V_sis, MIS = drl$V_mis, DRL = drl$V_hat, LSTD = V_lstd)
}

## dgp must carry init_states, as in evaluate_gym_estimators().
imp_estimator_gym <- function(dat, dgp, gamma, M = 20L, em = NULL) {
  dgp$init_states <- dat$init_states
  post <- imp_posterior_gym(dat, dgp, gamma, em = em)
  A_draws <- imp_draw_actions(post$eta1, M)
  est_list <- lapply(seq_len(M), function(m) {
    imp_standard_ope_gym(dat, A_draws[, m], dgp, gamma)
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
