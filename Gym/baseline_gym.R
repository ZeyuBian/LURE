################################################################################
## Baselines for Gym Environments
################################################################################

gym_poly_features <- function(state_mat, include_intercept = TRUE) {
  state_mat <- as.matrix(state_mat)
  n <- nrow(state_mat)
  d <- ncol(state_mat)

  feat_list <- list()
  if (include_intercept) {
    feat_list[[length(feat_list) + 1L]] <- rep(1, n)
  }
  for (j in seq_len(d)) {
    feat_list[[length(feat_list) + 1L]] <- state_mat[, j]
  }
  for (j in seq_len(d)) {
    feat_list[[length(feat_list) + 1L]] <- state_mat[, j]^2
  }
  if (d >= 2L) {
    for (j in 1:(d - 1L)) {
      for (k in (j + 1L):d) {
        feat_list[[length(feat_list) + 1L]] <- state_mat[, j] * state_mat[, k]
      }
    }
  }

  do.call(cbind, feat_list)
}

gym_poly_action_features <- function(state_mat, action_vec) {
  pf <- gym_poly_features(state_mat)
  action_vec <- as.integer(action_vec)
  k <- ncol(pf)
  out <- matrix(0, nrow(pf), 2L * k)
  idx0 <- which(action_vec == 0L)
  idx1 <- which(action_vec == 1L)
  if (length(idx0) > 0L) {
    out[idx0, seq_len(k)] <- pf[idx0, , drop = FALSE]
  }
  if (length(idx1) > 0L) {
    out[idx1, k + seq_len(k)] <- pf[idx1, , drop = FALSE]
  }
  out
}

gym_poly_policy_features <- function(state_mat, pi_func) {
  pf <- gym_poly_features(state_mat)
  pi_s <- pi_func(state_mat)
  k <- ncol(pf)
  out <- matrix(0, nrow(pf), 2L * k)
  out[, seq_len(k)] <- (1 - pi_s) * pf
  out[, k + seq_len(k)] <- pi_s * pf
  out
}

naive_fqe_gym <- function(dat, dgp, gamma, n_iter = 50, ridge = .01) {
  S_mat <- gym_flatten_states(dat$S)
  Sp_mat <- gym_flatten_states(dat$Sp)
  At_vec <- as.vector(dat$Atilde)
  R_vec <- as.vector(dat$R)
  pi_sp <- dgp$pi_func(Sp_mat)

  Q0_sp <- rep(0, nrow(S_mat))
  Q1_sp <- rep(0, nrow(S_mat))
  fit_Q0 <- fit_Q1 <- NULL
  idx0 <- which(At_vec == 0L)
  idx1 <- which(At_vec == 1L)

  for (iter in seq_len(n_iter)) {
    V_sp <- (1 - pi_sp) * Q0_sp + pi_sp * Q1_sp
    Y_target <- R_vec + gamma * V_sp

    fit_Q0 <- gym_weighted_ridge_fit(S_mat[idx0, , drop = FALSE],
                                     Y_target[idx0], ridge = ridge)
    fit_Q1 <- gym_weighted_ridge_fit(S_mat[idx1, , drop = FALSE],
                                     Y_target[idx1], ridge = ridge)

    Q0_sp <- fit_Q0$predict(Sp_mat)
    Q1_sp <- fit_Q1$predict(Sp_mat)
  }

  predict_Q <- function(state_new, a) {
    fit <- if (a == 0) fit_Q0 else fit_Q1
    fit$predict(state_new)
  }

  predict_V <- function(state_new) {
    pi_s <- dgp$pi_func(state_new)
    (1 - pi_s) * predict_Q(state_new, 0) + pi_s * predict_Q(state_new, 1)
  }

  init_states <- gym_draw_initial_states(dgp, 5000)
  V_hat <- mean(predict_V(init_states))

  list(V_hat = V_hat, predict_Q = predict_Q, predict_V = predict_V,
       fit_Q0 = fit_Q0, fit_Q1 = fit_Q1)
}

naive_mis_gym <- function(dat, dgp, gamma, ridge = 0.005) {
  S_mat <- gym_flatten_states(dat$S)
  Sp_mat <- gym_flatten_states(dat$Sp)
  At_vec <- as.vector(dat$Atilde)
  R_vec <- as.vector(dat$R)
  n <- nrow(S_mat)

  phi_obs <- gym_poly_action_features(S_mat, At_vec)
  phi_pi_sp <- gym_poly_policy_features(Sp_mat, dgp$pi_func)
  diff_mat <- phi_obs - gamma * phi_pi_sp
  a_mat <- crossprod(diff_mat, phi_obs) / n

  init_states <- gym_draw_initial_states(dgp, 5000)
  b_vec <- (1 - gamma) * colMeans(gym_poly_policy_features(init_states, dgp$pi_func))

  beta_hat <- solve(a_mat + ridge * diag(ncol(phi_obs)), b_vec)
  omega_hat <- drop(phi_obs %*% beta_hat)
  V_hat <- mean(omega_hat * R_vec) / (1 - gamma)

  list(V_hat = V_hat, omega_hat = omega_hat, beta = beta_hat)
}

naive_drl_gym <- function(dat, dgp, gamma) {
  S_mat <- gym_flatten_states(dat$S)
  Sp_mat <- gym_flatten_states(dat$Sp)
  At_vec <- as.vector(dat$Atilde)
  R_vec <- as.vector(dat$R)

  fqe <- naive_fqe_gym(dat, dgp, gamma)
  mis <- naive_mis_gym(dat, dgp, gamma)

  Q_obs <- ifelse(
    At_vec == 0L,
    fqe$fit_Q0$predict(S_mat),
    fqe$fit_Q1$predict(S_mat)
  )
  V_sp <- fqe$predict_V(Sp_mat)
  dr_inner <- R_vec + gamma * V_sp - Q_obs
  V_hat <- fqe$V_hat + mean(mis$omega_hat * dr_inner) / (1 - gamma)

  list(V_hat = V_hat, V_fqe = fqe$V_hat, V_mis = mis$V_hat)
}

naive_sis_gym <- function(dat, dgp, gamma) {
  n_traj <- dim(dat$S)[1]
  TT <- dim(dat$S)[2]
  S_mat <- gym_flatten_states(dat$S)
  At_vec <- as.vector(dat$Atilde)
  feature_df <- gym_model_feature_df(S_mat)
  fit_b <- glm(gym_model_formula("a", ncol(feature_df)), family = binomial(),
               data = cbind(data.frame(a = At_vec), feature_df))

  b_func <- function(state_new) {
    gym_clip(predict(fit_b, newdata = gym_model_feature_df(state_new), type = "response"),
             0.2, 0.8)
  }

  total <- 0
  for (i in seq_len(n_traj)) {
    w <- 1
    for (t in seq_len(TT)) {
      state_t <- dat$S[i, t, ]
      at <- dat$Atilde[i, t]
      p1 <- dgp$pi_func(matrix(state_t, nrow = 1))
      pi_a <- ifelse(at == 1L, p1, 1 - p1)
      b_a <- b_func(matrix(state_t, nrow = 1))
      b_a <- ifelse(at == 1L, b_a, 1 - b_a)
      w <- w * (pi_a / b_a)
      total <- total + w * gamma^(t - 1L) * dat$R[i, t]
    }
  }

  total / n_traj
}

naive_lstd_gym <- function(dat, dgp, gamma, ridge = 0.001) {
  S_mat <- gym_flatten_states(dat$S)
  Sp_mat <- gym_flatten_states(dat$Sp)
  At_vec <- as.vector(dat$Atilde)
  R_vec <- as.vector(dat$R)
  n <- nrow(S_mat)

  phi_obs <- gym_poly_action_features(S_mat, At_vec)
  phi_pi_sp <- gym_poly_policy_features(Sp_mat, dgp$pi_func)
  a_mat <- crossprod(phi_obs, phi_obs - gamma * phi_pi_sp) / n
  b_vec <- drop(crossprod(phi_obs, R_vec)) / n

  w_hat <- solve(a_mat + ridge * diag(ncol(phi_obs)), b_vec)
  init_states <- gym_draw_initial_states(dgp, 5000)
  V_hat <- mean(drop(gym_poly_policy_features(init_states, dgp$pi_func) %*% w_hat))

  list(V_hat = V_hat, w = w_hat)
}