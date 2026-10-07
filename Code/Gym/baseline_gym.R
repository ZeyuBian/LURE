################################################################################
## Baselines for Gym Environments
################################################################################

GYM_BASELINE_SPLINE_DF <- 6L
GYM_BASELINE_SPLINE_DEGREE <- 3L

gym_baseline_spline_features <- function(state_mat, specs = NULL,
                                         df = GYM_BASELINE_SPLINE_DF,
                                         degree = GYM_BASELINE_SPLINE_DEGREE,
                                         include_intercept = TRUE) {
  if (!requireNamespace("splines", quietly = TRUE)) {
    stop("Package 'splines' is required for spline-based Gym baselines.")
  }

  state_mat <- as.matrix(state_mat)
  n <- nrow(state_mat)
  d <- ncol(state_mat)

  if (is.null(specs)) {
    specs <- vector("list", d)
  }

  feat_list <- vector("list", d + as.integer(include_intercept))
  idx <- 1L

  if (include_intercept) {
    feat_list[[idx]] <- rep(1, n)
    idx <- idx + 1L
  }

  for (j in seq_len(d)) {
    x <- as.numeric(state_mat[, j])

    if (is.null(specs[[j]])) {
      boundary_knots <- range(x, na.rm = TRUE)
      if (!all(is.finite(boundary_knots)) || diff(boundary_knots) < 1e-8) {
        basis_j <- matrix(0, nrow = n, ncol = 1L)
        specs[[j]] <- list(type = "constant_zero")
      } else {
        basis_j <- splines::bs(
          x,
          df = df,
          degree = degree,
          intercept = FALSE,
          Boundary.knots = boundary_knots,
          warn.outside = FALSE
        )
        specs[[j]] <- list(
          type = "bs",
          knots = attr(basis_j, "knots"),
          Boundary.knots = attr(basis_j, "Boundary.knots"),
          degree = attr(basis_j, "degree"),
          intercept = FALSE
        )
      }
    } else if (identical(specs[[j]]$type, "constant_zero")) {
      basis_j <- matrix(0, nrow = n, ncol = 1L)
    } else {
      basis_j <- splines::bs(
        x,
        knots = specs[[j]]$knots,
        degree = specs[[j]]$degree,
        intercept = specs[[j]]$intercept,
        Boundary.knots = specs[[j]]$Boundary.knots,
        warn.outside = FALSE
      )
    }

    feat_list[[idx]] <- basis_j
    idx <- idx + 1L
  }

  list(x_mat = do.call(cbind, feat_list), specs = specs)
}

gym_baseline_action_feature_matrix <- function(basis_mat, action_vec) {
  action_vec <- as.integer(action_vec)
  k <- ncol(basis_mat)
  out <- matrix(0, nrow(basis_mat), 2L * k)
  idx0 <- which(action_vec == 0L)
  idx1 <- which(action_vec == 1L)
  if (length(idx0) > 0L) {
    out[idx0, seq_len(k)] <- basis_mat[idx0, , drop = FALSE]
  }
  if (length(idx1) > 0L) {
    out[idx1, k + seq_len(k)] <- basis_mat[idx1, , drop = FALSE]
  }
  out
}

gym_baseline_policy_feature_matrix <- function(basis_mat, pi_s) {
  k <- ncol(basis_mat)
  out <- matrix(0, nrow(basis_mat), 2L * k)
  out[, seq_len(k)] <- (1 - pi_s) * basis_mat
  out[, k + seq_len(k)] <- pi_s * basis_mat
  out
}

gym_poly_features <- function(state_mat, include_intercept = TRUE,
                              specs = NULL,
                              spline_df = GYM_BASELINE_SPLINE_DF,
                              spline_degree = GYM_BASELINE_SPLINE_DEGREE,
                              return_specs = FALSE) {
  feat_fit <- gym_baseline_spline_features(
    state_mat,
    specs = specs,
    df = spline_df,
    degree = spline_degree,
    include_intercept = include_intercept
  )
  if (return_specs) {
    return(feat_fit)
  }
  feat_fit$x_mat
}

gym_poly_action_features <- function(state_mat, action_vec, specs = NULL,
                                     spline_df = GYM_BASELINE_SPLINE_DF,
                                     spline_degree = GYM_BASELINE_SPLINE_DEGREE,
                                     return_specs = FALSE) {
  feat_fit <- gym_poly_features(
    state_mat,
    specs = specs,
    spline_df = spline_df,
    spline_degree = spline_degree,
    return_specs = TRUE
  )
  out <- gym_baseline_action_feature_matrix(feat_fit$x_mat, action_vec)
  if (return_specs) {
    return(list(x_mat = out, specs = feat_fit$specs))
  }
  out
}

gym_poly_policy_features <- function(state_mat, pi_func, specs = NULL,
                                     spline_df = GYM_BASELINE_SPLINE_DF,
                                     spline_degree = GYM_BASELINE_SPLINE_DEGREE,
                                     return_specs = FALSE) {
  feat_fit <- gym_poly_features(
    state_mat,
    specs = specs,
    spline_df = spline_df,
    spline_degree = spline_degree,
    return_specs = TRUE
  )
  out <- gym_baseline_policy_feature_matrix(feat_fit$x_mat, pi_func(state_mat))
  if (return_specs) {
    return(list(x_mat = out, specs = feat_fit$specs))
  }
  out
}

naive_fqe_gym <- function(dat, dgp, gamma, n_iter = 20, ridge = 0) {
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

naive_mis_gym <- function(dat, dgp, gamma, ridge = .0001) {
  S_mat <- gym_flatten_states(dat$S)
  Sp_mat <- gym_flatten_states(dat$Sp)
  At_vec <- as.vector(dat$Atilde)
  R_vec <- as.vector(dat$R)
  n <- nrow(S_mat)

  feat_fit <- gym_poly_features(S_mat, return_specs = TRUE)
  basis_specs <- feat_fit$specs
  phi_obs <- gym_baseline_action_feature_matrix(feat_fit$x_mat, At_vec)
  phi_pi_sp <- gym_poly_policy_features(Sp_mat, dgp$pi_func, specs = basis_specs)
  diff_mat <- phi_obs - gamma * phi_pi_sp
  a_mat <- crossprod(diff_mat, phi_obs) / n

  init_states <- gym_draw_initial_states(dgp, 5000)
  b_vec <- (1 - gamma) * colMeans(
    gym_poly_policy_features(init_states, dgp$pi_func, specs = basis_specs)
  )

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

naive_sis_gym <- function(dat, dgp, gamma,
                          min_prob = 0.01, max_ratio = 10,
                          max_weight = 20, normalize = T,
                          behavior_model = c("state", "intercept")) {
  n_traj <- dim(dat$S)[1]
  TT <- dim(dat$S)[2]
  state_dim <- dim(dat$S)[3]
  S_mat <- gym_flatten_states(dat$S)
  At_vec <- as.vector(dat$Atilde)
  At_mat <- as.matrix(dat$Atilde)
  R_mat <- as.matrix(dat$R)
  behavior_model <- match.arg(behavior_model)

  feature_df <- gym_model_feature_df(S_mat)
  fit_b <- if (behavior_model == "intercept") {
    glm(a ~ 1, family = binomial(), data = data.frame(a = At_vec))
  } else {
    glm(gym_model_formula("a", ncol(feature_df)), family = binomial(),
        data = cbind(data.frame(a = At_vec), feature_df))
  }

  b_prob <- matrix(
    gym_clip(
      predict(
        fit_b,
        newdata = if (behavior_model == "intercept") {
          data.frame(a = At_vec)
        } else {
          feature_df
        },
        type = "response"
      ),
      min_prob,
      1 - min_prob
    ),
    nrow = n_traj,
    ncol = TT
  )

  cum_w <- matrix(1, nrow = n_traj, ncol = TT)
  for (t in seq_len(TT)) {
    state_t <- dat$S[, t, , drop = FALSE]
    dim(state_t) <- c(n_traj, state_dim)

    p1_t <- dgp$pi_func(state_t)
    pi_a <- ifelse(At_mat[, t] == 1L, p1_t, 1 - p1_t)
    b_a <- ifelse(At_mat[, t] == 1L, b_prob[, t], 1 - b_prob[, t])

    step_ratio <- pi_a / b_a
    step_ratio[!is.finite(step_ratio)] <- max_ratio
    step_ratio <- pmin(step_ratio, max_ratio)

    if (t == 1L) {
      cum_w[, t] <- pmin(step_ratio, max_weight)
    } else {
      cum_w[, t] <- pmin(cum_w[, t - 1L] * step_ratio, max_weight)
    }
  }

  gamma_t <- gamma^(seq_len(TT) - 1L)

  if (normalize) {
    est_t <- numeric(TT)
    for (t in seq_len(TT)) {
      denom <- sum(cum_w[, t])
      if (!is.finite(denom) || denom <= 0) {
        next
      }
      est_t[t] <- sum(cum_w[, t] * R_mat[, t]) / denom
    }

    return(list(
      V_hat = sum(gamma_t * est_t),
      fit_b = fit_b,
      behavior_model = behavior_model,
      cum_w = cum_w
    ))
  }

  weighted_rewards <- cum_w *
    matrix(gamma_t, nrow = n_traj, ncol = TT, byrow = TRUE) * R_mat

  list(
    V_hat = sum(weighted_rewards) / n_traj,
    fit_b = fit_b,
    behavior_model = behavior_model,
    cum_w = cum_w
  )
}

naive_lstd_gym <- function(dat, dgp, gamma, ridge = .00012) {
  S_mat <- gym_flatten_states(dat$S)
  Sp_mat <- gym_flatten_states(dat$Sp)
  At_vec <- as.vector(dat$Atilde)
  R_vec <- as.vector(dat$R)
  n <- nrow(S_mat)

  feat_fit <- gym_poly_features(S_mat, return_specs = TRUE)
  basis_specs <- feat_fit$specs
  phi_obs <- gym_baseline_action_feature_matrix(feat_fit$x_mat, At_vec)
  phi_pi_sp <- gym_poly_policy_features(Sp_mat, dgp$pi_func, specs = basis_specs)
  a_mat <- crossprod(phi_obs, phi_obs - gamma * phi_pi_sp) / n
  b_vec <- drop(crossprod(phi_obs, R_vec)) / n

  w_hat <- solve(a_mat + ridge * diag(ncol(phi_obs)), b_vec)
  init_states <- gym_draw_initial_states(dgp, 5000)
  V_hat <- mean(drop(
    gym_poly_policy_features(init_states, dgp$pi_func, specs = basis_specs) %*% w_hat
  ))

  list(V_hat = V_hat, w = w_hat)
}