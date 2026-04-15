
# ==============================================================================
# 8. Naive FQE (treats surrogate as true action, 2D state)
# ==============================================================================
naive_fqe_continuous <- function(dat, dgp, gamma, n_iter = 50) {
  S1_vec  <- as.vector(dat$S1)
  S2_vec  <- as.vector(dat$S2)
  At_vec  <- as.vector(dat$Atilde)
  R_vec   <- as.vector(dat$R)
  Sp1_vec <- as.vector(dat$Sp1)
  Sp2_vec <- as.vector(dat$Sp2)
  n   <- length(S1_vec)
  pi_func <- dgp$pi_func
  pi_sp <- pi_func(Sp1_vec, Sp2_vec)

  Q0_sp <- rep(0, n);  Q1_sp <- rep(0, n)
  fit_Q0 <- fit_Q1 <- NULL
  idx0 <- which(At_vec == 0);  idx1 <- which(At_vec == 1)

  for (iter in 1:n_iter) {
    V_sp     <- (1 - pi_sp) * Q0_sp + pi_sp * Q1_sp
    Y_target <- R_vec + gamma * V_sp

    fit_Q0 <- lm(y ~ s1 + s2,
                 data = data.frame(y = Y_target[idx0],
                                   s1 = S1_vec[idx0], s2 = S2_vec[idx0]))
    fit_Q1 <- lm(y ~ s1 + s2,
                 data = data.frame(y = Y_target[idx1],
                                   s1 = S1_vec[idx1], s2 = S2_vec[idx1]))

    Q0_sp <- predict(fit_Q0, newdata = data.frame(s1 = Sp1_vec, s2 = Sp2_vec))
    Q1_sp <- predict(fit_Q1, newdata = data.frame(s1 = Sp1_vec, s2 = Sp2_vec))
  }

  predict_Q <- function(s1_new, s2_new, a) {
    fit <- if (a == 0) fit_Q0 else fit_Q1
    predict(fit, newdata = data.frame(s1 = s1_new, s2 = s2_new))
  }
  predict_V <- function(s1_new, s2_new) {
    pi_s <- pi_func(s1_new, s2_new)
    (1 - pi_s) * predict_Q(s1_new, s2_new, 0) +
      pi_s * predict_Q(s1_new, s2_new, 1)
  }

  ## Integrate over the known initial distribution
  S_mc <- draw_initial_states(dgp, 10000)
  S1_mc <- S_mc[, 1];  S2_mc <- S_mc[, 2]
  V_hat <- mean(predict_V(S1_mc, S2_mc))

  list(V_hat = V_hat, predict_Q = predict_Q, predict_V = predict_V,
       fit_Q0 = fit_Q0, fit_Q1 = fit_Q1)
}

# ==============================================================================
# 9. Naive MIS (treats surrogate as true action, 2D state)
# ==============================================================================
naive_mis_continuous <- function(dat, dgp, gamma,
                                 basis_deg = 2, ridge = 1e-6) {
  S1_vec  <- as.vector(dat$S1)
  S2_vec  <- as.vector(dat$S2)
  At_vec  <- as.vector(dat$Atilde)
  R_vec   <- as.vector(dat$R)
  Sp1_vec <- as.vector(dat$Sp1)
  Sp2_vec <- as.vector(dat$Sp2)
  N  <- nrow(dat$S1);  TT <- ncol(dat$S1)
  n  <- N * TT
  pi_func <- dgp$pi_func

  poly_feat_2d <- function(s1, s2) {
    terms <- list()
    for (j in 0:basis_deg) {
      for (k in 0:(basis_deg - j)) {
        terms[[length(terms) + 1]] <- s1^j * s2^k
      }
    }
    do.call(cbind, terms)
  }
  K <- sum(seq_len(basis_deg + 1))
  d <- 2L * K

  phi_mat <- function(s1, s2, a) {
    pf <- poly_feat_2d(s1, s2);  out <- matrix(0, length(s1), d)
    m0 <- (a == 0);  m1 <- (a == 1)
    if (any(m0)) out[m0, 1:K]     <- pf[m0, , drop = FALSE]
    if (any(m1)) out[m1, (K+1):d] <- pf[m1, , drop = FALSE]
    out
  }
  phi_pi <- function(s1, s2) {
    pf   <- poly_feat_2d(s1, s2);  out <- matrix(0, length(s1), d)
    pi_s <- pi_func(s1, s2)
    out[, 1:K]     <- (1 - pi_s) * pf
    out[, (K+1):d] <- pi_s * pf
    out
  }

  Phi_obs  <- phi_mat(S1_vec, S2_vec, At_vec)
  PhiPi_Sp <- phi_pi(Sp1_vec, Sp2_vec)
  diff_mat <- Phi_obs - gamma * PhiPi_Sp
  A_mat <- crossprod(diff_mat, Phi_obs) / n

  ## Integrate over the known initial distribution
  S_mc <- draw_initial_states(dgp, 10000)
  S1_mc <- S_mc[, 1];  S2_mc <- S_mc[, 2]
  b_vec <- (1 - gamma) * colMeans(phi_pi(S1_mc, S2_mc))

  beta_hat <- solve(A_mat + ridge * diag(d), b_vec)

  w_hat <- drop(Phi_obs %*% beta_hat)
  V_hat <- mean(w_hat * R_vec) / (1 - gamma)

  list(V_hat = V_hat, omega_hat = w_hat, beta = beta_hat)
}

# ==============================================================================
# 10. Naive DRL (FQE + MIS, treats surrogate as true action, 2D state)
# ==============================================================================
naive_drl_continuous <- function(dat, dgp, gamma) {
  S1_vec  <- as.vector(dat$S1)
  S2_vec  <- as.vector(dat$S2)
  At_vec  <- as.vector(dat$Atilde)
  R_vec   <- as.vector(dat$R)
  Sp1_vec <- as.vector(dat$Sp1)
  Sp2_vec <- as.vector(dat$Sp2)
  n <- length(S1_vec)

  fqe <- naive_fqe_continuous(dat, dgp, gamma)
  mis <- naive_mis_continuous(dat, dgp, gamma)

  Q_obs <- ifelse(At_vec == 0,
                  predict(fqe$fit_Q0, newdata = data.frame(s1 = S1_vec, s2 = S2_vec)),
                  predict(fqe$fit_Q1, newdata = data.frame(s1 = S1_vec, s2 = S2_vec)))

  V_sp <- fqe$predict_V(Sp1_vec, Sp2_vec)

  dr_inner <- R_vec + gamma * V_sp - Q_obs
  dr_corr  <- mean(mis$omega_hat * dr_inner) / (1 - gamma)

  V_hat <- fqe$V_hat + dr_corr

  list(V_hat = V_hat, V_fqe = fqe$V_hat, V_mis = mis$V_hat)
}

# ==============================================================================
# 11. Naive SIS (treats surrogate as true action, 2D state)
# ==============================================================================
naive_sis_continuous <- function(dat, dgp, gamma,
                                 min_prob = 0.05, max_ratio = 5,
                                 max_weight = 10, normalize = T,
                                 behavior_model = c("state", "intercept")) {
  N  <- nrow(dat$S1);  TT <- ncol(dat$S1)
  pi_func <- dgp$pi_func
  behavior_model <- match.arg(behavior_model)

  S1_mat <- as.matrix(dat$S1)
  S2_mat <- as.matrix(dat$S2)
  At_mat <- as.matrix(dat$Atilde)
  R_mat  <- as.matrix(dat$R)

  S1_vec <- as.vector(S1_mat)
  S2_vec <- as.vector(S2_mat)
  At_vec <- as.vector(At_mat)

  b_df <- data.frame(a = At_vec, s1 = S1_vec, s2 = S2_vec)
  fit_b <- if (behavior_model == "intercept") {
    glm(a ~ 1, family = binomial(), data = b_df)
  } else {
    glm(a ~ s1 + s2, family = binomial(), data = b_df)
  }
  b_prob <- matrix(
    clip(
      predict(fit_b,
              newdata = b_df,
              type = "response"),
      min_prob, 1 - min_prob
    ),
    nrow = N,
    ncol = TT
  )

  cum_w <- matrix(1, nrow = N, ncol = TT)
  for (t in seq_len(TT)) {
    p1_t <- pi_func(S1_mat[, t], S2_mat[, t])
    pi_a <- ifelse(At_mat[, t] == 1, p1_t, 1 - p1_t)
    b_a  <- ifelse(At_mat[, t] == 1, b_prob[, t], 1 - b_prob[, t])

    step_ratio <- pi_a / b_a
    step_ratio[!is.finite(step_ratio)] <- max_ratio
    step_ratio <- pmin(step_ratio, max_ratio)

    if (t == 1) {
      cum_w[, t] <- pmin(step_ratio, max_weight)
    } else {
      cum_w[, t] <- pmin(cum_w[, t - 1] * step_ratio, max_weight)
    }
  }

  gamma_t <- gamma^(seq_len(TT) - 1L)
  weighted_rewards <- cum_w *
    matrix(gamma_t, nrow = N, ncol = TT, byrow = TRUE) * R_mat

  if (normalize) {
    est_t <- numeric(TT)
    for (t in seq_len(TT)) {
      denom <- sum(cum_w[, t])
      if (!is.finite(denom) || denom <= 0) next
      est_t[t] <- sum(cum_w[, t] * R_mat[, t]) / denom
    }
    return(sum(gamma_t * est_t))
  }

  sum(weighted_rewards) / N
}

# ==============================================================================
# 12. Naive LSTD (treats surrogate as true action, 2D state)
# ==============================================================================
naive_lstd_continuous <- function(dat, dgp, gamma,
                                  basis_deg = 2, ridge = 1e-6) {
  S1_vec  <- as.vector(dat$S1)
  S2_vec  <- as.vector(dat$S2)
  At_vec  <- as.vector(dat$Atilde)
  R_vec   <- as.vector(dat$R)
  Sp1_vec <- as.vector(dat$Sp1)
  Sp2_vec <- as.vector(dat$Sp2)
  N  <- nrow(dat$S1);  TT <- ncol(dat$S1)
  n  <- N * TT
  pi_func <- dgp$pi_func

  poly_feat_2d <- function(s1, s2) {
    terms <- list()
    for (j in 0:basis_deg) {
      for (k in 0:(basis_deg - j)) {
        terms[[length(terms) + 1]] <- s1^j * s2^k
      }
    }
    do.call(cbind, terms)
  }
  K <- sum(seq_len(basis_deg + 1))
  d <- 2L * K

  phi_mat <- function(s1, s2, a) {
    pf <- poly_feat_2d(s1, s2);  out <- matrix(0, length(s1), d)
    m0 <- (a == 0);  m1 <- (a == 1)
    if (any(m0)) out[m0, 1:K]     <- pf[m0, , drop = FALSE]
    if (any(m1)) out[m1, (K+1):d] <- pf[m1, , drop = FALSE]
    out
  }
  phi_pi <- function(s1, s2) {
    pf   <- poly_feat_2d(s1, s2);  out <- matrix(0, length(s1), d)
    pi_s <- pi_func(s1, s2)
    out[, 1:K]     <- (1 - pi_s) * pf
    out[, (K+1):d] <- pi_s * pf
    out
  }

  Phi_obs  <- phi_mat(S1_vec, S2_vec, At_vec)
  PhiPi_Sp <- phi_pi(Sp1_vec, Sp2_vec)

  A_mat <- crossprod(Phi_obs, Phi_obs - gamma * PhiPi_Sp) / n
  b_vec <- drop(crossprod(Phi_obs, R_vec)) / n

  w_hat <- solve(A_mat + ridge * diag(d), b_vec)

  ## Integrate over the known initial distribution
  S_mc <- draw_initial_states(dgp, 10000)
  S1_mc <- S_mc[, 1];  S2_mc <- S_mc[, 2]
  V_hat <- mean(drop(phi_pi(S1_mc, S2_mc) %*% w_hat))

  list(V_hat = V_hat, w = w_hat)
}
