################################################################################
## Continuous-State MDP — "Learning from the Unseen"
##
## 2D state S = (S1, S2), binary action A in {0,1}, hidden actions.
## Linear transition + deterministic reward from next state => Q^pi linear
## when target policy is constant.
## Function approximation via linear regression in (s1, s2).
## Competing (naive) methods: FQE, SIS, MIS, DRL, LSTD — all treat surrogate as true.
## Our method: LURE (multiply robust) with EM-style nuisance estimation.
################################################################################

expit  <- function(x) 1 / (1 + exp(-x))
clip   <- function(x, lo = 1e-2, hi = 1 - 1e-2) pmax(pmin(x, hi), lo)

safe_ratio <- function(num, denom, lambda = 0.005) {
  num * denom / (denom^2 + lambda)
}

clip_abs_quantile <- function(x, q = 0.97) {
  cap <- as.numeric(quantile(abs(x), q, na.rm = TRUE, names = FALSE))
  if (!is.finite(cap) || cap <= 0) return(x)
  pmin(pmax(x, -cap), cap)
}

draw_initial_states <- function(dgp, n) {
  cbind(
    rnorm(n, mean = dgp$init_mean[1], sd = dgp$init_sd[1]),
    rnorm(n, mean = dgp$init_mean[2], sd = dgp$init_sd[2])
  )
}

# ==============================================================================
# 1. Data-generating process (2D state)
# ==============================================================================
generate_dgp_continuous <- function(
    sigma_tr = 0.5,          ## Transition noise sd (variance = 0.25)
    sigma_R  = 1,          ## Reward noise sd (independent of transition)
    b_prob   = 0.5,          ## Behavior policy: b(1|s) = 0.5 (constant)
    tr_shift = 3 / 5,        ## Common action shift in transitions
    s1_a_int = 0.5,          ## S1 transition action-state interaction
    s2_a_int = -0.5,         ## S2 transition action-state interaction
    init_mean = c(0.5, -0.5), ## Initial-state mean
    init_sd   = c(1, 1), ## Initial-state sd
    ## Target policy: pi_2(1|s) = I(s1<=0, s2<=0)
    pi_func  = function(s1, s2) as.numeric(s1 >= 0.25 & s2 >= -0.10)
) {
  list(sigma_tr = sigma_tr,
       sigma_R  = sigma_R,
       b_prob   = b_prob,
       tr_shift = tr_shift,
       s1_a_int = s1_a_int,
       s2_a_int = s2_a_int,
       init_mean = init_mean,
       init_sd = init_sd,
       pi_func  = pi_func)
}

# ==============================================================================
# 2. True value (Monte Carlo rollout)
# ==============================================================================
compute_true_value_continuous <- function(dgp, gamma,
                                          n_mc = 4000, TT_mc = 2000) {
  pi_func  <- dgp$pi_func
  sigma_tr <- dgp$sigma_tr
  tr_shift <- dgp$tr_shift
  s1_a_int <- dgp$s1_a_int
  s2_a_int <- dgp$s2_a_int

  total <- 0
  for (i in 1:n_mc) {
    s_init <- draw_initial_states(dgp, 1)
    s1 <- s_init[1, 1];  s2 <- s_init[1, 2]
    disc <- 1;  val <- 0
    for (t in 1:TT_mc) {
      p1 <- pi_func(s1, s2)
      a  <- rbinom(1, 1, p1)
      s1_new <- (1/2) * s1 + tr_shift * (2*a - 1) + s1_a_int * s1 * a +
        rnorm(1, 0, sigma_tr)
      s2_new <- (1/3) * s2 - tr_shift * (2*a - 1) + s2_a_int * s2 * a +
        rnorm(1, 0, sigma_tr)
      ## Reward depends on (S, A) only, NOT on S' => R ⊥ S' | (S, A)
      r <- 2+s1 + (1/2)*s2 + (3/2)*a  + rnorm(1, 0, .5)
      val  <- val + disc * r
      disc <- disc * gamma
      s1 <- s1_new;  s2 <- s2_new
    }
    total <- total + val
  }
  list(V_value = total / n_mc)
}

# ==============================================================================
# 3. Data generation (2D state)
# ==============================================================================
generate_data_continuous <- function(dgp, N, TT, epsilon) {
  tr_shift <- dgp$tr_shift
  s1_a_int <- dgp$s1_a_int
  s2_a_int <- dgp$s2_a_int
  S1 <- S2 <- A <- Atilde <- R <- Sp1 <- Sp2 <- matrix(NA_real_, N, TT)
  for (i in 1:N) {
    s_init <- draw_initial_states(dgp, 1)
    s1 <- s_init[1, 1];  s2 <- s_init[1, 2]
    for (t in 1:TT) {
      S1[i, t] <- s1;  S2[i, t] <- s2
      a <- rbinom(1, 1, dgp$b_prob)
      A[i, t] <- a
      Atilde[i, t] <- ifelse(runif(1) < epsilon, 1 - a, a)
      ## Transition with milder state-action interactions to improve overlap
      s1_new <- (1/2) * s1 + tr_shift * (2*a - 1) + s1_a_int * s1 * a +
        rnorm(1, 0, dgp$sigma_tr)
      s2_new <- (1/3) * s2 - tr_shift * (2*a - 1) + s2_a_int * s2 * a +
        rnorm(1, 0, dgp$sigma_tr)
      Sp1[i, t] <- s1_new;  Sp2[i, t] <- s2_new
      ## Reward depends on (S, A) only, NOT on S' => R ⊥ S' | (S, A)
      R[i, t] <- 2+s1 + (1/2)*s2 + (3/2)*a  + rnorm(1, 0, dgp$sigma_R)
      s1 <- s1_new;  s2 <- s2_new
    }
  }
  list(S1 = S1, S2 = S2, A = A, Atilde = Atilde, R = R,
       Sp1 = Sp1, Sp2 = Sp2)
}

# ==============================================================================
# 4. EM-style nuisance estimation (2D state, weighted linear models)
# ==============================================================================
.em_single_run <- function(S1_vec, S2_vec, At_vec, R_vec, Sp1_vec, Sp2_vec,
                            init_eta, max_iter = 100, tol = 1e-4) {
  n <- length(S1_vec)
  eta <- init_eta

  fit_b <- fit_mu0 <- fit_mu1 <- NULL
  fit_R0 <- fit_R1 <- NULL
  fit_Sp1_0 <- fit_Sp1_1 <- fit_Sp2_0 <- fit_Sp2_1 <- NULL
  sigma_R   <- c(1, 1)
  sigma_Sp1 <- c(1, 1)
  sigma_Sp2 <- c(1, 1)
  loglik <- -Inf

  df_s <- data.frame(s1 = S1_vec, s2 = S2_vec)

  for (iter in 1:max_iter) {
    eta_old <- eta

    ## ---- M-step ----
    fit_b <- lm(eta1 ~ s1 + s2,
                data = cbind(data.frame(eta1 = eta[, 2]), df_s))
    b_hat <- clip(fitted(fit_b), 0.02, 0.98)

    fit_mu0 <- glm(at ~ s1 + s2, family = quasibinomial(),
                   weights = pmax(eta[, 1], 1e-8),
                   data = cbind(data.frame(at = At_vec), df_s))
    fit_mu1 <- glm(at ~ s1 + s2, family = quasibinomial(),
                   weights = pmax(eta[, 2], 1e-8),
                   data = cbind(data.frame(at = At_vec), df_s))
    mu_hat_0 <- clip(fitted(fit_mu0, type = "response"), 0.02, 0.98)
    mu_hat_1 <- clip(fitted(fit_mu1, type = "response"), 0.02, 0.98)

    ## Reward models
    fit_R0 <- lm(r ~ s1 + s2, weights = pmax(eta[, 1], 1e-8),
                 data = cbind(data.frame(r = R_vec), df_s))
    fit_R1 <- lm(r ~ s1 + s2, weights = pmax(eta[, 2], 1e-8),
                 data = cbind(data.frame(r = R_vec), df_s))
    tR0 <- fitted(fit_R0);  tR1 <- fitted(fit_R1)
    sigma_R[1] <- sqrt(max(sum(eta[, 1] * (R_vec - tR0)^2) / sum(eta[, 1]), 0.01))
    sigma_R[2] <- sqrt(max(sum(eta[, 2] * (R_vec - tR1)^2) / sum(eta[, 2]), 0.01))

    ## Transition models — S1'
    fit_Sp1_0 <- lm(sp ~ s1 + s2, weights = pmax(eta[, 1], 1e-4),
                    data = cbind(data.frame(sp = Sp1_vec), df_s))
    fit_Sp1_1 <- lm(sp ~ s1 + s2, weights = pmax(eta[, 2], 1e-4),
                    data = cbind(data.frame(sp = Sp1_vec), df_s))
    tSp1_0 <- fitted(fit_Sp1_0);  tSp1_1 <- fitted(fit_Sp1_1)
    sigma_Sp1[1] <- sqrt(max(sum(eta[, 1] * (Sp1_vec - tSp1_0)^2) / sum(eta[, 1]), 0.01))
    sigma_Sp1[2] <- sqrt(max(sum(eta[, 2] * (Sp1_vec - tSp1_1)^2) / sum(eta[, 2]), 0.01))

    ## Transition models — S2'
    fit_Sp2_0 <- lm(sp ~ s1 + s2, weights = pmax(eta[, 1], 1e-4),
                    data = cbind(data.frame(sp = Sp2_vec), df_s))
    fit_Sp2_1 <- lm(sp ~ s1 + s2, weights = pmax(eta[, 2], 1e-4),
                    data = cbind(data.frame(sp = Sp2_vec), df_s))
    tSp2_0 <- fitted(fit_Sp2_0);  tSp2_1 <- fitted(fit_Sp2_1)
    sigma_Sp2[1] <- sqrt(max(sum(eta[, 1] * (Sp2_vec - tSp2_0)^2) / sum(eta[, 1]), 0.01))
    sigma_Sp2[2] <- sqrt(max(sum(eta[, 2] * (Sp2_vec - tSp2_1)^2) / sum(eta[, 2]), 0.01))

    ## ---- E-step ----
    log_eta <- matrix(0, n, 2)
    for (a in 0:1) {
      log_b  <- ifelse(a == 1, log(b_hat), log(1 - b_hat))
      mu_a   <- if (a == 0) mu_hat_0 else mu_hat_1
      log_mu <- dbinom(At_vec, 1, mu_a, log = TRUE)
      tRa    <- if (a == 0) tR0 else tR1
      log_h  <- dnorm(R_vec, tRa, sigma_R[a + 1], log = TRUE)
      tSp1a  <- if (a == 0) tSp1_0 else tSp1_1
      log_q1 <- dnorm(Sp1_vec, tSp1a, sigma_Sp1[a + 1], log = TRUE)
      tSp2a  <- if (a == 0) tSp2_0 else tSp2_1
      log_q2 <- dnorm(Sp2_vec, tSp2a, sigma_Sp2[a + 1], log = TRUE)
      log_eta[, a + 1] <- log_b + log_mu + log_h + log_q1 + log_q2
    }
    mx  <- apply(log_eta, 1, max)
    loglik <- sum(log(rowSums(exp(log_eta - mx))) + mx)
    eta <- exp(log_eta - mx)
    eta <- eta / rowSums(eta)
    eta <- pmax(eta, 1e-5)
    eta <- eta / rowSums(eta)

    if (max(abs(eta - eta_old)) < tol) break
  }

  list(eta = eta, loglik = loglik, n_iter = iter,
       sigma_R = sigma_R, sigma_Sp1 = sigma_Sp1, sigma_Sp2 = sigma_Sp2,
       fit_R0 = fit_R0, fit_R1 = fit_R1,
       fit_Sp1_0 = fit_Sp1_0, fit_Sp1_1 = fit_Sp1_1,
       fit_Sp2_0 = fit_Sp2_0, fit_Sp2_1 = fit_Sp2_1,
       fit_mu0 = fit_mu0, fit_mu1 = fit_mu1, fit_b = fit_b)
}

em_continuous <- function(dat, gamma, max_iter = 100, tol = 1e-3,
                           n_restarts = 2) {
  S1_vec  <- as.vector(dat$S1)
  S2_vec  <- as.vector(dat$S2)
  At_vec  <- as.vector(dat$Atilde)
  R_vec   <- as.vector(dat$R)
  Sp1_vec <- as.vector(dat$Sp1)
  Sp2_vec <- as.vector(dat$Sp2)
  n <- length(S1_vec)

  best <- NULL

  for (rst in 1:n_restarts) {
    eta0 <- matrix(0.5, n, 2)
    if (rst == 1) {
      eta0[cbind(1:n, At_vec + 1L)] <- 0.7
      eta0[cbind(1:n, 2L - At_vec)] <- 0.3
    } else {
      p <- runif(n, 0.55, 0.85)
      eta0[cbind(1:n, At_vec + 1L)] <- p
      eta0[cbind(1:n, 2L - At_vec)] <- 1 - p
    }

    res <- .em_single_run(S1_vec, S2_vec, At_vec, R_vec, Sp1_vec, Sp2_vec,
                          eta0, max_iter = max_iter, tol = tol)
    if (is.null(best) || res$loglik > best$loglik) best <- res
  }

  eta <- best$eta
  fit_R0 <- best$fit_R0;  fit_R1 <- best$fit_R1
  fit_Sp1_0 <- best$fit_Sp1_0;  fit_Sp1_1 <- best$fit_Sp1_1
  fit_Sp2_0 <- best$fit_Sp2_0;  fit_Sp2_1 <- best$fit_Sp2_1
  fit_mu0 <- best$fit_mu0;  fit_mu1 <- best$fit_mu1
  fit_b   <- best$fit_b
  sigma_R   <- best$sigma_R
  sigma_Sp1 <- best$sigma_Sp1;  sigma_Sp2 <- best$sigma_Sp2

  ## Relabel so class 2 (A=1) has HIGHER mean Atilde
  mean_at <- c(sum(eta[, 1] * At_vec) / sum(eta[, 1]),
               sum(eta[, 2] * At_vec) / sum(eta[, 2]))
  if (mean_at[1] > mean_at[2]) {
    eta <- eta[, 2:1, drop = FALSE]
    tmp <- fit_R0;  fit_R0  <- fit_R1;  fit_R1  <- tmp
    tmp <- fit_Sp1_0; fit_Sp1_0 <- fit_Sp1_1; fit_Sp1_1 <- tmp
    tmp <- fit_Sp2_0; fit_Sp2_0 <- fit_Sp2_1; fit_Sp2_1 <- tmp
    tmp <- fit_mu0;   fit_mu0   <- fit_mu1;   fit_mu1   <- tmp
    sigma_R   <- rev(sigma_R)
    sigma_Sp1 <- rev(sigma_Sp1)
    sigma_Sp2 <- rev(sigma_Sp2)
    fit_b <- lm(eta1 ~ s1 + s2,
                data = data.frame(eta1 = eta[, 2], s1 = S1_vec, s2 = S2_vec))
  }

  predict_theta_R <- function(s1_new, s2_new, a) {
    fit <- if (a == 0) fit_R0 else fit_R1
    predict(fit, newdata = data.frame(s1 = s1_new, s2 = s2_new))
  }
  predict_theta_Sp1 <- function(s1_new, s2_new, a) {
    fit <- if (a == 0) fit_Sp1_0 else fit_Sp1_1
    predict(fit, newdata = data.frame(s1 = s1_new, s2 = s2_new))
  }
  predict_theta_Sp2 <- function(s1_new, s2_new, a) {
    fit <- if (a == 0) fit_Sp2_0 else fit_Sp2_1
    predict(fit, newdata = data.frame(s1 = s1_new, s2 = s2_new))
  }
  predict_mu <- function(s1_new, s2_new, a) {
    fit <- if (a == 0) fit_mu0 else fit_mu1
    clip(predict(fit, newdata = data.frame(s1 = s1_new, s2 = s2_new),
                 type = "response"), 0.02, 0.98)
  }
  predict_b <- function(s1_new, s2_new) {
    clip(predict(fit_b, newdata = data.frame(s1 = s1_new, s2 = s2_new)),
         0.02, 0.98)
  }

  list(eta = eta, n_iter = best$n_iter,
       sigma_R = sigma_R, sigma_Sp1 = sigma_Sp1, sigma_Sp2 = sigma_Sp2,
       predict_theta_R = predict_theta_R,
       predict_theta_Sp1 = predict_theta_Sp1,
       predict_theta_Sp2 = predict_theta_Sp2,
       predict_mu = predict_mu, predict_b = predict_b,
       fit_R0 = fit_R0, fit_R1 = fit_R1,
       fit_Sp1_0 = fit_Sp1_0, fit_Sp1_1 = fit_Sp1_1,
       fit_Sp2_0 = fit_Sp2_0, fit_Sp2_1 = fit_Sp2_1,
       fit_mu0 = fit_mu0, fit_mu1 = fit_mu1, fit_b = fit_b)
}

# ==============================================================================
# 5. Density-ratio estimation (linear features, moment equation)
# ==============================================================================
solve_omega_continuous <- function(em_out, dat, dgp, gamma, ridge = 0.001) {
  S1_vec  <- as.vector(dat$S1)
  S2_vec  <- as.vector(dat$S2)
  Sp1_vec <- as.vector(dat$Sp1)
  Sp2_vec <- as.vector(dat$Sp2)
  N  <- nrow(dat$S1);  TT <- ncol(dat$S1)
  n  <- N * TT
  eta <- em_out$eta
  pi_func <- dgp$pi_func

  ## Linear features: (1, s1, s2) — same as FQE
  K <- 3L
  d <- 2L * K   # action-specific blocks

  phi_mat <- function(s1, s2, a) {
    pf  <- cbind(1, s1, s2)
    out <- matrix(0, length(s1), d)
    m0  <- (a == 0);  m1 <- (a == 1)
    if (any(m0)) out[m0, 1:K]     <- pf[m0, , drop = FALSE]
    if (any(m1)) out[m1, (K+1):d] <- pf[m1, , drop = FALSE]
    out
  }

  phi_pi <- function(s1, s2) {
    pf   <- cbind(1, s1, s2)
    pi_s <- pi_func(s1, s2)
    out  <- matrix(0, length(s1), d)
    out[, 1:K]     <- (1 - pi_s) * pf
    out[, (K+1):d] <- pi_s * pf
    out
  }

  Phi_0    <- phi_mat(S1_vec, S2_vec, rep(0L, n))
  Phi_1    <- phi_mat(S1_vec, S2_vec, rep(1L, n))
  PhiPi_Sp <- phi_pi(Sp1_vec, Sp2_vec)

  A_mat <- matrix(0, d, d)
  for (a in 0:1) {
    Phi_a <- if (a == 0) Phi_0 else Phi_1
    eta_a <- eta[, a + 1]
    diff_a <- Phi_a - gamma * PhiPi_Sp
    A_mat <- A_mat + crossprod(diff_a, Phi_a * eta_a) / n
  }

  ## Integrate over the known initial distribution
  S_mc <- draw_initial_states(dgp, 10000)
  S1_mc <- S_mc[, 1];  S2_mc <- S_mc[, 2]
  b_vec <- (1 - gamma) * colMeans(phi_pi(S1_mc, S2_mc))

  beta_hat <- solve(A_mat + ridge * diag(d), b_vec)

  ## Post-normalize so E[sum_a eta_a * omega(s,a)] = 1  (ridge may break this)
  om0_tmp <- drop(Phi_0 %*% beta_hat)
  om1_tmp <- drop(Phi_1 %*% beta_hat)
  norm_c  <- mean(eta[, 1] * om0_tmp + eta[, 2] * om1_tmp)
  if (abs(norm_c) > 1e-6) beta_hat <- beta_hat / norm_c

  predict_omega <- function(s1, s2, a) drop(phi_mat(s1, s2, a) %*% beta_hat)

  omega_all <- cbind(predict_omega(S1_vec, S2_vec, rep(0L, n)),
                     predict_omega(S1_vec, S2_vec, rep(1L, n)))

  list(beta = beta_hat, predict_omega = predict_omega, omega_all = omega_all)
}

# ==============================================================================
# 6. Weighted FQE for Q^pi (2D state, using EM weights)
# ==============================================================================
weighted_fqe_continuous <- function(em_out, dat, dgp, gamma, n_iter = 50) {
  S1_vec  <- as.vector(dat$S1)
  S2_vec  <- as.vector(dat$S2)
  Sp1_vec <- as.vector(dat$Sp1)
  Sp2_vec <- as.vector(dat$Sp2)
  R_vec   <- as.vector(dat$R)
  n   <- length(S1_vec)
  pi_func <- dgp$pi_func
  eta <- em_out$eta

  pi_sp <- pi_func(Sp1_vec, Sp2_vec)

  Q0_sp <- rep(0, n);  Q1_sp <- rep(0, n)
  fit_Q0 <- fit_Q1 <- NULL

  for (iter in 1:n_iter) {
    V_sp     <- (1 - pi_sp) * Q0_sp + pi_sp * Q1_sp
    Y_target <- R_vec + gamma * V_sp

    fit_Q0 <- lm(y ~ s1 + s2, weights = pmax(eta[, 1], 1e-5),
                 data = data.frame(y = Y_target, s1 = S1_vec, s2 = S2_vec))
    fit_Q1 <- lm(y ~ s1 + s2, weights = pmax(eta[, 2], 1e-5),
                 data = data.frame(y = Y_target, s1 = S1_vec, s2 = S2_vec))

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

  list(predict_Q = predict_Q, predict_V = predict_V,
       fit_Q0 = fit_Q0, fit_Q1 = fit_Q1)
}

# ==============================================================================
# 7. Helper: subset data by trajectory (row) indices
# ==============================================================================
subset_dat <- function(dat, row_idx) {
  list(S1 = dat$S1[row_idx, , drop = FALSE],
       S2 = dat$S2[row_idx, , drop = FALSE],
       A  = dat$A[row_idx, , drop = FALSE],
       Atilde = dat$Atilde[row_idx, , drop = FALSE],
       R  = dat$R[row_idx, , drop = FALSE],
       Sp1 = dat$Sp1[row_idx, , drop = FALSE],
       Sp2 = dat$Sp2[row_idx, , drop = FALSE])
}

# ==============================================================================
# 8. Helper: compute out-of-fold eta (one E-step with trained EM models)
# ==============================================================================
compute_eta_outfold <- function(em, S1_te, S2_te, At_te, R_te, Sp1_te, Sp2_te) {
  n <- length(S1_te)
  b_hat  <- clip(em$predict_b(S1_te, S2_te), 0.02, 0.98)
  mu_hat_0 <- em$predict_mu(S1_te, S2_te, 0)
  mu_hat_1 <- em$predict_mu(S1_te, S2_te, 1)
  tR0    <- em$predict_theta_R(S1_te, S2_te, 0)
  tR1    <- em$predict_theta_R(S1_te, S2_te, 1)
  tSp1_0 <- em$predict_theta_Sp1(S1_te, S2_te, 0)
  tSp1_1 <- em$predict_theta_Sp1(S1_te, S2_te, 1)
  tSp2_0 <- em$predict_theta_Sp2(S1_te, S2_te, 0)
  tSp2_1 <- em$predict_theta_Sp2(S1_te, S2_te, 1)

  log_eta <- matrix(0, n, 2)
  for (a in 0:1) {
    log_b  <- ifelse(a == 1, log(b_hat), log(1 - b_hat))
    mu_a   <- if (a == 0) mu_hat_0 else mu_hat_1
    log_mu <- dbinom(At_te, 1, mu_a, log = TRUE)
    tRa    <- if (a == 0) tR0 else tR1
    log_h  <- dnorm(R_te, tRa, em$sigma_R[a + 1], log = TRUE)
    tSp1a  <- if (a == 0) tSp1_0 else tSp1_1
    log_q1 <- dnorm(Sp1_te, tSp1a, em$sigma_Sp1[a + 1], log = TRUE)
    tSp2a  <- if (a == 0) tSp2_0 else tSp2_1
    log_q2 <- dnorm(Sp2_te, tSp2a, em$sigma_Sp2[a + 1], log = TRUE)
    log_eta[, a + 1] <- log_b + log_mu + log_h + log_q1 + log_q2
  }
  mx  <- apply(log_eta, 1, max)
  eta <- exp(log_eta - mx)
  eta <- eta / rowSums(eta)
  eta <- pmax(eta, 1e-5)
  eta <- eta / rowSums(eta)
  eta
}

# ==============================================================================
# 9. LURE (multiply robust) estimator with K=2 cross-fitting
# ==============================================================================
mr_estimator_continuous <- function(dat, dgp, gamma) {
  N  <- nrow(dat$S1)
  TT <- ncol(dat$S1)
  bridge_clip_q <- 0.97

  ## K=2 cross-fitting: split trajectories into 2 folds
  fold_ids <- sample(rep(1:2, length.out = N))

  phi_list    <- vector("list", 2)
  direct_list <- numeric(2)

  for (k in 1:2) {
    train_idx <- which(fold_ids != k)
    test_idx  <- which(fold_ids == k)
    n_test    <- length(test_idx) * TT

    dat_train <- subset_dat(dat, train_idx)

    ## Test-fold vectors
    S1_te  <- as.vector(dat$S1[test_idx, ])
    S2_te  <- as.vector(dat$S2[test_idx, ])
    At_te  <- as.vector(dat$Atilde[test_idx, ])
    R_te   <- as.vector(dat$R[test_idx, ])
    Sp1_te <- as.vector(dat$Sp1[test_idx, ])
    Sp2_te <- as.vector(dat$Sp2[test_idx, ])

    ## Fit nuisances on training fold
    em        <- em_continuous(dat_train, gamma)
    omega_out <- solve_omega_continuous(em, dat_train, dgp, gamma)
    fqe       <- weighted_fqe_continuous(em, dat_train, dgp, gamma)

    ## Out-of-fold eta for test data
    eta_te <- compute_eta_outfold(em, S1_te, S2_te, At_te, R_te,
                                  Sp1_te, Sp2_te)

    ## Evaluate nuisances on test fold
    tR0  <- em$predict_theta_R(S1_te, S2_te, 0)
    tR1  <- em$predict_theta_R(S1_te, S2_te, 1)
    tAt0 <- em$predict_mu(S1_te, S2_te, 0)
    tAt1 <- em$predict_mu(S1_te, S2_te, 1)

    om0 <- omega_out$predict_omega(S1_te, S2_te, rep(0L, n_test))
    om1 <- omega_out$predict_omega(S1_te, S2_te, rep(1L, n_test))

    ## Clip omega for MR augmentation terms
    omega_clip <- quantile(c(abs(om0), abs(om1)), 0.97)
    om0_clip <- pmin(pmax(om0, -omega_clip), omega_clip)
    om1_clip <- pmin(pmax(om1, -omega_clip), omega_clip)

    V_sp <- fqe$predict_V(Sp1_te, Sp2_te)
    Q0   <- fqe$predict_Q(S1_te, S2_te, 0)
    Q1   <- fqe$predict_Q(S1_te, S2_te, 1)
    MV0  <- (Q0 - tR0) / gamma
    MV1  <- (Q1 - tR1) / gamma

    ## Direct term
    S_mc <- draw_initial_states(dgp, 10000)
    S1_mc <- S_mc[, 1];  S2_mc <- S_mc[, 2]
    direct_list[k] <- mean(fqe$predict_V(S1_mc, S2_mc))

    ## Bridge functions
    tSp2_0 <- em$predict_theta_Sp2(S1_te, S2_te, 0)
    tSp2_1 <- em$predict_theta_Sp2(S1_te, S2_te, 1)

    d_At_0  <- tAt0 - tAt1
    br_At_0 <- clip_abs_quantile(safe_ratio(At_te - tAt1, d_At_0), bridge_clip_q)
    d_At_1  <- tAt1 - tAt0
    br_At_1 <- clip_abs_quantile(safe_ratio(At_te - tAt0, d_At_1), bridge_clip_q)

    d_R_0  <- tR0 - tR1
    br_R_0 <- clip_abs_quantile(safe_ratio(R_te - tR1, d_R_0), bridge_clip_q)
    d_R_1  <- tR1 - tR0
    br_R_1 <- clip_abs_quantile(safe_ratio(R_te - tR0, d_R_1), bridge_clip_q)

    d_Sp2_0  <- tSp2_0 - tSp2_1
    br_Sp2_0 <- clip_abs_quantile(safe_ratio(Sp2_te - tSp2_1, d_Sp2_0), bridge_clip_q)
    d_Sp2_1  <- tSp2_1 - tSp2_0
    br_Sp2_1 <- clip_abs_quantile(safe_ratio(Sp2_te - tSp2_0, d_Sp2_1), bridge_clip_q)

    g0  <- clip_abs_quantile(br_At_0 * br_R_0, bridge_clip_q)
    g1  <- clip_abs_quantile(br_At_1 * br_R_1, bridge_clip_q)
    gp0 <- clip_abs_quantile(br_At_0 * br_Sp2_0, bridge_clip_q)
    gp1 <- clip_abs_quantile(br_At_1 * br_Sp2_1, bridge_clip_q)

    ## IF components
    T1 <- gp0 * om0_clip * (R_te - tR0) + gp1 * om1_clip * (R_te - tR1)
    T2 <- g0  * om0_clip * (V_sp - MV0) + g1  * om1_clip * (V_sp - MV1)

    phi_list[[k]] <- T1 / (1 - gamma) + T2 * gamma / (1 - gamma)

  }

  direct  <- mean(direct_list)
  phi_all <- unlist(phi_list)

  V_hat <- direct + mean(phi_all)
  se    <- sqrt(var(phi_all) / (N * TT))

  list(V_hat = V_hat, se = se,
      ci_lo = V_hat - 1.96 * se, ci_hi = V_hat + 1.96 * se)
}


# ==============================================================================
# 12. One-replication wrapper
# ==============================================================================
one_rep_continuous <- function(dgp, N, TT, epsilon, gamma) {
  dat <- generate_data_continuous(dgp, N, TT, epsilon)

  mr_out <- tryCatch(
    mr_estimator_continuous(dat, dgp, gamma),
    error = function(e) list(V_hat = NA_real_, ci_lo = NA_real_, ci_hi = NA_real_)
  )
  V_mr     <- mr_out$V_hat
  mr_cilo  <- mr_out$ci_lo
  mr_cihi  <- mr_out$ci_hi

  V_fqe <- tryCatch(
    naive_fqe_continuous(dat, dgp, gamma)$V_hat,
    error = function(e) NA_real_
  )

  V_sis <- tryCatch(
    naive_sis_continuous(dat, dgp, gamma),
    error = function(e) NA_real_
  )

  V_mis <- tryCatch(
    naive_mis_continuous(dat, dgp, gamma)$V_hat,
    error = function(e) NA_real_
  )

  V_drl <- tryCatch(
    naive_drl_continuous(dat, dgp, gamma)$V_hat,
    error = function(e) NA_real_
  )

  V_lstd <- tryCatch(
    naive_lstd_continuous(dat, dgp, gamma)$V_hat,
    error = function(e) NA_real_
  )

  c(FQE = V_fqe, SIS = V_sis, MIS = V_mis, DRL = V_drl, LSTD = V_lstd,
    MR = V_mr,
    MR_ci_lo = mr_cilo, MR_ci_hi = mr_cihi)
}
