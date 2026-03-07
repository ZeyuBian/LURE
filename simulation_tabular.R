################################################################################
## Tabular MDP Simulation — "Learning from the Unseen"
##
## DGP: |S|=3, A in {0,1}, gamma=0.9, with hidden actions
## Competing methods (FQE, MIS, DRL) loaded from Methods.R
## Our method: multiply robust (MR) estimator with EM-style nuisance estimation
################################################################################

source("Methods.R")
library(rpart)

# ==============================================================================
# 1. Fixed tabular MDP (DGP) — simple & interpretable
# ==============================================================================
generate_dgp <- function() {
  nS <- 3   # three states: "good" (1), "medium" (2), "bad" (3)

  ## Transition kernel: P_trans[s, s', a+1] = P(S'=s' | S=s, A=a)
  ##   Action 1 = "treat" (tends to move toward good state)
  ##   Action 0 = "no treat" (tends to drift toward bad state)
  P_trans <- array(NA, dim = c(nS, nS, 2))

  #                       s'=1   s'=2   s'=3
  P_trans[1, , 1] <- c(  0.2,   0.5,   0.3 )   # s=1, a=0
  P_trans[2, , 1] <- c(  0.1,   0.3,   0.6 )   # s=2, a=0
  P_trans[3, , 1] <- c(  0.05,  0.25,  0.7 )   # s=3, a=0

  P_trans[1, , 2] <- c(  0.7,   0.2,   0.1 )   # s=1, a=1
  P_trans[2, , 2] <- c(  0.4,   0.4,   0.2 )   # s=2, a=1
  P_trans[3, , 2] <- c(  0.2,   0.5,   0.3 )   # s=3, a=1

  ## Reward function: theta_R(s,a), nS x 2 matrix [col1 = a=0, col2 = a=1]
  theta_R <- matrix(c(
    1.0,  2.0,    # s=1: base 1.0, treatment adds 1.0
    0.5,  1.5,    # s=2: base 0.5, treatment adds 1.0
    0.0,  0.5     # s=3: base 0.0, treatment adds 0.5
  ), nS, 2, byrow = TRUE)

  ## Behavior policy: b(1|s) — doctor treats more often in bad states
  b_policy <- c(0.3, 0.5, 0.7)

  ## Target policy: pi(1|s) — always treat
  pi_policy <- c(0.8, 0.8, 0.8)

  ## Initial distribution: uniform
  p_e <- rep(1 / nS, nS)

  ## Reward noise sd
  sigma_R <- 0.5

  list(nS = nS, P_trans = P_trans, theta_R = theta_R,
       b_policy = b_policy, pi_policy = pi_policy,
       p_e = p_e, sigma_R = sigma_R)
}

# ==============================================================================
# 2. True V(pi) by Bellman matrix inversion
# ==============================================================================
compute_true_value <- function(dgp, gamma) {
  nS <- dgp$nS

  ## Transition under pi: P_pi[s, s'] = sum_a pi(a|s) P(s'|s,a)
  P_pi <- (1 - dgp$pi_policy) * dgp$P_trans[, , 1] +
              dgp$pi_policy  * dgp$P_trans[, , 2]

  ## Mean reward under pi: r_pi(s) = sum_a pi(a|s) theta_R(s,a)
  r_pi <- (1 - dgp$pi_policy) * dgp$theta_R[, 1] +
               dgp$pi_policy  * dgp$theta_R[, 2]

  ## V^pi = (I - gamma P_pi)^{-1} r_pi
  V_pi <- as.numeric(solve(diag(nS) - gamma * P_pi, r_pi))

  ## Q^pi(s,a) = theta_R(s,a) + gamma P(·|s,a)^T V^pi
  Q_pi <- dgp$theta_R + gamma * cbind(dgp$P_trans[, , 1] %*% V_pi,
                                       dgp$P_trans[, , 2] %*% V_pi)

  V_value <- sum(dgp$p_e * V_pi)
  list(V_pi = V_pi, Q_pi = Q_pi, V_value = V_value)
}

# ==============================================================================
# 3. Generate trajectory data
# ==============================================================================
generate_data <- function(dgp, N, TT, epsilon) {
  nS <- dgp$nS
  S <- A <- Atilde <- R <- Sprime <- matrix(NA, N, TT)

  for (i in 1:N) {
    s <- sample(1:nS, 1, prob = dgp$p_e)
    for (t in 1:TT) {
      S[i, t] <- s

      ## True action from behavior policy
      a <- rbinom(1, 1, dgp$b_policy[s])
      A[i, t] <- a

      ## Surrogate action (constant misclassification rate epsilon)
      Atilde[i, t] <- ifelse(runif(1) < epsilon, 1 - a, a)

      ## Reward
      R[i, t] <- dgp$theta_R[s, a + 1] + rnorm(1, 0, dgp$sigma_R)

      ## Next state
      s <- sample(1:nS, 1, prob = dgp$P_trans[s, , a + 1])
      Sprime[i, t] <- s
    }
  }
  list(S = S, A = A, Atilde = Atilde, R = R, Sprime = Sprime)
}

# ==============================================================================
# 4. EM-style nuisance estimation (tabular)
# ==============================================================================
em_tabular <- function(dat, nS, gamma, max_iter = 100, tol = 1e-4) {
  S  <- as.vector(dat$S)
  At <- as.vector(dat$Atilde)
  R  <- as.vector(dat$R)
  Sp <- as.vector(dat$Sprime)
  n  <- length(S)

  ## Initialise responsibilities from the surrogate
  eta <- matrix(0.5, n, 2)
  eta[cbind(1:n, At + 1)] <- 0.7
  eta[cbind(1:n, 2 - At)] <- 0.3

  for (iter in 1:max_iter) {
    eta_old <- eta

    ## ---- M-step ----
    b_hat      <- numeric(nS)
    mu_hat     <- matrix(0.5, nS, 2)
    theta_R_hat <- matrix(0, nS, 2)
    sigma_R_hat <- matrix(1, nS, 2)
    P_hat      <- array(1 / nS, dim = c(nS, nS, 2))

    for (s in 1:nS) {
      idx <- which(S == s)
      if (length(idx) == 0) next

      ## Behavior policy
      b_hat[s] <- mean(eta[idx, 2])

      for (a in 0:1) {
        w <- eta[idx, a + 1]
        sw <- sum(w)
        if (sw < 1e-10) next

        ## Measurement model
        mu_hat[s, a + 1] <- sum(w * At[idx]) / sw

        ## Reward regression
        theta_R_hat[s, a + 1] <- sum(w * R[idx]) / sw
        sigma_R_hat[s, a + 1] <- sqrt(max(sum(w * (R[idx] - theta_R_hat[s, a + 1])^2) / sw, 0.01))

        ## Transition model
        for (sp in 1:nS) P_hat[s, sp, a + 1] <- sum(w * (Sp[idx] == sp))
        tot <- sum(P_hat[s, , a + 1])
        if (tot > 1e-10) P_hat[s, , a + 1] <- P_hat[s, , a + 1] / tot
      }
    }
    b_hat  <- pmax(pmin(b_hat, 0.99), 0.01)
    mu_hat <- pmax(pmin(mu_hat, 0.99), 0.01)

    ## ---- E-step ----
    for (i in 1:n) {
      s <- S[i]; at <- At[i]; r <- R[i]; sp <- Sp[i]
      for (a in 0:1) {
        eta[i, a + 1] <-
          log(ifelse(a == 1, b_hat[s], 1 - b_hat[s])) +
          dbinom(at, 1, mu_hat[s, a + 1], log = TRUE) +
          dnorm(r, theta_R_hat[s, a + 1], sigma_R_hat[s, a + 1], log = TRUE) +
          log(max(P_hat[s, sp, a + 1], 1e-10))
      }
      mx <- max(eta[i, ])
      eta[i, ] <- exp(eta[i, ] - mx)
      eta[i, ] <- eta[i, ] / sum(eta[i, ])
    }

    if (max(abs(eta - eta_old)) < tol) break
  }

  ## theta_{S'}(s,a) = E[S' | S=s, A=a]   (state index treated as numeric)
  theta_Sp_hat <- matrix(0, nS, 2)
  for (s in 1:nS) {
    idx <- which(S == s)
    if (length(idx) == 0) next
    for (a in 0:1) {
      w <- eta[idx, a + 1]; sw <- sum(w)
      if (sw > 1e-10) theta_Sp_hat[s, a + 1] <- sum(w * Sp[idx]) / sw
    }
  }

  list(eta = eta, b_hat = b_hat, mu_hat = mu_hat,
       theta_R_hat = theta_R_hat, sigma_R_hat = sigma_R_hat,
       P_hat = P_hat, theta_Sp_hat = theta_Sp_hat,
       theta_At_hat = mu_hat)
}

# ==============================================================================
# 5. Solve for density ratio omega (tabular)
# ==============================================================================
solve_omega_tabular <- function(em_out, dat, dgp, gamma) {
  nS  <- dgp$nS
  pi  <- dgp$pi_policy          # pi(1|s)
  p_e <- dgp$p_e
  S   <- as.vector(dat$S)

  ## Helper: (s,a) -> flat index among nS*2 entries
  sa <- function(s, a) a * nS + s          # a∈{0,1}

  dim_sa <- nS * 2

  ## Solve for discounted visitation d^pi(s,a)
  ##   d^pi(s,a) = (1-γ) p_e(s) π(a|s) + γ Σ_{s',a'} d^pi(s',a') P̂(s|s',a') π(a|s)
  build_discounted_target <- function(P) {
    A <- diag(dim_sa)
    b <- numeric(dim_sa)
    for (s in 1:nS) for (a in 0:1) {
      i <- sa(s, a)
      pa <- ifelse(a == 1, pi[s], 1 - pi[s])
      b[i] <- (1 - gamma) * p_e[s] * pa
      for (sp in 1:nS) for (ap in 0:1)
        A[i, sa(sp, ap)] <- A[i, sa(sp, ap)] - gamma * P[sp, s, ap + 1] * pa
    }
    pmax(solve(A, b), 1e-10)
  }

  ## Estimate \bar p_T^b(s,a) from the pooled offline data distribution.
  ## Since A is latent, use EM responsibilities eta_it(a) as soft counts.
  p_bar_b <- numeric(dim_sa)
  for (i in seq_along(S)) {
    s <- S[i]
    for (a in 0:1) p_bar_b[sa(s, a)] <- p_bar_b[sa(s, a)] + em_out$eta[i, a + 1]
  }
  p_bar_b <- p_bar_b / length(S)
  p_bar_b <- pmax(p_bar_b, 1e-10)
  p_bar_b <- p_bar_b / sum(p_bar_b)

  d_pi <- build_discounted_target(em_out$P_hat)

  omega <- matrix(0, nS, 2)
  for (s in 1:nS) for (a in 0:1) omega[s, a + 1] <- d_pi[sa(s, a)] / p_bar_b[sa(s, a)]
  omega
}

# ==============================================================================
# 6. Solve for Q^pi (tabular, using EM-estimated transition & reward)
# ==============================================================================
solve_Q_tabular <- function(em_out, dgp, gamma) {
  nS <- dgp$nS
  pi <- dgp$pi_policy

  sa <- function(s, a) a * nS + s
  dim_sa <- nS * 2

  A <- diag(dim_sa)
  b <- numeric(dim_sa)
  for (s in 1:nS) for (a in 0:1) {
    i <- sa(s, a)
    b[i] <- em_out$theta_R_hat[s, a + 1]
    for (sp in 1:nS) for (ap in 0:1) {
      pa <- ifelse(ap == 1, pi[sp], 1 - pi[sp])
      A[i, sa(sp, ap)] <- A[i, sa(sp, ap)] -
        gamma * em_out$P_hat[s, sp, a + 1] * pa
    }
  }
  Q_vec <- solve(A, b)
  Q <- matrix(0, nS, 2)
  for (s in 1:nS) for (a in 0:1) Q[s, a + 1] <- Q_vec[sa(s, a)]
  Q
}

# ==============================================================================
# 7. Multiply robust (MR) estimator (no cross-fitting)
# ==============================================================================
mr_estimator <- function(dat, dgp, gamma, K = NULL) {
  N  <- nrow(dat$S);  TT <- ncol(dat$S)
  nS <- dgp$nS;  pi <- dgp$pi_policy

  ## EM on all data
  em <- em_tabular(dat, nS, gamma)

  ## Nuisance functions
  omega <- solve_omega_tabular(em, dat, dgp, gamma)
  Q     <- solve_Q_tabular(em, dgp, gamma)
  V     <- (1 - pi) * Q[, 1] + pi * Q[, 2]

  ## M V^pi(s,a) = Σ_{s'} P̂(s'|s,a) V(s')
  MV <- matrix(0, nS, 2)
  for (a in 0:1) MV[, a + 1] <- em$P_hat[, , a + 1] %*% V

  tR  <- em$theta_R_hat
  tAt <- em$theta_At_hat0
  tSp <- em$theta_Sp_hat

  direct <- sum(dgp$p_e * V)

  ## Evaluate IF on all data
  phi_all <- numeric(N * TT)
  for (i in 1:N) for (t in 1:TT) {
    s  <- dat$S[i, t]
    at <- dat$Atilde[i, t]
    r  <- dat$R[i, t]
    sp <- dat$Sprime[i, t]

    T1 <- 0; T2 <- 0
    for (a in 0:1) {
      oa <- 1 - a

      d_At <- tAt[s, a + 1] - tAt[s, oa + 1]
      d_At <- ifelse(abs(d_At) < 1e-10, sign(d_At + 1e-20) * 1e-10, d_At)
      br_At <- (at - tAt[s, oa + 1]) / d_At

      d_R  <- tR[s, a + 1] - tR[s, oa + 1]
      d_R  <- ifelse(abs(d_R) < 1e-10, sign(d_R + 1e-20) * 1e-10, d_R)
      br_R <- (r - tR[s, oa + 1]) / d_R

      d_Sp <- tSp[s, a + 1] - tSp[s, oa + 1]
      d_Sp <- ifelse(abs(d_Sp) < 1e-10, sign(d_Sp + 1e-20) * 1e-10, d_Sp)
      br_Sp <- (sp - tSp[s, oa + 1]) / d_Sp

      ga  <- br_At * br_R
      gap <- br_At * br_Sp

      T1 <- T1 + gap * omega[s, a + 1] * (r  - tR[s, a + 1])
      T2 <- T2 + ga  * omega[s, a + 1] * (V[sp] - MV[s, a + 1])
    }
    phi_all[(i - 1) * TT + t] <- T1 / (1 - gamma) + T2 * gamma / (1 - gamma)
  }

  V_hat <- direct + mean(phi_all)
  se    <- sqrt(var(phi_all) / (N * TT))

  list(V_hat = V_hat, se = se,
       ci_lo = V_hat - 1.96 * se,
       ci_hi = V_hat + 1.96 * se)
}

# ==============================================================================
# 8. Format data for Methods.R (stacked by time period)
# ==============================================================================
stack_for_methods <- function(dat) {
  N <- nrow(dat$S);  TT <- ncol(dat$S)
  S_vec <- A_vec <- R_vec <- numeric(N * TT)
  for (t in 1:TT) {
    idx <- ((t - 1) * N + 1):(t * N)
    S_vec[idx] <- dat$S[, t]
    A_vec[idx] <- dat$Atilde[, t]          # naive: use surrogate
    R_vec[idx] <- dat$R[, t]
  }
  list(S = data.frame(S = S_vec), A = A_vec, R = R_vec, H = TT)
}

# ==============================================================================
# 9. Run one replication and return all estimates
# ==============================================================================
one_rep <- function(dgp, N, TT, epsilon, gamma) {
  nS <- dgp$nS;  pi <- dgp$pi_policy

  dat <- generate_data(dgp, N, TT, epsilon)

  ## ---- tabular basis for Methods.R (one-hot for (s,a)) ----
  phi_tab <- function(S_df, A) {
    s <- if (is.data.frame(S_df)) S_df$S else S_df
    n <- length(s)
    Phi <- matrix(0, n, nS * 2)
    for (i in 1:n) Phi[i, A[i] * nS + s[i]] <- 1
    Phi
  }
  pi1_func <- function(S_df) {
    s <- if (is.data.frame(S_df)) S_df$S else S_df
    pi[s]
  }

  st <- stack_for_methods(dat)

  ## --- naive baselines (FQE, MIS, DRL via Methods.R) ---
  V_fqe <- V_mis <- V_drl <- NA
  tryCatch({
    out <- DRL(S = st$S, A = st$A, R = st$R, H = st$H,
               pi1 = pi1_func, phi = phi_tab,
               gamma = gamma, fqe_iter = 30, mis_ridge = 1e-4)
    V_fqe <- out$Vhat_FQE
    V_mis <- out$Vhat_MIS
    V_drl <- out$Vhat_DRL
  }, error = function(e) NULL)

  ## --- SIS via Methods.R ---
  V_sis <- tryCatch(
    SIS(dat$S, dat$Atilde, dat$R, pi1 = pi1_func, gamma = gamma),
    error = function(e) NA
  )

  ## --- our MR estimator ---
  mr <- tryCatch(mr_estimator(dat, dgp, gamma, K = 5), error = function(e)
    list(V_hat = NA, se = NA, ci_lo = NA, ci_hi = NA))

  c(FQE = V_fqe, SIS = V_sis, MIS = V_mis, DRL = V_drl,
    MR = mr$V_hat, MR_se = mr$se, MR_lo = mr$ci_lo, MR_hi = mr$ci_hi)
}

# ==============================================================================
# 11. Full simulation driver
# ==============================================================================
run_experiment <- function(dgp, N, TT, epsilon, gamma, n_rep = 500) {
  V_true <- compute_true_value(dgp, gamma)$V_value
  cat(sprintf("Setting: N=%d  T=%d  eps=%.1f  |  True V(pi)=%.4f\n",
              N, TT, epsilon, V_true))

  res <- matrix(NA, n_rep, 8)
  colnames(res) <- c("FQE","SIS","MIS","DRL","MR","MR_se","MR_lo","MR_hi")

  for (rep in 1:n_rep) {
    if (rep %% 50 == 0) cat("  rep", rep, "/", n_rep, "\n")
    res[rep, ] <- one_rep(dgp, N, TT, epsilon, gamma)
  }

  methods <- c("FQE","SIS","MIS","DRL","MR")
  bias <- colMeans(res[, methods], na.rm = TRUE) - V_true
  rmse <- sqrt(colMeans((res[, methods] - V_true)^2, na.rm = TRUE))
  mr_cov <- mean(res[, "MR_lo"] <= V_true & V_true <= res[, "MR_hi"], na.rm = TRUE)

  out <- data.frame(Method = methods,
                    Bias = round(bias, 4),
                    RMSE = round(rmse, 4),
                    Coverage = c(NA, NA, NA, NA, round(mr_cov, 3)))
  cat("----\n"); print(out, row.names = FALSE); cat("\n")

  list(raw = res, summary = out, V_true = V_true)
}

# ==============================================================================
# 12. Main: run all experiments
# ==============================================================================
dgp   <- generate_dgp()
gamma <- 0.9

cat("\n*** True V(pi) =", compute_true_value(dgp, gamma)$V_value, "***\n\n")

## ---------- Experiment 1: varying epsilon ----------
cat("======== Experiment 1: Varying misclassification rate ========\n")
exp1 <- list()
for (eps in c(0, 0.1, 0.2, 0.3)) {
  cat("\n--- epsilon =", eps, "---\n")
  exp1[[as.character(eps)]] <- run_experiment(dgp, N = 200, TT = 50,
                                              epsilon = eps, gamma = gamma,
                                              n_rep = 500)
}

## ---------- Experiment 2: varying N ----------
cat("\n======== Experiment 2: Varying sample size N ========\n")
exp2 <- list()
for (nn in c(100, 200, 500, 1000)) {
  cat("\n--- N =", nn, "---\n")
  exp2[[as.character(nn)]] <- run_experiment(dgp, N = nn, TT = 50,
                                             epsilon = 0.2, gamma = gamma,
                                             n_rep = 500)
}

## ---------- Save results ----------
save(dgp, gamma, exp1, exp2, file = "sim_tabular_results.RData")
cat("\nResults saved to sim_tabular_results.RData\n")


# ==============================================================================
# 13. Format results into LaTeX tables
# ==============================================================================
format_latex_table1 <- function(exp1_list, eps_vals) {
  cat("\n% --- Table 1: Varying epsilon ---\n")
  methods <- c("FQE","SIS","MIS","DRL","MR")
  for (m in methods) {
    row <- m
    for (eps in eps_vals) {
      s <- exp1_list[[as.character(eps)]]$summary
      row <- paste0(row, " & ",
                    sprintf("%.3f", s$Bias[s$Method == m]), " & ",
                    sprintf("%.3f", s$RMSE[s$Method == m]), " & ")
      if (m == "MR") {
        row <- paste0(row, sprintf("%.2f", s$Coverage[s$Method == m]))
      } else {
        row <- paste0(row, "--")
      }
    }
    cat(row, "\\\\\n")
  }
}

format_latex_table2 <- function(exp2_list, n_vals) {
  cat("\n% --- Table 2: Varying N ---\n")
  methods <- c("FQE","SIS","MIS","DRL","MR")
  for (m in methods) {
    row <- m
    for (nn in n_vals) {
      s <- exp2_list[[as.character(nn)]]$summary
      row <- paste0(row, " & ",
                    sprintf("%.3f", s$Bias[s$Method == m]), " & ",
                    sprintf("%.3f", s$RMSE[s$Method == m]), " & ")
      if (m == "MR") {
        row <- paste0(row, sprintf("%.2f", s$Coverage[s$Method == m]))
      } else {
        row <- paste0(row, "--")
      }
    }
    cat(row, "\\\\\n")
  }
}

format_latex_table1(exp1, c(0, 0.1, 0.2, 0.3))
format_latex_table2(exp2, c(100, 200, 500, 1000))
