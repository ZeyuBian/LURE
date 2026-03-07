################################################################################
## Competing OPE Methods (naive — treat surrogate as true action)
##
##   MIS:  Marginalized importance sampling (standalone)
##   FQE:  Fitted Q-evaluation
##   DRL:  Double reinforcement learning (FQE + MIS)
##   SIS:  Sequential importance sampling
##
## Data layout: S, A, R are vectors of length N*H, stacked time-by-time:
##   rows 1..N = t=1,  rows N+1..2N = t=2,  …,  rows (H-1)*N+1..H*N = t=H.
## phi(S, A) returns an n x d basis matrix for the density-ratio model.
## pi1: either a scalar P(A=1) (constant policy) or a function(S_df) -> vector.
################################################################################

# ==============================================================================
# Helper: resolve pi1 to a probability vector
# ==============================================================================
.resolve_pi <- function(pi1, S_sub) {
  if (is.function(pi1)) pi1(S_sub) else rep(pi1, NROW(S_sub))
}

# ==============================================================================
# MIS — Marginalized Importance Sampling (standalone)
# ==============================================================================
MIS <- function(S, A, R, H, phi, pi1 = 0.8, gamma, ridge = 1e-4) {

  N <- length(A) / H

  idx_prev <- 1:((H - 1) * N)
  idx_next <- (N + 1):(H * N)

  Phi_prev <- phi(S[idx_prev], A[idx_prev])
  Phi_next <- phi(S[idx_next], A[idx_next])

  pi_prev <- .resolve_pi(pi1, S[idx_prev])
  pi_next <- .resolve_pi(pi1, S[idx_next])

  p_prev <- pi_prev * A[idx_prev] + (1 - pi_prev) * (1 - A[idx_prev])
  p_next <- pi_next * A[idx_next] + (1 - pi_next) * (1 - A[idx_next])

  D_prev <- Phi_prev * p_prev
  D_next <- Phi_next * p_next

  PP <- t(Phi_prev) %*% Phi_prev - gamma * t(Phi_prev) %*% D_next
  if (ridge > 0) PP <- PP + ridge * diag(ncol(PP))

  beta    <- (1 - gamma) * solve(PP, t(D_prev) %*% R[idx_prev])
  w_hat   <- as.numeric(phi(S, A) %*% beta)
  V_hat   <- mean(w_hat * R) / (1 - gamma)

  list(V_hat = V_hat, omega_hat = w_hat, beta = beta)
}

# ==============================================================================
# FQE — Fitted Q-Evaluation
# ==============================================================================
FQE <- function(S, A, R, H, pi1, gamma, fqe_iter = 50) {

  S <- as.data.frame(S)
  N <- nrow(S) / H

  Q0_all <- Q1_all <- rep(0, nrow(S))

  for (it in 1:fqe_iter) {
    y <- numeric(H * N)

    for (h in 1:H) {
      idx <- ((h - 1) * N + 1):(h * N)

      if (h == H) {
        y[idx] <- R[idx]
      } else {
        idxn <- (h * N + 1):((h + 1) * N)
        p1n  <- .resolve_pi(pi1, S[idxn, , drop = FALSE])
        Vn   <- (1 - p1n) * Q0_all[idxn] + p1n * Q1_all[idxn]
        y[idx] <- R[idx] + gamma * Vn
      }
    }

    q_model <- rpart::rpart(y ~ ., data = data.frame(y = y, S, A = A),
                             method = "anova")
    Q0_all <- predict(q_model, data.frame(S, A = 0))
    Q1_all <- predict(q_model, data.frame(S, A = 1))
  }

  Q_obs <- predict(q_model, data.frame(S, A = A))

  # V_hat from initial states (t = 1 block)
  s1   <- S[1:N, , drop = FALSE]
  p1_1 <- .resolve_pi(pi1, s1)
  Q0_1 <- predict(q_model, data.frame(s1, A = 0))
  Q1_1 <- predict(q_model, data.frame(s1, A = 1))
  V_hat <- mean((1 - p1_1) * Q0_1 + p1_1 * Q1_1)

  list(V_hat = V_hat, q_model = q_model, Q_obs = Q_obs,
       Q0_all = Q0_all, Q1_all = Q1_all)
}

# ==============================================================================
# DRL — Double Reinforcement Learning  (FQE + MIS)
# ==============================================================================
DRL <- function(S, A, R, H, pi1, phi, gamma,
                fqe_iter = 50, mis_ridge = 1e-4) {

  S <- as.data.frame(S)
  N <- nrow(S) / H

  # ---- FQE component ----
  fqe <- FQE(S, A, R, H, pi1, gamma, fqe_iter)
  Vhat_FQE <- fqe$V_hat
  q_model  <- fqe$q_model
  Q_obs    <- fqe$Q_obs

  # Q(S_{t+1}, pi) for t = 1 .. H-1
  idx_next <- (N + 1):(H * N)
  p1_next  <- .resolve_pi(pi1, S[idx_next, , drop = FALSE])
  Q0_next  <- predict(q_model, data.frame(S[idx_next, , drop = FALSE], A = 0))
  Q1_next  <- predict(q_model, data.frame(S[idx_next, , drop = FALSE], A = 1))
  Q_next_pi <- (1 - p1_next) * Q0_next + p1_next * Q1_next

  # ---- MIS component ----
  idx_prev <- 1:((H - 1) * N)

  Phi_prev <- phi(S[idx_prev, , drop = FALSE], A[idx_prev])
  Phi_next <- phi(S[idx_next, , drop = FALSE], A[idx_next])

  pi_prev <- .resolve_pi(pi1, S[idx_prev, , drop = FALSE])
  pi_next <- .resolve_pi(pi1, S[idx_next, , drop = FALSE])

  p_prev <- pi_prev * A[idx_prev] + (1 - pi_prev) * (1 - A[idx_prev])
  p_next <- pi_next * A[idx_next] + (1 - pi_next) * (1 - A[idx_next])

  D_prev <- Phi_prev * p_prev
  D_next <- Phi_next * p_next

  PP <- t(Phi_prev) %*% Phi_prev - gamma * t(Phi_prev) %*% D_next
  if (mis_ridge > 0) PP <- PP + mis_ridge * diag(ncol(PP))

  beta      <- (1 - gamma) * solve(PP, t(D_prev) %*% R[idx_prev])
  omega_hat <- as.numeric(phi(S, A) %*% beta)

  # ---- MIS value ----
  Vhat_MIS <- mean(omega_hat * R) / (1 - gamma)

  # ---- DRL value ----
  dr_inner <- R[idx_prev] + gamma * Q_next_pi - Q_obs[idx_prev]
  Vhat_DRL <- mean(omega_hat[idx_prev] * dr_inner) / (1 - gamma) + Vhat_FQE

  list(
    Vhat_DRL   = Vhat_DRL,
    Vhat_FQE   = Vhat_FQE,
    Vhat_MIS   = Vhat_MIS,
    omega_hat  = omega_hat,
    beta_mis   = beta,
    q_model    = q_model,
    Q_obs      = Q_obs,
    Q_next_pi  = Q_next_pi
  )
}

# ==============================================================================
# SIS — Sequential Importance Sampling
# ==============================================================================
SIS <- function(S_mat, A_mat, R_mat, pi1, gamma, b_hat = NULL) {
  # S_mat, A_mat, R_mat: N x T matrices (one row per trajectory)
  N  <- nrow(S_mat)
  TT <- ncol(S_mat)

  # Estimate behavior policy from data if not supplied
  if (is.null(b_hat)) {
    Sv <- as.vector(S_mat); Av <- as.vector(A_mat)
    states <- sort(unique(Sv))
    b_hat <- numeric(max(states))
    for (s in states) b_hat[s] <- mean(Av[Sv == s])
    b_hat <- pmax(pmin(b_hat, 0.99), 0.01)
  }

  total <- 0
  for (i in 1:N) {
    w <- 1
    for (t in 1:TT) {
      s  <- S_mat[i, t]
      a  <- A_mat[i, t]
      pi_a <- .resolve_pi(pi1, s)
      pi_a <- ifelse(a == 1, pi_a, 1 - pi_a)
      b_a  <- ifelse(a == 1, b_hat[s], 1 - b_hat[s])
      w <- w * (pi_a / b_a)
      total <- total + w * gamma^(t - 1) * R_mat[i, t]
    }
  }
  total / N
}
