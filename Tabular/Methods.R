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

.coerce_tabular_state <- function(S) {
  if (is.data.frame(S)) {
    if (ncol(S) != 1) {
      stop("Tabular MIS/FQE currently require S to be a vector or a one-column data frame.")
    }
    return(S[[1]])
  }
  S
}

# ==============================================================================
# MIS — Marginalized Importance Sampling (standalone)
# ==============================================================================

# MIS <- function(S, A, R, H, phi, pi1 = 0.8, gamma, ridge = 1e-4) {
#   N <- length(A) / H
#   
#   if (length(S) != length(A) || length(R) != length(A)) {
#     stop("S, A, and R must have the same length.")
#   }
#   if (N != floor(N)) {
#     stop("length(A) must be divisible by H.")
#   }
#   
#   # time-major ordering:
#   # time 1: 1..N, time 2: (N+1)..2N, ..., time H: ((H-1)N+1)..HN
#   idx_t    <- 1:((H - 1) * N)
#   idx_tp1  <- (N + 1):(H * N)
#   idx_init <- 1:N
#   
#   # observed-action features phi(S_t, A_t)
#   Phi_t <- phi(S[idx_t], A[idx_t])
#   if (!is.matrix(Phi_t)) stop("phi() must return a matrix.")
#   
#   # evaluation-policy averaged features at next states
#   pi_tp1 <- .resolve_pi(pi1, S[idx_tp1])
#   Phi_tp1_pi <- sweep(phi(S[idx_tp1], 1), 1, pi_tp1, `*`) +
#     sweep(phi(S[idx_tp1], 0), 1, 1 - pi_tp1, `*`)
#   
#   if (!all(dim(Phi_tp1_pi) == dim(Phi_t))) {
#     stop("phi() returned inconsistent dimensions.")
#   }
#   
#   # empirical initial-state expectation under evaluation policy
#   pi_init <- .resolve_pi(pi1, S[idx_init])
#   Phi_init_pi <- sweep(phi(S[idx_init], 1), 1, pi_init, `*`) +
#     sweep(phi(S[idx_init], 0), 1, 1 - pi_init, `*`)
#   mu0 <- colMeans(Phi_init_pi)
#   
#   # linear MIS moment equation
#   PP <- crossprod(Phi_t, Phi_t - gamma * Phi_tp1_pi) / (N * (H - 1))
#   if (ridge > 0) {
#     PP <- PP + ridge * diag(ncol(PP))
#   }
#   
#   rhs <- drop((1 - gamma) * mu0)
#   beta <- qr.solve(PP, rhs)
#   
#   # density-ratio estimate on all observed samples
#   w_hat <- drop(phi(S, A) %*% beta)
#   
#   # plug-in value estimate
#   V_hat <- mean(w_hat * R) / (1 - gamma)
#   
#   list(V_hat = V_hat, omega_hat = w_hat, beta = beta)
# }

MIS <- function(S, A, R, H, phi = NULL, pi1 = 0.8, gamma, ridge = 1e-8) {
  S <- .coerce_tabular_state(S)
  N <- length(A) / H
  
  if (length(S) != length(A) || length(R) != length(A)) {
    stop("S, A, and R must have the same length.")
  }
  if (N != floor(N)) {
    stop("length(A) must be divisible by H.")
  }
  
  state_levels <- sort(unique(S))
  state_labels <- as.character(state_levels)
  S_chr <- as.character(S)
  nS <- length(state_levels)
  actions <- c(0, 1)
  nA <- 2
  K <- nS * nA
  
  state_to_idx <- setNames(seq_along(state_levels), state_labels)
  
  sa_index <- function(s, a) {
    (state_to_idx[as.character(s)] - 1L) * nA + (a + 1L)
  }
  
  idx_t   <- 1:((H - 1) * N)
  idx_tp1 <- (N + 1):(H * N)
  idx_all <- 1:(H * N)
  idx_init <- 1:N
  
  # Evaluation-policy probabilities
  pi_prob <- function(s, a) {
    p1 <- .resolve_pi(pi1, s)
    ifelse(a == 1, p1, 1 - p1)
  }
  
  # Build A matrix and b vector in A %*% omega = b
  A_mat <- matrix(0, nrow = K, ncol = K)
  b_vec <- numeric(K)
  
  # Empirical initial distribution p_e(s) from t = 1
  p_init <- table(factor(S_chr[idx_init], levels = state_labels)) / N
  
  # b = (1-gamma) * mu0^pi
  for (s in state_levels) {
    for (a in actions) {
      j <- sa_index(s, a)
      b_vec[j] <- (1 - gamma) * p_init[as.character(s)] * pi_prob(s, a)
    }
  }
  
  # A[j, k] corresponds to coefficient of omega(s,a) in moment j=(s',a')
  for (m in seq_along(idx_t)) {
    it  <- idx_t[m]
    it1 <- idx_tp1[m]
    
    s  <- S[it]
    a  <- A[it]
    sp <- S[it1]
    
    k <- sa_index(s, a)
    
    # current term: -1{(S_t,A_t)=(s',a')}
    j_current <- sa_index(s, a)
    A_mat[j_current, k] <- A_mat[j_current, k] + 1 / (N * (H - 1))
    
    # next-state policy-averaged term: gamma * 1{S_{t+1}=s'} pi(a'|s')
    for (ap in actions) {
      j_next <- sa_index(sp, ap)
      A_mat[j_next, k] <- A_mat[j_next, k] - 
        gamma * pi_prob(sp, ap) / (N * (H - 1))
    }
  }
  
  if (ridge > 0) {
    A_mat <- A_mat + ridge * diag(K)
  }
  
  omega_vec <- solve(A_mat, b_vec)
  
  # evaluate omega_hat on all observed samples
  w_hat <- numeric(length(A))
  for (i in seq_along(A)) {
    w_hat[i] <- omega_vec[sa_index(S[i], A[i])]
  }
  
  V_hat <- mean(w_hat * R) / (1 - gamma)
  
  list(
    V_hat = V_hat,
    omega_hat = w_hat,
    omega_sa = omega_vec,
    states = state_levels,
    phi_ignored = !is.null(phi)
  )
}




# ==============================================================================
# FQE — Fitted Q-Evaluation
# ==============================================================================
FQE <- function(S, A, R, H, pi1, gamma, fqe_iter = 50) {

  S <- as.data.frame(S)
  N <- nrow(S) / H

  is_tabular <- ncol(S) == 1

  if (is_tabular) {
    S_vec <- S[[1]]
    state_levels <- sort(unique(S_vec))
    state_labels <- as.character(state_levels)
    state_to_idx <- setNames(seq_along(state_levels), state_labels)
    nS <- length(state_levels)
    nA <- 2

    sa_index <- function(s, a) {
      (state_to_idx[as.character(s)] - 1L) * nA + (a + 1L)
    }

    Q_sa <- numeric(nS * nA)
    Q0_all <- Q1_all <- numeric(H * N)

    for (it in 1:fqe_iter) {
      y <- numeric(H * N)

      for (h in 1:H) {
        idx <- ((h - 1) * N + 1):(h * N)

        if (h == H) {
          y[idx] <- R[idx]
        } else {
          idxn <- (h * N + 1):((h + 1) * N)
          p1n  <- .resolve_pi(pi1, S[idxn, , drop = FALSE])

          Q0_next <- numeric(N)
          Q1_next <- numeric(N)
          for (i in seq_along(idxn)) {
            sn <- S_vec[idxn[i]]
            Q0_next[i] <- Q_sa[sa_index(sn, 0)]
            Q1_next[i] <- Q_sa[sa_index(sn, 1)]
          }

          Vn <- (1 - p1n) * Q0_next + p1n * Q1_next
          y[idx] <- R[idx] + gamma * Vn
        }
      }

      Q_new <- Q_sa
      counts <- numeric(length(Q_sa))

      for (i in seq_along(A)) {
        key <- sa_index(S_vec[i], A[i])
        Q_new[key] <- Q_new[key] + y[i]
        counts[key] <- counts[key] + 1
      }

      seen <- counts > 0
      Q_new[seen] <- Q_new[seen] / counts[seen]
      Q_sa <- Q_new

      for (i in seq_along(S_vec)) {
        Q0_all[i] <- Q_sa[sa_index(S_vec[i], 0)]
        Q1_all[i] <- Q_sa[sa_index(S_vec[i], 1)]
      }
    }

    predict_Q <- function(S_new, A_new) {
      S_new_df <- as.data.frame(S_new)
      if (ncol(S_new_df) != 1) {
        stop("Tabular FQE prediction requires a vector or a one-column data frame.")
      }
      S_new_vec <- S_new_df[[1]]
      out <- numeric(length(A_new))
      for (i in seq_along(A_new)) {
        out[i] <- Q_sa[sa_index(S_new_vec[i], A_new[i])]
      }
      out
    }

    Q_obs <- predict_Q(S, A)

    s1   <- S[1:N, , drop = FALSE]
    p1_1 <- .resolve_pi(pi1, s1)
    Q0_1 <- predict_Q(s1, rep(0, N))
    Q1_1 <- predict_Q(s1, rep(1, N))
    V_hat <- mean((1 - p1_1) * Q0_1 + p1_1 * Q1_1)

    return(list(
      V_hat = V_hat,
      q_model = NULL,
      Q_obs = Q_obs,
      Q0_all = Q0_all,
      Q1_all = Q1_all,
      predict_Q = predict_Q,
      q_table = Q_sa,
      states = state_levels,
      model_type = "tabular"
    ))
  }

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

  predict_Q <- function(S_new, A_new) {
    S_new_df <- as.data.frame(S_new)
    predict(q_model, data.frame(S_new_df, A = A_new))
  }

  # V_hat from initial states (t = 1 block)
  s1   <- S[1:N, , drop = FALSE]
  p1_1 <- .resolve_pi(pi1, s1)
  Q0_1 <- predict_Q(s1, rep(0, N))
  Q1_1 <- predict_Q(s1, rep(1, N))
  V_hat <- mean((1 - p1_1) * Q0_1 + p1_1 * Q1_1)

  list(V_hat = V_hat, q_model = q_model, Q_obs = Q_obs,
       Q0_all = Q0_all, Q1_all = Q1_all,
       predict_Q = predict_Q, model_type = "rpart")
}

# ==============================================================================
# DRL — Double Reinforcement Learning  (FQE + MIS)
# ==============================================================================
DRL <- function(S, A, R, H, pi1, phi = NULL, gamma,
                fqe_iter = 50, mis_ridge = 1e-4) {
  
  S <- as.data.frame(S)
  N <- nrow(S) / H
  
  # ---- FQE component ----
  fqe <- FQE(S, A, R, H, pi1, gamma, fqe_iter)
  Vhat_FQE <- fqe$V_hat
  q_model  <- fqe$q_model
  predict_Q <- fqe$predict_Q
  Q_obs    <- fqe$Q_obs
  
  idx_prev <- 1:((H - 1) * N)
  idx_next <- (N + 1):(H * N)
  
  # Q(S_{t+1}, pi)
  p1_next  <- .resolve_pi(pi1, S[idx_next, , drop = FALSE])
  Q0_next  <- predict_Q(S[idx_next, , drop = FALSE], rep(0, length(idx_next)))
  Q1_next  <- predict_Q(S[idx_next, , drop = FALSE], rep(1, length(idx_next)))
  Q_next_pi <- (1 - p1_next) * Q0_next + p1_next * Q1_next
  
  # ---- MIS component ----
  mis <- MIS(S, A, R, H, phi = phi, pi1 = pi1, gamma = gamma, ridge = mis_ridge)
  
  omega_hat <- mis$omega_hat
  Vhat_MIS  <- mis$V_hat
  omega_sa  <- mis$omega_sa
  states    <- mis$states
  
  # ---- DRL value ----
  dr_inner <- R[idx_prev] + gamma * Q_next_pi - Q_obs[idx_prev]
  dr_correction <- mean(omega_hat[idx_prev] * dr_inner) / (1 - gamma)
  Vhat_DRL <- dr_correction + Vhat_FQE
  
  list(
    Vhat_DRL   = Vhat_DRL,
    Vhat_FQE   = Vhat_FQE,
    Vhat_MIS   = Vhat_MIS,
    dr_correction = dr_correction,
    mis_drl_gap = Vhat_MIS - Vhat_DRL,
    omega_hat  = omega_hat,
    omega_sa   = omega_sa,
    q_model    = q_model,
    predict_Q  = predict_Q,
    Q_obs      = Q_obs,
    Q_next_pi  = Q_next_pi,
    states     = states,
    fqe_model_type = fqe$model_type,
    mis_phi_ignored = mis$phi_ignored
  )
}
# ==============================================================================
# SIS — Sequential Importance Sampling
# ==============================================================================
SIS <- function(S_mat, A_mat, R_mat, pi1, gamma, b_hat = NULL,
                min_prob = 0.05, max_ratio = 8, max_weight = 20,
                normalize = TRUE) {
  # S_mat, A_mat, R_mat: N x T matrices (one row per trajectory)
  S_mat <- as.matrix(S_mat)
  A_mat <- as.matrix(A_mat)
  R_mat <- as.matrix(R_mat)

  N  <- nrow(S_mat)
  TT <- ncol(S_mat)

  if (!all(dim(A_mat) == c(N, TT)) || !all(dim(R_mat) == c(N, TT))) {
    stop("S_mat, A_mat, and R_mat must have identical dimensions.")
  }

  # Estimate behavior policy from pooled data if not supplied.
  if (is.null(b_hat)) {
    Sv <- as.vector(S_mat)
    Av <- as.vector(A_mat)
    state_levels <- sort(unique(Sv))
    b_lookup <- tapply(Av, factor(Sv, levels = state_levels), mean)
    names(b_lookup) <- as.character(state_levels)
  } else {
    b_lookup <- b_hat
    if (!is.function(b_lookup) && is.null(names(b_lookup))) {
      names(b_lookup) <- as.character(seq_along(b_lookup))
    }
  }

  get_b1 <- function(s) {
    if (is.function(b_hat)) {
      b1 <- b_hat(s)
    } else {
      key <- as.character(s)
      if (key %in% names(b_lookup)) {
        b1 <- unname(b_lookup[key])
      } else if (is.numeric(s) && s >= 1 && s <= length(b_lookup)) {
        b1 <- b_lookup[as.integer(s)]
      } else {
        stop("Missing behavior-policy estimate for at least one observed state.")
      }
    }
    pmax(pmin(b1, 1 - min_prob), min_prob)
  }

  cum_w <- matrix(1, nrow = N, ncol = TT)
  for (t in seq_len(TT)) {
    s_t <- S_mat[, t]
    a_t <- A_mat[, t]

    p1_t <- .resolve_pi(pi1, s_t)
    pi_a <- ifelse(a_t == 1, p1_t, 1 - p1_t)
    b1_t <- vapply(s_t, get_b1, numeric(1))
    b_a  <- ifelse(a_t == 1, b1_t, 1 - b1_t)

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
  if (normalize) {
    est_t <- numeric(TT)
    for (t in seq_len(TT)) {
      denom <- sum(cum_w[, t])
      if (!is.finite(denom) || denom <= 0) next
      est_t[t] <- sum(cum_w[, t] * R_mat[, t]) / denom
    }
    return(sum(gamma_t * est_t))
  }

  mean(rowSums(cum_w * matrix(gamma_t, nrow = N, ncol = TT, byrow = TRUE) * R_mat))
}

# ==============================================================================
# LSTD — Least-Squares Temporal Difference (tabular)
# ==============================================================================
LSTD <- function(S, A, R, H, pi1, gamma, ridge = 1e-6) {
  S <- .coerce_tabular_state(S)
  N <- length(A) / H

  state_levels <- sort(unique(S))
  state_labels <- as.character(state_levels)
  nS <- length(state_levels)
  nA <- 2
  d  <- nS * nA

  state_to_idx <- setNames(seq_along(state_levels), state_labels)

  ## Feature vector φ(s,a): indicator in R^{nS*nA}
  phi_vec <- function(s, a) {
    v <- numeric(d)
    v[(state_to_idx[as.character(s)] - 1L) * nA + (a + 1L)] <- 1
    v
  }

  ## Build feature matrix for all observations
  n <- length(A)
  Phi_obs <- matrix(0, n, d)
  for (i in seq_len(n)) {
    Phi_obs[i, ] <- phi_vec(S[i], A[i])
  }

  ## Policy-averaged feature at next state: φ_π(s') = Σ_a π(a|s') φ(s',a)
  ## Next-state indices: for t=1..H-1
  idx_t   <- 1:((H - 1) * N)
  idx_tp1 <- (N + 1):(H * N)

  Phi_t <- Phi_obs[idx_t, , drop = FALSE]
  R_t   <- R[idx_t]
  S_tp1 <- S[idx_tp1]

  pi_tp1 <- .resolve_pi(pi1, S_tp1)
  PhiPi_tp1 <- matrix(0, length(idx_tp1), d)
  for (i in seq_along(idx_tp1)) {
    PhiPi_tp1[i, ] <- (1 - pi_tp1[i]) * phi_vec(S_tp1[i], 0) +
                       pi_tp1[i]       * phi_vec(S_tp1[i], 1)
  }

  ## LSTD: A w = b
  n_pairs <- length(idx_t)
  A_mat <- crossprod(Phi_t, Phi_t - gamma * PhiPi_tp1) / n_pairs
  b_vec <- drop(crossprod(Phi_t, R_t)) / n_pairs

  w_hat <- solve(A_mat + ridge * diag(d), b_vec)

  ## V_hat = (1/N) Σ_i V(s_{i,1})  where V(s) = Σ_a π(a|s) Q(s,a)
  S_init  <- S[1:N]
  pi_init <- .resolve_pi(pi1, S_init)
  V_init  <- numeric(N)
  for (i in 1:N) {
    q0 <- sum(phi_vec(S_init[i], 0) * w_hat)
    q1 <- sum(phi_vec(S_init[i], 1) * w_hat)
    V_init[i] <- (1 - pi_init[i]) * q0 + pi_init[i] * q1
  }
  V_hat <- mean(V_init)

  list(V_hat = V_hat, w = w_hat, states = state_levels)
}

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
  
  relabel_by_surrogate <- function(eta, b_hat, mu_hat, theta_R_hat,
                                   sigma_R_hat, P_hat) {
    class_prob_tilde1 <- numeric(2)
    for (k in 1:2) {
      sw <- sum(eta[, k])
      if (sw > 1e-10) class_prob_tilde1[k] <- sum(eta[, k] * At) / sw
    }
    
    if (class_prob_tilde1[2] >= class_prob_tilde1[1]) {
      return(list(
        eta = eta,
        b_hat = b_hat,
        mu_hat = mu_hat,
        theta_R_hat = theta_R_hat,
        sigma_R_hat = sigma_R_hat,
        P_hat = P_hat
      ))
    }
    
    list(
      eta = eta[, 2:1, drop = FALSE],
      b_hat = 1 - b_hat,
      mu_hat = mu_hat[, 2:1, drop = FALSE],
      theta_R_hat = theta_R_hat[, 2:1, drop = FALSE],
      sigma_R_hat = sigma_R_hat[, 2:1, drop = FALSE],
      P_hat = P_hat[, , 2:1, drop = FALSE]
    )
  }
  
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
  
  relabeled <- relabel_by_surrogate(
    eta = eta,
    b_hat = b_hat,
    mu_hat = mu_hat,
    theta_R_hat = theta_R_hat,
    sigma_R_hat = sigma_R_hat,
    P_hat = P_hat
  )
  eta <- relabeled$eta
  b_hat <- relabeled$b_hat
  mu_hat <- relabeled$mu_hat
  theta_R_hat <- relabeled$theta_R_hat
  sigma_R_hat <- relabeled$sigma_R_hat
  P_hat <- relabeled$P_hat
  
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
  tAt <- em$theta_At_hat
  tSp <- em$theta_Sp_hat
  
  direct <- sum(dgp$p_e * V)
  
  ## Evaluate IF and IS on all data (single loop)
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
    idx <- (i - 1) * TT + t
    phi_all[idx] <- T1 / (1 - gamma) + T2 * gamma / (1 - gamma)
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

one_rep <- function(dgp, N, TT, epsilon, gamma) {
  nS <- dgp$nS
  pi <- dgp$pi_policy
  
  dat <- generate_data(dgp, N, TT, epsilon)
  
  ## ---- tabular basis (kept for compatibility with FQE / DRL interface) ----
  phi_tab <- function(S_df, A) {
    s <- if (is.data.frame(S_df)) S_df[[1]] else S_df
    if (!all(A %in% c(0, 1))) stop("A must be coded as 0/1.")
    if (!all(s %in% seq_len(nS))) stop("States must be coded as 1,...,nS.")
    
    n <- length(s)
    Phi <- matrix(0, n, nS * 2)
    col_idx <- A * nS + s
    Phi[cbind(seq_len(n), col_idx)] <- 1
    Phi
  }
  
  pi1_func <- function(S_df) {
    s <- if (is.data.frame(S_df)) S_df[[1]] else S_df
    pi[s]
  }
  
  st <- stack_for_methods(dat)

  ## --- clipped SIS ---
  V_sis <- tryCatch(
    SIS(S_mat = dat$S, A_mat = dat$Atilde, R_mat = dat$R,
        pi1 = pi1_func, gamma = gamma),
    error = function(e) NA_real_
  )
  
  ## --- naive baselines (FQE, MIS, DRL via Methods.R) ---
  V_fqe <- V_mis <- V_drl <- NA
  tryCatch({
    out <- DRL(
      S = st$S, A = st$A, R = st$R, H = st$H,
      pi1 = pi1_func, phi = phi_tab,
      gamma = gamma, fqe_iter = 30, mis_ridge = 1e-4
    )
    V_fqe <- out$Vhat_FQE
    V_mis <- out$Vhat_MIS
    V_drl <- out$Vhat_DRL
  }, error = function(e) NULL)
  
  ## --- naive LSTD ---
  V_lstd <- tryCatch(
    LSTD(S = st$S, A = st$A, R = st$R, H = st$H,
         pi1 = pi1_func, gamma = gamma)$V_hat,
    error = function(e) NA
  )
  
  ## --- our MR estimator ---
  mr <- tryCatch(
    mr_estimator(dat, dgp, gamma, K = 5),
    error = function(e) list(V_hat = NA, ci_lo = NA, ci_hi = NA)
  )
  
  c(
    FQE      = V_fqe,
    SIS      = V_sis,
    MIS      = V_mis,
    DRL      = V_drl,
    LSTD     = V_lstd,
    MR       = mr$V_hat,
    MR_ci_lo = mr$ci_lo,
    MR_ci_hi = mr$ci_hi
  )
}

