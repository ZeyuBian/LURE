################################################################################
## Posterior-imputation (IMP) baseline: helpers shared by the Tabular,
## Continuous, and Gym implementations.
##
## The baseline imputes the latent action from the posterior
##   eta(a | O) ∝ b(a | S) mu(Atilde | S, a) h(R | S, a) q(S' | S, a)
## and then runs standard OPE estimators on the imputed data.
##
##   mu : Zhou and Tchetgen Tchetgen (2024), proof of Theorem 1 for a continuous
##        outcome (Supplement S2), with surrogate Atilde, outcome Y = R, and a
##        binary proxy Z = 1{l(S') > E[l(S') | S]}. Writing (.) for the
##        elementwise product,
##          {E(R | Atilde, Z, s) (.) P(Atilde, Z | s)} P(Atilde, Z | s)^{-1}
##            = P(Atilde | A, s) diag{E(R | A = a, s)} P(Atilde | A, s)^{-1}.
##        The eigenvalues are theta_R(s, a); the eigenvectors, normalized so
##        each column sums to one, are the columns (P(Atilde = 0 | a, s),
##        P(Atilde = 1 | a, s)). The column with the larger P(Atilde = 1 | .)
##        is assigned to A = 1 (surrogate condition).
##   b  : proof of Theorem 1 in the main paper,
##          P_A(s) = P_{Atilde, A}(s)^{-1} P_Atilde(s),
##        i.e. b(1 | s) = {P(Atilde = 1 | s) - mu_0(s)} / {mu_1(s) - mu_0(s)}.
##   h, q : outputs of the EM algorithm (Algorithm 2) fitted to the full data.
################################################################################

## Cells of (Atilde, Z) are ordered 00, 01, 10, 11 (first index is Atilde), so
## columns 1-4 of an n x 4 matrix are the row-major entries of a 2 x 2 matrix
## with rows Atilde = 0, 1 and columns Z = 0, 1.
imp_az_cell <- function(At, Z) 2L * as.integer(At) + as.integer(Z) + 1L

## Vectorized eigendecomposition of M = G P^{-1} for 2 x 2 matrices.
##   p_az : n x 4 matrix of P(Atilde = a, Z = z | s)
##   g_az : n x 4 matrix of E(R | Atilde = a, Z = z, s) P(Atilde = a, Z = z | s)
## Returns mu0/mu1 = P(Atilde = 1 | A = 0/1, s) from the normalized eigenvectors
## and theta0/theta1 = E(R | A = 0/1, s) from the paired eigenvalues.
imp_eigen_measurement <- function(p_az, g_az, clip_lo = 0.01, clip_hi = 0.99) {
  p_az <- matrix(p_az, ncol = 4L)
  g_az <- matrix(g_az, ncol = 4L)
  p00 <- p_az[, 1]; p01 <- p_az[, 2]; p10 <- p_az[, 3]; p11 <- p_az[, 4]
  g00 <- g_az[, 1]; g01 <- g_az[, 2]; g10 <- g_az[, 3]; g11 <- g_az[, 4]

  det_p <- p00 * p11 - p01 * p10
  singular <- !is.finite(det_p) | abs(det_p) < 1e-12
  det_p <- ifelse(singular, ifelse(det_p < 0, -1e-12, 1e-12), det_p)

  ## M = G P^{-1}, with P^{-1} = adj(P) / det(P).
  m11 <- (g00 * p11 - g01 * p10) / det_p
  m12 <- (g01 * p00 - g00 * p01) / det_p
  m21 <- (g10 * p11 - g11 * p10) / det_p
  m22 <- (g11 * p00 - g10 * p01) / det_p

  tr_m <- m11 + m22
  disc <- (m11 - m22)^2 + 4 * m12 * m21  # = tr^2 - 4 det
  complex_root <- !is.finite(disc) | disc < 0
  root <- sqrt(pmax(ifelse(is.finite(disc), disc, 0), 0))
  lambda <- cbind((tr_m + root) / 2, (tr_m - root) / 2)

  ## Eigenvector for lambda: (m12, lambda - m11) or (lambda - m22, m21),
  ## whichever is longer; normalized so its entries sum to one, its second
  ## entry is P(Atilde = 1 | A = a, s).
  pr_at1 <- matrix(NA_real_, nrow(p_az), 2L)
  degenerate <- complex_root
  for (k in 1:2) {
    u1 <- m12; u2 <- lambda[, k] - m11
    w1 <- lambda[, k] - m22; w2 <- m21
    use_u <- u1^2 + u2^2 >= w1^2 + w2^2
    v1 <- ifelse(use_u, u1, w1)
    v2 <- ifelse(use_u, u2, w2)
    total <- v1 + v2
    degenerate <- degenerate | !is.finite(total) | abs(total) < 1e-10
    pr_at1[, k] <- v2 / total
  }
  out_of_range <- !degenerate & (pr_at1[, 1] < 0 | pr_at1[, 1] > 1 |
                                   pr_at1[, 2] < 0 | pr_at1[, 2] > 1)

  ## Surrogate condition: the eigenvector with the larger P(Atilde = 1 | .)
  ## belongs to A = 1; its eigenvalue is E(R | A = 1, s).
  first_is_one <- pr_at1[, 1] >= pr_at1[, 2]
  mu1 <- ifelse(first_is_one, pr_at1[, 1], pr_at1[, 2])
  mu0 <- ifelse(first_is_one, pr_at1[, 2], pr_at1[, 1])
  theta1 <- ifelse(first_is_one, lambda[, 1], lambda[, 2])
  theta0 <- ifelse(first_is_one, lambda[, 2], lambda[, 1])

  ## Points with no real, normalizable eigenvectors take the average over the
  ## remaining points.
  if (any(degenerate)) {
    if (all(degenerate)) stop("Eigendecomposition failed at every state.")
    mu1[degenerate] <- mean(pmin(pmax(mu1[!degenerate], clip_lo), clip_hi))
    mu0[degenerate] <- mean(pmin(pmax(mu0[!degenerate], clip_lo), clip_hi))
    theta1[degenerate] <- mean(theta1[!degenerate])
    theta0[degenerate] <- mean(theta0[!degenerate])
  }

  list(
    mu0 = pmin(pmax(mu0, clip_lo), clip_hi),
    mu1 = pmin(pmax(mu1, clip_lo), clip_hi),
    theta0 = theta0,
    theta1 = theta1,
    singular = singular,
    complex_root = complex_root,
    degenerate = degenerate,
    out_of_range = out_of_range
  )
}

## Closed-form behavior policy from the proof of Theorem 1 (binary case).
imp_behavior_closed_form <- function(p_at1, mu0, mu1, min_gap = 0.05,
                                     clip_lo = 0.01, clip_hi = 0.99) {
  gap <- pmax(mu1 - mu0, min_gap)
  b1 <- (p_at1 - mu0) / gap
  list(b1 = pmin(pmax(b1, clip_lo), clip_hi),
       b1_raw = b1,
       gap_floored = (mu1 - mu0) < min_gap)
}

## Posterior from per-action log-likelihood components (log-sum-exp).
imp_posterior <- function(log_b1, log_b0, log_mu1, log_mu0,
                          log_h1, log_h0, log_q1, log_q0) {
  l1 <- log_b1 + log_mu1 + log_h1 + log_q1
  l0 <- log_b0 + log_mu0 + log_h0 + log_q0
  mx <- pmax(l0, l1)
  e1 <- exp(l1 - mx)
  e0 <- exp(l0 - mx)
  e1 / (e0 + e1)
}

## Draws M imputed action vectors; returns an n x M integer matrix.
imp_draw_actions <- function(eta1, M) {
  n <- length(eta1)
  matrix(as.integer(stats::runif(n * M) < rep(eta1, times = M)), nrow = n, ncol = M)
}

## Conditional models for the (Atilde, Z) cells given state features:
## P(Atilde, Z | s) from a multinomial logit and E(R | Atilde, Z, s) from
## cell-specific least squares, both on the same features. Returns a predictor
## giving p_az and g_az = E(R | cell, s) * P(cell | s) for new feature rows.
imp_fit_cell_model <- function(feature_df, At, Z, R, decay = 1e-3, maxit = 1000L) {
  if (!requireNamespace("nnet", quietly = TRUE)) {
    stop("Package 'nnet' is required for the conditional cell model.")
  }
  cell <- imp_az_cell(At, Z)
  feats <- names(feature_df)
  fit_df <- cbind(data.frame(cell = factor(cell, levels = 1:4)), feature_df)
  fit_p <- nnet::multinom(stats::reformulate(feats, response = "cell"), data = fit_df,
                          decay = decay, maxit = maxit, trace = FALSE, MaxNWts = 10000L)
  fit_r <- lapply(1:4, function(k) {
    idx <- cell == k
    if (sum(idx) <= length(feats) + 1L) stop("Too few observations in (Atilde, Z) cell ", k, ".")
    stats::lm(stats::reformulate(feats, response = "r"),
              data = cbind(data.frame(r = R[idx]), feature_df[idx, , drop = FALSE]))
  })

  predict_cells <- function(new_df) {
    prob <- matrix(stats::predict(fit_p, newdata = new_df, type = "probs"),
                   ncol = length(fit_p$lev))
    p_az <- matrix(0, nrow(prob), 4L)
    p_az[, as.integer(fit_p$lev)] <- prob
    r_az <- vapply(fit_r, function(f) stats::predict(f, newdata = new_df), numeric(nrow(new_df)))
    r_az <- matrix(r_az, ncol = 4L)
    list(p_az = p_az, g_az = r_az * p_az)
  }

  list(predict_cells = predict_cells, converged = isTRUE(fit_p$convergence == 0L))
}

## Averages the named estimates across imputations.
imp_average_estimates <- function(est_list) {
  est_mat <- do.call(rbind, est_list)
  list(mean = colMeans(est_mat, na.rm = TRUE),
       between_sd = apply(est_mat, 2, stats::sd, na.rm = TRUE),
       n_finite = colSums(is.finite(est_mat)))
}

## Diagnostics shared by all settings.
imp_posterior_diagnostics <- function(eta1, A_true = NULL, eig = NULL, beh = NULL) {
  out <- c(
    eta_mean = mean(eta1),
    eta_entropy = mean(-(eta1 * log(pmax(eta1, 1e-12)) +
                           (1 - eta1) * log(pmax(1 - eta1, 1e-12))))
  )
  if (!is.null(A_true)) {
    A_true <- as.vector(A_true)
    out <- c(out,
             map_accuracy = mean(as.integer(eta1 > 0.5) == A_true),
             expected_accuracy = mean(ifelse(A_true == 1, eta1, 1 - eta1)))
  }
  if (!is.null(eig)) {
    out <- c(out,
             mu0_mean = mean(eig$mu0),
             mu1_mean = mean(eig$mu1),
             thetaR0_mean = mean(eig$theta0),
             thetaR1_mean = mean(eig$theta1),
             eig_complex_frac = mean(eig$complex_root),
             eig_degenerate_frac = mean(eig$degenerate),
             eig_out_of_range_frac = mean(eig$out_of_range),
             eig_singular_frac = mean(eig$singular))
  }
  if (!is.null(beh)) {
    out <- c(out, b1_mean = mean(beh$b1), gap_floored_frac = mean(beh$gap_floored))
  }
  out
}
