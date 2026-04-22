################################################################################
## Gym Environments for LURE
##
## Binary-action setup for CartPole-v1 and MountainCar-v0.
## For MountainCar, binary action 0/1 is mapped to Gym actions left/right,
## skipping the neutral action.
################################################################################

gym_expit <- function(x) 1 / (1 + exp(-x))

gym_clip <- function(x, lo = 1e-2, hi = 1 - 1e-2) {
  pmax(pmin(x, hi), lo)
}

gym_safe_ratio <- function(num, denom, lambda = 0.001) {
  num * denom / (denom^2 + lambda)
}

gym_safe_abs_cor <- function(x, y) {
  ok <- is.finite(x) & is.finite(y)
  x <- x[ok]
  y <- y[ok]
  if (length(x) < 2L) {
    return(0)
  }
  sx <- stats::sd(x)
  sy <- stats::sd(y)
  if (!is.finite(sx) || !is.finite(sy) || sx < 1e-8 || sy < 1e-8) {
    return(0)
  }
  abs(stats::cor(x, y))
}

gym_clip_abs_quantile <- function(x, q = 0.98, abs_cap = NULL) {
  cap <- as.numeric(quantile(abs(x), q, na.rm = TRUE, names = FALSE))
  if (!is.null(abs_cap) && is.finite(abs_cap)) {
    cap <- min(cap, abs_cap)
  }
  if (!is.finite(cap) || cap <= 0) {
    return(x)
  }
  pmin(pmax(x, -cap), cap)
}

gym_reward_quantile_bounds <- function(reward_vec, probs = c(0.04, 0.96)) {
  bounds <- as.numeric(stats::quantile(
    reward_vec,
    probs = probs,
    na.rm = TRUE,
    names = FALSE
  ))
  if (length(bounds) != 2L || any(!is.finite(bounds)) || bounds[1] > bounds[2]) {
    bounds <- c(-Inf, Inf)
  }
  stats::setNames(bounds, c("lo", "hi"))
}

gym_state_names <- function(d) {
  paste0("x", seq_len(d))
}

gym_resolve_state_names <- function(state_names, d) {
  if (!is.null(state_names)) {
    state_names <- as.character(unlist(state_names, use.names = FALSE))
    if (length(state_names) == d) {
      return(state_names)
    }
  }
  gym_state_names(d)
}

gym_state_df <- function(state_mat) {
  state_mat <- as.matrix(state_mat)
  out <- as.data.frame(state_mat)
  names(out) <- gym_state_names(ncol(state_mat))
  out
}

gym_state_formula <- function(response, d) {
  reformulate(gym_state_names(d), response = response)
}

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

gym_model_feature_names <- function(k) {
  paste0("f", seq_len(k))
}

gym_model_feature_df <- function(state_mat) {
  feat_mat <- gym_poly_features(state_mat, include_intercept = FALSE)
  out <- as.data.frame(feat_mat)
  names(out) <- gym_model_feature_names(ncol(out))
  out
}

gym_model_formula <- function(response, k) {
  reformulate(gym_model_feature_names(k), response = response)
}

gym_weighted_ridge_fit <- function(state_mat, response_vec,
                                   weights = NULL, ridge = 0.001) {
  x_mat <- gym_poly_features(state_mat)
  if (is.null(weights)) {
    weights <- rep(1, nrow(x_mat))
  }
  weights <- pmax(as.numeric(weights), 1e-8)
  center <- colSums(x_mat * weights) / sum(weights)
  center[1] <- 0
  x_centered <- sweep(x_mat, 2L, center, FUN = "-")
  scale_vec <- sqrt(colSums((x_centered^2) * weights) / sum(weights))
  scale_vec[1] <- 1
  scale_vec[!is.finite(scale_vec) | scale_vec < 1e-6] <- 1
  x_scaled <- sweep(x_centered, 2L, scale_vec, FUN = "/")
  sqrt_w <- sqrt(weights)
  xw <- x_scaled * sqrt_w
  yw <- response_vec * sqrt_w
  penalty <- diag(ncol(x_scaled))
  penalty[1, 1] <- 0
  beta_hat <- solve(crossprod(xw) + ridge * penalty,
                    crossprod(xw, yw))

  predict_fn <- function(state_new) {
    x_new <- gym_poly_features(state_new)
    x_new <- sweep(x_new, 2L, center, FUN = "-")
    x_new <- sweep(x_new, 2L, scale_vec, FUN = "/")
    drop(x_new %*% beta_hat)
  }

  list(beta = beta_hat, center = center, scale = scale_vec, predict = predict_fn)
}

gym_spline_features <- function(state_mat, specs = NULL,
                                df = 6L, degree = 3L) {
  if (!requireNamespace("splines", quietly = TRUE)) {
    stop("Package 'splines' is required for spline-based weighted FQE.")
  }

  state_mat <- as.matrix(state_mat)
  n <- nrow(state_mat)
  d <- ncol(state_mat)

  if (is.null(specs)) {
    specs <- vector("list", d)
  }

  feat_list <- vector("list", d + 1L)
  feat_list[[1L]] <- rep(1, n)

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

    feat_list[[j + 1L]] <- basis_j
  }

  list(x_mat = do.call(cbind, feat_list), specs = specs)
}

gym_weighted_spline_ridge_fit <- function(state_mat, response_vec,
                                          weights = NULL, ridge = 0.001,
                                          spline_df = 6L,
                                          spline_degree = 3L) {
  spline_fit <- gym_spline_features(state_mat, df = spline_df, degree = spline_degree)
  x_mat <- spline_fit$x_mat

  if (is.null(weights)) {
    weights <- rep(1, nrow(x_mat))
  }
  weights <- pmax(as.numeric(weights), 1e-8)
  center <- colSums(x_mat * weights) / sum(weights)
  center[1] <- 0
  x_centered <- sweep(x_mat, 2L, center, FUN = "-")
  scale_vec <- sqrt(colSums((x_centered^2) * weights) / sum(weights))
  scale_vec[1] <- 1
  scale_vec[!is.finite(scale_vec) | scale_vec < 1e-6] <- 1
  x_scaled <- sweep(x_centered, 2L, scale_vec, FUN = "/")
  sqrt_w <- sqrt(weights)
  xw <- x_scaled * sqrt_w
  yw <- response_vec * sqrt_w
  penalty <- diag(ncol(x_scaled))
  penalty[1, 1] <- 0
  beta_hat <- solve(crossprod(xw) + ridge * penalty,
                    crossprod(xw, yw))

  predict_fn <- function(state_new) {
    x_new <- gym_spline_features(state_new, specs = spline_fit$specs)$x_mat
    x_new <- sweep(x_new, 2L, center, FUN = "-")
    x_new <- sweep(x_new, 2L, scale_vec, FUN = "/")
    drop(x_new %*% beta_hat)
  }

  list(beta = beta_hat, center = center, scale = scale_vec,
       specs = spline_fit$specs, predict = predict_fn)
}

gym_flatten_states <- function(state_arr) {
  dims <- dim(state_arr)
  if (length(dims) != 3L) {
    stop("Expected a 3D state array with dimensions N x T x d.")
  }
  matrix(as.numeric(state_arr), nrow = dims[1] * dims[2], ncol = dims[3])
}

gym_draw_initial_states <- function(dgp, n) {
  if (is.null(dgp$init_states) || nrow(dgp$init_states) == 0L) {
    stop("dgp$init_states must be populated before drawing initial states.")
  }
  idx <- sample(seq_len(nrow(dgp$init_states)), n, replace = TRUE)
  dgp$init_states[idx, , drop = FALSE]
}

gym_subset_dat <- function(dat, row_idx) {
  list(
    S = dat$S[row_idx, , , drop = FALSE],
    A = dat$A[row_idx, , drop = FALSE],
    Atilde = dat$Atilde[row_idx, , drop = FALSE],
    R = dat$R[row_idx, , drop = FALSE],
    Sp = dat$Sp[row_idx, , , drop = FALSE],
    init_states = dat$init_states[row_idx, , drop = FALSE],
    discounted_returns = dat$discounted_returns[row_idx]
  )
}

gym_resolve_dir <- function() {
  opt_dir <- getOption("lure.gym.dir")
  if (!is.null(opt_dir)) {
    return(normalizePath(opt_dir, winslash = "/", mustWork = TRUE))
  }
  if (file.exists("gym_data.py") && file.exists("Methods_gym.R")) {
    return(normalizePath(getwd(), winslash = "/", mustWork = TRUE))
  }
  stop("Set options(lure.gym.dir = '<path-to-Gym>') before running Gym scripts.")
}

gym_python_bin <- function() {
  py <- Sys.which("python3")
  if (nzchar(py)) {
    return(py)
  }
  py <- Sys.which("python")
  if (nzchar(py)) {
    return(py)
  }
  stop("Could not find python3 or python on PATH.")
}

gym_parse_json_dataset <- function(payload) {
  state_dim <- length(payload$state_names)
  summary_only <- isTRUE(payload$summary_only)
  init_states <- matrix(as.numeric(unlist(payload$init_states, use.names = FALSE)),
                        nrow = payload$N, ncol = state_dim)
  discounted_returns <- as.numeric(unlist(payload$discounted_returns,
                                          use.names = FALSE))

  if (summary_only) {
    return(list(
      env_name = payload$env_name,
      dataset = payload$dataset,
      summary_only = TRUE,
      N = payload$N,
      T = payload$T,
      state_names = unlist(payload$state_names, use.names = FALSE),
      S = NULL,
      A = NULL,
      Atilde = NULL,
      R = NULL,
      Sp = NULL,
      init_states = init_states,
      discounted_returns = discounted_returns
    ))
  }

  list(
    env_name = payload$env_name,
    dataset = payload$dataset,
    summary_only = FALSE,
    N = payload$N,
    T = payload$T,
    state_names = unlist(payload$state_names, use.names = FALSE),
    S = array(as.numeric(unlist(payload$S, use.names = FALSE)),
              dim = c(payload$N, payload$T, state_dim)),
    A = matrix(as.integer(unlist(payload$A, use.names = FALSE)),
               nrow = payload$N, ncol = payload$T),
    Atilde = matrix(as.integer(unlist(payload$Atilde, use.names = FALSE)),
                    nrow = payload$N, ncol = payload$T),
    R = matrix(as.numeric(unlist(payload$R, use.names = FALSE)),
               nrow = payload$N, ncol = payload$T),
    Sp = array(as.numeric(unlist(payload$Sp, use.names = FALSE)),
               dim = c(payload$N, payload$T, state_dim)),
    init_states = init_states,
    discounted_returns = discounted_returns
  )
}

gym_run_generator <- function(env_name, dataset, N, TT, tau = 0,
                              gamma, seed = 1L,
                              summary_only = FALSE) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("Package 'jsonlite' is required to read Gym datasets.")
  }

  script_path <- file.path(gym_resolve_dir(), "gym_data.py")
  output_path <- tempfile(fileext = ".json")
  on.exit(unlink(output_path), add = TRUE)

  args <- c(
    script_path,
    "--env", env_name,
    "--dataset", dataset,
    "--N", as.character(N),
    "--T", as.character(TT),
    "--tau", as.character(tau),
    "--gamma", as.character(gamma),
    "--seed", as.character(seed),
    "--output", output_path
  )
  if (summary_only) {
    args <- c(args, "--summary-only")
  }

  output <- system2(gym_python_bin(), args = args, stdout = TRUE, stderr = TRUE)
  status <- attr(output, "status")
  if (!is.null(status) && status != 0) {
    stop(paste(output, collapse = "\n"))
  }

  payload <- jsonlite::fromJSON(output_path, simplifyVector = FALSE)
  gym_parse_json_dataset(payload)
}

load_gym_dataset <- function(file_path) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("Package 'jsonlite' is required to read Gym datasets.")
  }
  payload <- jsonlite::fromJSON(txt = paste(readLines(file_path, warn = FALSE),
                                            collapse = "\n"),
                                simplifyVector = FALSE)
  gym_parse_json_dataset(payload)
}

format_gym_tau <- function(tau) {
  sprintf("%.2f", tau)
}

format_gym_gamma <- function(gamma) {
  out <- sprintf("%.6f", gamma)
  out <- sub("0+$", "", out)
  sub("\\.$", "", out)
}

resolve_offline_gym_data_dir <- function(base_dir = gym_resolve_dir()) {
  candidates <- c(
    file.path(base_dir, "data", "offline"),
    base_dir
  )
  for (cand in candidates) {
    if (dir.exists(cand)) {
      return(cand)
    }
  }
  stop("Could not find an offline Gym dataset directory under: ", base_dir)
}

offline_gym_dataset_path <- function(dgp, tau, rep,
                                     data_dir = resolve_offline_gym_data_dir()) {
  base_dir <- gym_resolve_dir()
  candidate_dirs <- unique(c(
    data_dir,
    file.path(base_dir, "data", "offline"),
    base_dir
  ))
  rel_path <- file.path(
    dgp$env_name,
    paste0("tau_", format_gym_tau(tau)),
    sprintf("rep_%03d.json", rep)
  )
  candidate_paths <- file.path(candidate_dirs, rel_path)
  existing_path <- candidate_paths[file.exists(candidate_paths)][1]
  if (!is.na(existing_path)) {
    return(existing_path)
  }
  candidate_paths[1]
}

target_gym_dataset_path <- function(dgp, N, TT, seed, gamma = NULL,
                                    data_dir = gym_resolve_dir()) {
  base_dir <- gym_resolve_dir()
  candidate_dirs <- unique(c(
    data_dir,
    file.path(base_dir, "data", "target"),
    base_dir
  ))
  file_name <- if (is.null(gamma)) {
    sprintf("N_%d_T_%d_seed_%d.json", N, TT, seed)
  } else {
    sprintf("N_%d_T_%d_gamma_%s_seed_%d.json",
            N, TT, format_gym_gamma(gamma), seed)
  }
  rel_path <- file.path(
    dgp$env_name,
    "target_mc",
    file_name
  )
  candidate_paths <- file.path(candidate_dirs, rel_path)
  existing_path <- candidate_paths[file.exists(candidate_paths)][1]
  if (!is.na(existing_path)) {
    return(existing_path)
  }
  candidate_paths[1]
}

mountain_car_target_policy <- function(state_mat) {
  state_mat <- as.matrix(state_mat)
  rep(0.5, nrow(state_mat))
}

cartpole_target_policy <- function(state_mat) {
  state_mat <- as.matrix(state_mat)
  gym_expit(0.10 * state_mat[, 1] + 0.25 * state_mat[, 2] +
              1.20 * state_mat[, 3] + 0.35 * state_mat[, 4])
}

generate_gym_dgp <- function(env_name, bridge_index = NULL,
                             init_states = NULL, pi_func = NULL) {
  state_dim <- switch(
    env_name,
    "MountainCar-v0" = 2L,
    "CartPole-v1" = 4L,
    stop("Unsupported environment: ", env_name)
  )

  if (!is.null(bridge_index)) {
    bridge_index <- as.integer(bridge_index)[1]
    if (!is.finite(bridge_index) || bridge_index < 1L || bridge_index > state_dim) {
      stop("bridge_index must be between 1 and ", state_dim,
           " for ", env_name, ".")
    }
  }

  if (is.null(pi_func)) {
    pi_func <- switch(
      env_name,
      "MountainCar-v0" = mountain_car_target_policy,
      "CartPole-v1" = cartpole_target_policy
    )
  }

  list(
    env_name = env_name,
    state_dim = state_dim,
    bridge_index = bridge_index,
    init_states = init_states,
    pi_func = pi_func
  )
}

generate_offline_data_gym <- function(dgp, N, TT, tau, gamma, seed = 1L) {
  gym_run_generator(
    env_name = dgp$env_name,
    dataset = "offline",
    N = N,
    TT = TT,
    tau = tau,
    gamma = gamma,
    seed = seed
  )
}

generate_target_data_gym <- function(dgp, N, TT, gamma, seed = 1L) {
  gym_run_generator(
    env_name = dgp$env_name,
    dataset = "target",
    N = N,
    TT = TT,
    tau = 0,
    gamma = gamma,
    seed = seed,
    summary_only = TRUE
  )
}

compute_true_value_gym <- function(eval_dat) {
  mean(eval_dat$discounted_returns)
}

estimate_true_value_gym <- function(dgp, N, TT, gamma, seed = 1L,
                                    eval_dat = NULL) {
  if (is.null(eval_dat)) {
    eval_dat <- generate_target_data_gym(dgp, N = N, TT = TT,
                                         gamma = gamma, seed = seed)
  }
  v_value <- compute_true_value_gym(eval_dat)
  mc_se <- stats::sd(eval_dat$discounted_returns) /
    sqrt(length(eval_dat$discounted_returns))
  list(
    V_value = v_value,
    mc_se = mc_se,
    eval_dat = eval_dat
  )
}

select_bridge_index_gym <- function(dat) {
  S_mat <- gym_flatten_states(dat$S)
  At_vec <- as.vector(dat$Atilde)
  Sp_mat <- gym_flatten_states(dat$Sp)
  state_names <- gym_resolve_state_names(dat$state_names, ncol(Sp_mat))

  feature_df <- gym_model_feature_df(S_mat)
  n_feat <- ncol(feature_df)

  fit_at <- glm(
    gym_model_formula("at", n_feat),
    family = quasibinomial(),
    data = cbind(data.frame(at = At_vec), feature_df)
  )
  at_hat <- gym_clip(
    predict(fit_at, newdata = feature_df, type = "response"),
    0.02,
    0.98
  )
  at_resid <- At_vec - at_hat

  bridge_scores <- numeric(ncol(Sp_mat))
  names(bridge_scores) <- state_names

  for (j in seq_len(ncol(Sp_mat))) {
    fit_sp <- lm(
      gym_model_formula("sp", n_feat),
      data = cbind(data.frame(sp = Sp_mat[, j]), feature_df)
    )
    sp_hat <- predict(fit_sp, newdata = feature_df)
    bridge_scores[j] <- gym_safe_abs_cor(at_resid, Sp_mat[, j] - sp_hat)
  }

  list(
    bridge_index = as.integer(which.max(bridge_scores)),
    bridge_scores = bridge_scores
  )
}

gym_poly_action_features <- function(state_mat, action_vec) {
  state_mat <- as.matrix(state_mat)
  action_vec <- as.integer(action_vec)
  pf <- gym_poly_features(state_mat)
  k <- ncol(pf)
  out <- matrix(0, nrow(state_mat), 2L * k)
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
  state_mat <- as.matrix(state_mat)
  pf <- gym_poly_features(state_mat)
  pi_s <- pi_func(state_mat)
  k <- ncol(pf)
  out <- matrix(0, nrow(state_mat), 2L * k)
  out[, seq_len(k)] <- (1 - pi_s) * pf
  out[, k + seq_len(k)] <- pi_s * pf
  out
}

.em_single_run_gym <- function(S_mat, At_vec, R_vec, Sp_mat,
                               init_eta, max_iter = 90, tol = 1e-4) {
  n <- nrow(S_mat)
  d <- ncol(S_mat)
  eta <- init_eta
  reward_clip_bounds <- gym_reward_quantile_bounds(R_vec)

  fit_b <- fit_mu0 <- fit_mu1 <- NULL
  fit_R0 <- fit_R1 <- NULL
  fit_Sp0 <- fit_Sp1 <- NULL
  sigma_R <- c(1, 1)
  sigma_Sp <- matrix(1, nrow = 2L, ncol = d)
  loglik <- -Inf

  df_s <- gym_model_feature_df(S_mat)
  n_feat <- ncol(df_s)
  form_eta <- gym_model_formula("eta1", n_feat)
  form_at <- gym_model_formula("at", n_feat)
  form_r <- gym_model_formula("r", n_feat)
  form_sp <- gym_model_formula("sp", n_feat)

  for (iter in seq_len(max_iter)) {
    eta_old <- eta
    w0 <- pmax(eta[, 1], 1e-8)
    w1 <- pmax(eta[, 2], 1e-8)
    w0_sp <- pmax(eta[, 1], 1e-4)
    w1_sp <- pmax(eta[, 2], 1e-4)

    fit_b <- lm(form_eta,
                data = cbind(data.frame(eta1 = eta[, 2]), df_s))
    b_hat <- gym_clip(fitted(fit_b), 0.04, 0.96)

    fit_mu0 <- glm(form_at, family = quasibinomial(),
                   weights = w0,
                   data = cbind(data.frame(at = At_vec, w0 = w0), df_s))
    fit_mu1 <- glm(form_at, family = quasibinomial(),
                   weights = w1,
                   data = cbind(data.frame(at = At_vec, w1 = w1), df_s))
    mu_hat_0 <- gym_clip(fitted(fit_mu0), 0.04, 0.96)
    mu_hat_1 <- gym_clip(fitted(fit_mu1), 0.04, 0.96)

    fit_R0 <- lm(form_r, weights = w0,
                 data = cbind(data.frame(r = R_vec, w0 = w0), df_s))
    fit_R1 <- lm(form_r, weights = w1,
                 data = cbind(data.frame(r = R_vec, w1 = w1), df_s))
    tR0 <- fitted(fit_R0)
    tR1 <- fitted(fit_R1)
    tR0 <- pmin(pmax(tR0, reward_clip_bounds["lo"]), reward_clip_bounds["hi"])
    tR1 <- pmin(pmax(tR1, reward_clip_bounds["lo"]), reward_clip_bounds["hi"])
    sigma_R[1] <- sqrt(max(sum(eta[, 1] * (R_vec - tR0)^2) / sum(eta[, 1]), 0.01))
    sigma_R[2] <- sqrt(max(sum(eta[, 2] * (R_vec - tR1)^2) / sum(eta[, 2]), 0.01))

    fit_Sp0 <- vector("list", d)
    fit_Sp1 <- vector("list", d)
    tSp0 <- matrix(0, n, d)
    tSp1 <- matrix(0, n, d)
    for (j in seq_len(d)) {
      fit_Sp0[[j]] <- lm(form_sp, weights = w0_sp,
                         data = cbind(data.frame(sp = Sp_mat[, j], w0_sp = w0_sp), df_s))
      fit_Sp1[[j]] <- lm(form_sp, weights = w1_sp,
                         data = cbind(data.frame(sp = Sp_mat[, j], w1_sp = w1_sp), df_s))
      tSp0[, j] <- fitted(fit_Sp0[[j]])
      tSp1[, j] <- fitted(fit_Sp1[[j]])
      sigma_Sp[1, j] <- sqrt(max(sum(eta[, 1] * (Sp_mat[, j] - tSp0[, j])^2) /
                                   sum(eta[, 1]), 0.01))
      sigma_Sp[2, j] <- sqrt(max(sum(eta[, 2] * (Sp_mat[, j] - tSp1[, j])^2) /
                                   sum(eta[, 2]), 0.01))
    }

    log_eta <- matrix(0, n, 2)
    for (a in 0:1) {
      log_b <- ifelse(a == 1, log(b_hat), log(1 - b_hat))
      mu_a <- if (a == 0) mu_hat_0 else mu_hat_1
      log_mu <- dbinom(At_vec, 1, mu_a, log = TRUE)
      tRa <- if (a == 0) tR0 else tR1
      log_h <- dnorm(R_vec, tRa, sigma_R[a + 1], log = TRUE)
      tSpa <- if (a == 0) tSp0 else tSp1
      log_q <- numeric(n)
      for (j in seq_len(d)) {
        log_q <- log_q + dnorm(Sp_mat[, j], tSpa[, j], sigma_Sp[a + 1, j],
                               log = TRUE)
      }
      log_eta[, a + 1] <- log_b + log_mu + log_h + log_q
    }

    mx <- apply(log_eta, 1, max)
    loglik <- sum(log(rowSums(exp(log_eta - mx))) + mx)
    eta <- exp(log_eta - mx)
    eta <- eta / rowSums(eta)
    eta <- pmax(eta, 1e-4)
    eta <- eta / rowSums(eta)

    if (max(abs(eta - eta_old)) < tol) {
      break
    }
  }

  list(
    eta = eta,
    loglik = loglik,
    n_iter = iter,
    sigma_R = sigma_R,
    sigma_Sp = sigma_Sp,
    fit_R0 = fit_R0,
    fit_R1 = fit_R1,
    fit_Sp0 = fit_Sp0,
    fit_Sp1 = fit_Sp1,
    fit_mu0 = fit_mu0,
    fit_mu1 = fit_mu1,
    fit_b = fit_b
  )
}

em_gym <- function(dat, gamma, max_iter = 90, tol = 1e-3,
                   n_restarts = 2) {
  S_mat <- gym_flatten_states(dat$S)
  At_vec <- as.vector(dat$Atilde)
  R_vec <- as.vector(dat$R)
  Sp_mat <- gym_flatten_states(dat$Sp)
  n <- nrow(S_mat)
  reward_clip_bounds <- gym_reward_quantile_bounds(R_vec)

  best <- NULL
  for (rst in seq_len(n_restarts)) {
    eta0 <- matrix(0.5, n, 2)
    if (rst == 1L) {
      eta0[cbind(seq_len(n), At_vec + 1L)] <- 0.7
      eta0[cbind(seq_len(n), 2L - At_vec)] <- 0.3
    } else {
      p <- runif(n, 0.55, 0.85)
      eta0[cbind(seq_len(n), At_vec + 1L)] <- p
      eta0[cbind(seq_len(n), 2L - At_vec)] <- 1 - p
    }

    res <- .em_single_run_gym(S_mat, At_vec, R_vec, Sp_mat, eta0,
                              max_iter = max_iter, tol = tol)
    if (is.null(best) || res$loglik > best$loglik) {
      best <- res
    }
  }

  eta <- best$eta
  fit_R0 <- best$fit_R0
  fit_R1 <- best$fit_R1
  fit_Sp0 <- best$fit_Sp0
  fit_Sp1 <- best$fit_Sp1
  fit_mu0 <- best$fit_mu0
  fit_mu1 <- best$fit_mu1
  fit_b <- best$fit_b
  sigma_R <- best$sigma_R
  sigma_Sp <- best$sigma_Sp

  mean_at <- c(sum(eta[, 1] * At_vec) / sum(eta[, 1]),
               sum(eta[, 2] * At_vec) / sum(eta[, 2]))
  if (mean_at[1] > mean_at[2]) {
    eta <- eta[, 2:1, drop = FALSE]
    tmp <- fit_R0
    fit_R0 <- fit_R1
    fit_R1 <- tmp
    tmp <- fit_Sp0
    fit_Sp0 <- fit_Sp1
    fit_Sp1 <- tmp
    tmp <- fit_mu0
    fit_mu0 <- fit_mu1
    fit_mu1 <- tmp
    sigma_R <- rev(sigma_R)
    sigma_Sp <- sigma_Sp[2:1, , drop = FALSE]
    fit_b <- lm(gym_model_formula("eta1", ncol(gym_model_feature_df(S_mat))),
                data = cbind(data.frame(eta1 = eta[, 2]), gym_model_feature_df(S_mat)))
  }

  predict_theta_R <- function(state_new, a) {
    fit <- if (a == 0) fit_R0 else fit_R1
    reward_hat <- predict(fit, newdata = gym_model_feature_df(state_new))
    pmin(pmax(reward_hat, reward_clip_bounds["lo"]), reward_clip_bounds["hi"])
  }

  predict_theta_Sp <- function(state_new, a) {
    fits <- if (a == 0) fit_Sp0 else fit_Sp1
    state_new <- as.matrix(state_new)
    out <- matrix(0, nrow(state_new), length(fits))
    new_df <- gym_model_feature_df(state_new)
    for (j in seq_along(fits)) {
      out[, j] <- predict(fits[[j]], newdata = new_df)
    }
    out
  }

  predict_mu <- function(state_new, a) {
    fit <- if (a == 0) fit_mu0 else fit_mu1
    gym_clip(predict(fit, newdata = gym_model_feature_df(state_new), type = "response"),
             0.04, 0.96)
  }

  predict_b <- function(state_new) {
    gym_clip(predict(fit_b, newdata = gym_model_feature_df(state_new)), 0.04, 0.96)
  }

  list(
    eta = eta,
    n_iter = best$n_iter,
    reward_clip_bounds = reward_clip_bounds,
    sigma_R = sigma_R,
    sigma_Sp = sigma_Sp,
    predict_theta_R = predict_theta_R,
    predict_theta_Sp = predict_theta_Sp,
    predict_mu = predict_mu,
    predict_b = predict_b,
    fit_R0 = fit_R0,
    fit_R1 = fit_R1,
    fit_Sp0 = fit_Sp0,
    fit_Sp1 = fit_Sp1,
    fit_mu0 = fit_mu0,
    fit_mu1 = fit_mu1,
    fit_b = fit_b
  )
}

solve_omega_gym <- function(em_out, dat, dgp, gamma, ridge = 0.001) {
  S_mat <- gym_flatten_states(dat$S)
  Sp_mat <- gym_flatten_states(dat$Sp)
  eta <- em_out$eta
  n <- nrow(S_mat)

  phi0 <- gym_poly_action_features(S_mat, rep(0L, n))
  phi1 <- gym_poly_action_features(S_mat, rep(1L, n))
  phi_pi_sp <- gym_poly_policy_features(Sp_mat, dgp$pi_func)

  d_feat <- ncol(phi0)
  a_mat <- matrix(0, d_feat, d_feat)
  for (a in 0:1) {
    phi_a <- if (a == 0) phi0 else phi1
    eta_a <- eta[, a + 1]
    diff_a <- phi_a - gamma * phi_pi_sp
    a_mat <- a_mat + crossprod(diff_a, phi_a * eta_a) / n
  }

  init_states <- gym_draw_initial_states(dgp, 5000)
  b_vec <- (1 - gamma) * colMeans(gym_poly_policy_features(init_states, dgp$pi_func))

  beta_hat <- solve(a_mat + ridge * diag(d_feat), b_vec)

  om0 <- drop(phi0 %*% beta_hat)
  om1 <- drop(phi1 %*% beta_hat)
  norm_c <- mean(eta[, 1] * om0 + eta[, 2] * om1)
  if (abs(norm_c) > 1e-6) {
    beta_hat <- beta_hat / norm_c
  }

  predict_omega <- function(state_new, action_vec) {
    drop(gym_poly_action_features(state_new, action_vec) %*% beta_hat)
  }

  list(
    beta = beta_hat,
    predict_omega = predict_omega,
    omega_all = cbind(
      predict_omega(S_mat, rep(0L, n)),
      predict_omega(S_mat, rep(1L, n))
    )
  )
}

weighted_fqe_gym <- function(em_out, dat, dgp, gamma, n_iter = 40, ridge = .0005,
                             spline_df = 6L, spline_degree = 3L) {
  S_mat <- gym_flatten_states(dat$S)
  Sp_mat <- gym_flatten_states(dat$Sp)
  R_vec <- as.vector(dat$R)
  eta <- em_out$eta
  value_clip_bounds <- c(lo = -Inf, hi = Inf)
  value_clip_mode <- "none"

  if (is.finite(gamma) && gamma < 1) {
    reward_bounds <- gym_reward_quantile_bounds(R_vec)
    value_clip_bounds <- reward_bounds / (1 - gamma)
    value_clip_mode <- "quantile"
    if (any(!is.finite(value_clip_bounds)) || value_clip_bounds["lo"] > value_clip_bounds["hi"]) {
      value_clip_bounds[] <- c(-Inf, Inf)
      value_clip_mode <- "none"
    } else if (abs(value_clip_bounds["hi"] - value_clip_bounds["lo"]) < 1e-8) {
      value_abs_cap <- max(abs(R_vec), na.rm = TRUE) / (1 - gamma)
      if (is.finite(value_abs_cap) && value_abs_cap > 0) {
        value_clip_bounds[] <- c(-value_abs_cap, value_abs_cap)
        value_clip_mode <- "abs_fallback"
      } else {
        value_clip_bounds[] <- c(-Inf, Inf)
        value_clip_mode <- "none"
      }
    }
  }

  clip_value <- function(x) {
    pmin(pmax(x, value_clip_bounds["lo"]), value_clip_bounds["hi"])
  }

  pi_sp <- dgp$pi_func(Sp_mat)
  Q0_sp <- rep(0, nrow(S_mat))
  Q1_sp <- rep(0, nrow(S_mat))
  fit_Q0 <- fit_Q1 <- NULL

  for (iter in seq_len(n_iter)) {
    V_sp <- (1 - pi_sp) * Q0_sp + pi_sp * Q1_sp
    Y_target <- R_vec + gamma * V_sp
    w0 <- pmax(eta[, 1], 1e-3)
    w1 <- pmax(eta[, 2], 1e-3)

    fit_Q0 <- gym_weighted_spline_ridge_fit(
      S_mat, Y_target,
      weights = w0,
      ridge = ridge,
      spline_df = spline_df,
      spline_degree = spline_degree
    )
    fit_Q1 <- gym_weighted_spline_ridge_fit(
      S_mat, Y_target,
      weights = w1,
      ridge = ridge,
      spline_df = spline_df,
      spline_degree = spline_degree
    )

    Q0_sp <- clip_value(fit_Q0$predict(Sp_mat))
    Q1_sp <- clip_value(fit_Q1$predict(Sp_mat))
  }

  predict_Q <- function(state_new, a) {
    fit <- if (a == 0) fit_Q0 else fit_Q1
    clip_value(fit$predict(state_new))
  }

  predict_V <- function(state_new) {
    pi_s <- dgp$pi_func(state_new)
    v_hat <- (1 - pi_s) * predict_Q(state_new, 0) + pi_s * predict_Q(state_new, 1)
    clip_value(v_hat)
  }

  list(
    predict_Q = predict_Q,
    predict_V = predict_V,
    fit_Q0 = fit_Q0,
    fit_Q1 = fit_Q1,
    value_clip_bounds = value_clip_bounds,
    value_clip_mode = value_clip_mode
  )
}

compute_eta_outfold_gym <- function(em, S_te, At_te, R_te, Sp_te) {
  n <- nrow(S_te)
  d <- ncol(S_te)
  b_hat <- gym_clip(em$predict_b(S_te), 0.04, 0.96)
  mu_hat_0 <- em$predict_mu(S_te, 0)
  mu_hat_1 <- em$predict_mu(S_te, 1)
  tR0 <- em$predict_theta_R(S_te, 0)
  tR1 <- em$predict_theta_R(S_te, 1)
  tSp0 <- em$predict_theta_Sp(S_te, 0)
  tSp1 <- em$predict_theta_Sp(S_te, 1)

  log_eta <- matrix(0, n, 2)
  for (a in 0:1) {
    log_b <- ifelse(a == 1, log(b_hat), log(1 - b_hat))
    mu_a <- if (a == 0) mu_hat_0 else mu_hat_1
    log_mu <- dbinom(At_te, 1, mu_a, log = TRUE)
    tRa <- if (a == 0) tR0 else tR1
    log_h <- dnorm(R_te, tRa, em$sigma_R[a + 1], log = TRUE)
    tSpa <- if (a == 0) tSp0 else tSp1
    log_q <- numeric(n)
    for (j in seq_len(d)) {
      log_q <- log_q + dnorm(Sp_te[, j], tSpa[, j], em$sigma_Sp[a + 1, j],
                             log = TRUE)
    }
    log_eta[, a + 1] <- log_b + log_mu + log_h + log_q
  }

  mx <- apply(log_eta, 1, max)
  eta <- exp(log_eta - mx)
  eta <- eta / rowSums(eta)
  eta <- pmax(eta, 1e-5)
  eta / rowSums(eta)
}

mr_components_gym <- function(dat, dgp, gamma) {
  n_traj <- dim(dat$S)[1]
  TT <- dim(dat$S)[2]
  state_dim <- dim(dat$S)[3]
  bridge_clip_q <- 0.95
  bridge_abs_cap <- 5
  omega_clip_q <- 0.95
  omega_abs_cap <- 6
  bridge_out <- select_bridge_index_gym(dat)
  bridge_scores <- bridge_out$bridge_scores
  bridge_index <- bridge_out$bridge_index

  if (!is.null(dgp$bridge_index)) {
    bridge_index <- as.integer(dgp$bridge_index)[1]
  }
  if (!is.finite(bridge_index) || bridge_index < 1L || bridge_index > state_dim) {
    stop("bridge_index must be between 1 and ", state_dim, ".")
  }

  fold_ids <- sample(rep(1:2, length.out = n_traj))

  phi_list <- vector("list", 2)
  direct_list <- numeric(2)

  for (k in 1:2) {
    train_idx <- which(fold_ids != k)
    test_idx <- which(fold_ids == k)

    dat_train <- gym_subset_dat(dat, train_idx)
    S_te <- gym_flatten_states(dat$S[test_idx, , , drop = FALSE])
    At_te <- as.vector(dat$Atilde[test_idx, , drop = FALSE])
    R_te <- as.vector(dat$R[test_idx, , drop = FALSE])
    Sp_te <- gym_flatten_states(dat$Sp[test_idx, , , drop = FALSE])
    n_test <- nrow(S_te)

    em <- em_gym(dat_train, gamma)
    omega_out <- solve_omega_gym(em, dat_train, dgp, gamma)
    fqe <- weighted_fqe_gym(em, dat_train, dgp, gamma)

    eta_te <- compute_eta_outfold_gym(em, S_te, At_te, R_te, Sp_te)

    tR0 <- em$predict_theta_R(S_te, 0)
    tR1 <- em$predict_theta_R(S_te, 1)
    tAt0 <- em$predict_mu(S_te, 0)
    tAt1 <- em$predict_mu(S_te, 1)
    tSp0 <- em$predict_theta_Sp(S_te, 0)[, bridge_index]
    tSp1 <- em$predict_theta_Sp(S_te, 1)[, bridge_index]

    om0 <- omega_out$predict_omega(S_te, rep(0L, n_test))
    om1 <- omega_out$predict_omega(S_te, rep(1L, n_test))
    omega_clip <- as.numeric(quantile(c(abs(om0), abs(om1)), omega_clip_q,
                                      na.rm = TRUE, names = FALSE))
    if (is.finite(omega_abs_cap)) {
      omega_clip <- min(omega_clip, omega_abs_cap)
    }
    om0_clip <- pmin(pmax(om0, -omega_clip), omega_clip)
    om1_clip <- pmin(pmax(om1, -omega_clip), omega_clip)

    V_sp <- fqe$predict_V(Sp_te)
    Q0 <- fqe$predict_Q(S_te, 0)
    Q1 <- fqe$predict_Q(S_te, 1)
    MV0 <- (Q0 - tR0) / gamma
    MV1 <- (Q1 - tR1) / gamma

    direct_states <- gym_draw_initial_states(dgp, 5000)
    direct_list[k] <- mean(fqe$predict_V(direct_states))

    Sp_bridge <- Sp_te[, bridge_index]
    d_At_0 <- tAt0 - tAt1
    d_At_1 <- tAt1 - tAt0
    d_R_0 <- tR0 - tR1
    d_R_1 <- tR1 - tR0
    d_Sp_0 <- tSp0 - tSp1
    d_Sp_1 <- tSp1 - tSp0

    br_At_0 <- gym_clip_abs_quantile(gym_safe_ratio(At_te - tAt1, d_At_0),
                     bridge_clip_q, bridge_abs_cap)
    br_At_1 <- gym_clip_abs_quantile(gym_safe_ratio(At_te - tAt0, d_At_1),
                     bridge_clip_q, bridge_abs_cap)
    br_R_0 <- gym_clip_abs_quantile(gym_safe_ratio(R_te - tR1, d_R_0),
                    bridge_clip_q, bridge_abs_cap)
    br_R_1 <- gym_clip_abs_quantile(gym_safe_ratio(R_te - tR0, d_R_1),
                    bridge_clip_q, bridge_abs_cap)
    br_Sp_0 <- gym_clip_abs_quantile(gym_safe_ratio(Sp_bridge - tSp1, d_Sp_0),
                     bridge_clip_q, bridge_abs_cap)
    br_Sp_1 <- gym_clip_abs_quantile(gym_safe_ratio(Sp_bridge - tSp0, d_Sp_1),
                     bridge_clip_q, bridge_abs_cap)

    g0 <- gym_clip_abs_quantile(br_At_0 * br_R_0, bridge_clip_q, bridge_abs_cap)
    g1 <- gym_clip_abs_quantile(br_At_1 * br_R_1, bridge_clip_q, bridge_abs_cap)
    gp0 <- gym_clip_abs_quantile(br_At_0 * br_Sp_0, bridge_clip_q, bridge_abs_cap)
    gp1 <- gym_clip_abs_quantile(br_At_1 * br_Sp_1, bridge_clip_q, bridge_abs_cap)

    T1 <- gp0 * om0_clip * (R_te - tR0) + gp1 * om1_clip * (R_te - tR1)
    T2 <- g0 * om0_clip * (V_sp - MV0) + g1 * om1_clip * (V_sp - MV1)

    phi_list[[k]] <- T1 / (1 - gamma) + T2 * gamma / (1 - gamma)
  }

  list(
    n_traj = n_traj,
    TT = TT,
    direct_list = direct_list,
    phi_list = phi_list,
    bridge_index = bridge_index,
    bridge_scores = bridge_scores
  )
}

summarize_mr_components_gym <- function(comps) {
  direct <- mean(comps$direct_list)
  phi_all <- unlist(comps$phi_list)
  V_hat <- direct + mean(phi_all)
  se <- sqrt(stats::var(phi_all) / (comps$n_traj * comps$TT))

  list(
    direct = direct,
    V_hat = V_hat,
    se = se,
    ci_lo = V_hat - 1.96 * se,
    ci_hi = V_hat + 1.96 * se,
    bridge_index = comps$bridge_index,
    bridge_scores = comps$bridge_scores
  )
}

direct_only_estimator_gym <- function(dat, dgp, gamma) {
  comps <- mr_components_gym(dat, dgp, gamma)
  summary <- summarize_mr_components_gym(comps)
  list(
    V_hat = summary$direct,
    bridge_index = summary$bridge_index,
    bridge_scores = summary$bridge_scores
  )
}

mr_estimator_gym <- function(dat, dgp, gamma) {
  comps <- mr_components_gym(dat, dgp, gamma)
  summarize_mr_components_gym(comps)
}

evaluate_gym_estimators <- function(dat, dgp, gamma, seed = NULL) {
  if (!is.null(seed)) {
    set.seed(seed)
  }

  dgp_rep <- dgp
  dgp_rep$init_states <- dat$init_states
  state_names <- gym_resolve_state_names(dat$state_names, dim(dat$S)[3])
  bridge_score_names <- paste0("bridge_score_", make.names(state_names))

  mr_out <- tryCatch({
    comps <- mr_components_gym(dat, dgp_rep, gamma)
    summarize_mr_components_gym(comps)
  }, error = function(e) list(
    direct = NA_real_,
    V_hat = NA_real_,
    ci_lo = NA_real_,
    ci_hi = NA_real_,
    bridge_index = NA_real_,
    bridge_scores = stats::setNames(rep(NA_real_, length(state_names)), state_names)
  ))
  V_direct <- mr_out$direct
  bridge_scores <- mr_out$bridge_scores
  if (is.null(bridge_scores) || length(bridge_scores) != length(state_names)) {
    bridge_scores <- stats::setNames(rep(NA_real_, length(state_names)), state_names)
  }
  bridge_score_vec <- stats::setNames(as.numeric(bridge_scores), bridge_score_names)

  V_fqe <- tryCatch(
    naive_fqe_gym(dat, dgp_rep, gamma)$V_hat,
    error = function(e) NA_real_
  )
  V_sis <- tryCatch(
    naive_sis_gym(dat, dgp_rep, gamma)$V_hat,
    error = function(e) NA_real_
  )
  V_mis <- tryCatch(
    naive_mis_gym(dat, dgp_rep, gamma)$V_hat,
    error = function(e) NA_real_
  )
  V_drl <- tryCatch(
    naive_drl_gym(dat, dgp_rep, gamma)$V_hat,
    error = function(e) NA_real_
  )
  V_lstd <- tryCatch(
    naive_lstd_gym(dat, dgp_rep, gamma)$V_hat,
    error = function(e) NA_real_
  )

  c(
    DIRECT = V_direct,
    FQE = V_fqe,
    SIS = V_sis,
    MIS = V_mis,
    DRL = V_drl,
    LSTD = V_lstd,
    MR = mr_out$V_hat,
    MR_ci_lo = mr_out$ci_lo,
    MR_ci_hi = mr_out$ci_hi,
    bridge_index = unname(mr_out$bridge_index),
    bridge_score_vec
  )
}

one_rep_gym <- function(dgp, N, TT, tau, gamma, seed = 1L) {
  dat <- generate_offline_data_gym(dgp, N = N, TT = TT, tau = tau,
                                   gamma = gamma, seed = seed)
  evaluate_gym_estimators(dat, dgp, gamma, seed = seed)
}