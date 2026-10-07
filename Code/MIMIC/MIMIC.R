# Definitions only. This file has no top-level side effects: source() it to obtain
# the MIMIC pipeline and estimators, then run the analysis with run_mimic.R.

suppressPackageStartupMessages({
  library(dplyr)
})

# readr is used only for read_csv(). Fall back to base R when it is unavailable so
# that this file can be sourced in a bare environment.
mimic_read_csv <- function(file_path) {
  if (requireNamespace("readr", quietly = TRUE)) {
    return(as.data.frame(
      readr::read_csv(file_path, show_col_types = FALSE),
      stringsAsFactors = FALSE
    ))
  }
  utils::read.csv(file_path, check.names = FALSE, stringsAsFactors = FALSE)
}

get_mimic_script_dir <- function() {
  frame_files <- vapply(sys.frames(), function(x) {
    if (!is.null(x$ofile)) x$ofile else ""
  }, character(1))
  frame_files <- frame_files[nzchar(frame_files)]
  if (length(frame_files) > 0L) {
    return(dirname(normalizePath(frame_files[length(frame_files)])))
  }
  normalizePath(getwd())
}

mimic_dir <- get_mimic_script_dir()
project_dir <- normalizePath(file.path(mimic_dir, ".."))
options(lure.mimic.dir = mimic_dir)

source(file.path(project_dir, "Gym", "Methods_gym.R"))
source(file.path(project_dir, "Gym", "baseline_gym.R"))

mimic_linear_feature_fit <- function(state_mat,
                                     include_intercept = TRUE,
                                     specs = NULL,
                                     return_specs = FALSE) {
  state_mat <- as.matrix(state_mat)
  storage.mode(state_mat) <- "double"

  x_mat <- if (include_intercept) {
    cbind(`(Intercept)` = 1, state_mat)
  } else {
    state_mat
  }

  if (is.null(specs)) {
    specs <- list(
      include_intercept = include_intercept,
      state_dim = ncol(state_mat)
    )
  }

  if (return_specs) {
    return(list(x_mat = x_mat, specs = specs))
  }

  x_mat
}

mimic_model_feature_names <- function(k) {
  paste0("f", seq_len(k))
}

gym_poly_features <- function(state_mat, include_intercept = TRUE,
                              specs = NULL,
                              spline_df = NULL,
                              spline_degree = NULL,
                              return_specs = FALSE) {
  mimic_linear_feature_fit(
    state_mat,
    include_intercept = include_intercept,
    specs = specs,
    return_specs = return_specs
  )
}

gym_model_feature_df <- function(state_mat, specs = NULL,
                                 spline_df = NULL,
                                 spline_degree = NULL,
                                 return_specs = FALSE) {
  feat_fit <- gym_poly_features(
    state_mat,
    include_intercept = FALSE,
    specs = specs,
    spline_df = spline_df,
    spline_degree = spline_degree,
    return_specs = TRUE
  )
  out <- as.data.frame(feat_fit$x_mat)
  names(out) <- mimic_model_feature_names(ncol(out))
  if (return_specs) {
    return(list(feature_df = out, specs = feat_fit$specs))
  }
  out
}

gym_weighted_spline_ridge_fit <- function(state_mat, response_vec,
                                          weights = NULL, ridge = 0.001,
                                          spline_df = NULL,
                                          spline_degree = NULL) {
  weighted_fit <- get("gym_weighted_ridge_fit", mode = "function")
  weighted_fit(
    state_mat = state_mat,
    response_vec = response_vec,
    weights = weights,
    ridge = ridge,
    spline_df = spline_df,
    spline_degree = spline_degree
  )
}

mimic_clip_prob <- function(x, lo = 0.02, hi = 0.98) {
  pmax(pmin(x, hi), lo)
}

mimic_named_state_mat <- function(state_mat, state_names) {
  state_mat <- as.matrix(state_mat)
  if (ncol(state_mat) != length(state_names)) {
    stop(
      "State matrix has ", ncol(state_mat),
      " columns, but ", length(state_names),
      " state names were provided."
    )
  }
  colnames(state_mat) <- state_names
  state_mat
}

mimic_normalize_policy_type <- function(policy_type) {
  alias_map <- c(
    high_dose = "high_dose",
    low_dose = "low_dose",
    sofa_11 = "sofa_11",
    always_treat = "high_dose",
    never_treat = "low_dose"
  )

  policy_type <- as.character(policy_type)[1]
  if (!nzchar(policy_type) || is.na(policy_type) || !(policy_type %in% names(alias_map))) {
    stop(
      "Unsupported policy_type: ", policy_type,
      ". Use one of: high_dose, low_dose, sofa_11."
    )
  }
  unname(alias_map[[policy_type]])
}

mimic_state_df <- function(state_mat, state_names) {
  state_mat <- mimic_named_state_mat(state_mat, state_names)
  out <- as.data.frame(state_mat)
  names(out) <- state_names
  out
}

MIMIC_CURATED_STATE_COLS <- c(
  "Albumin",
  "Arterial_BE",
  "Arterial_lactate",
  "Arterial_pH",
  "BUN",
  "CO2_mEqL",
  "Calcium",
  "Chloride",
  "Creatinine",
  "DiaBP",
  "FiO2_1",
  "GCS",
  "Glucose",
  "HCO3",
  "HR",
  "Hb",
  "INR",
  "Ionised_Ca",
  "Magnesium",
  "MeanBP",
  "PT",
  "PTT",
  "PaO2_FiO2",
  "Platelets_count",
  "Potassium",
  "RR",
  "SGOT",
  "SGPT",
  "SIRS",
  "SOFA",
  "Shock_Index",
  "Sodium",
  "SpO2",
  "SysBP",
  "Temp_C",
  "Total_bili",
  "WBC_count",
  "Weight_kg",
  "age",
  "elixhauser",
  "gender",
  "mechvent",
  "output_total",
  "paCO2",
  "paO2",
  "re_admission"
)

mimic_default_state_cols <- function(ds,
                                     id_col = "icustayid",
                                     time_col = "bloc",
                                     action_col = "iv_input",
                                     reward_col = "SOFA") {
  excluded <- c(
    id_col,
    time_col,
    "charttime",
    "presumed_onset",
    action_col,
    "died_in_hosp",
    "died_within_48h_of_out_time",
    "mortality_90d",
    "delay_end_of_record_and_discharge_or_death"
  )

  state_cols <- setdiff(MIMIC_CURATED_STATE_COLS, excluded)
  if (reward_col %in% MIMIC_CURATED_STATE_COLS) {
    state_cols <- unique(c(reward_col, state_cols))
  }

  missing_cols <- setdiff(state_cols, names(ds))
  if (length(missing_cols) > 0L) {
    stop(
      "Requested default state columns are missing from the dataset: ",
      paste(missing_cols, collapse = ", ")
    )
  }

  state_cols
}

# Candidate covariate sets (README section 8; evidence in study_covariates.R).
# "reduced" drops near-duplicate, derived, composite, acid-base-redundant and rarely
# re-measured variables, and swaps cumulative urine output (output_total) for the
# per-bloc output_4hourly. "compact" keeps SOFA, frequently measured vitals, lactate,
# urine output and baseline covariates. Pass the result as state_cols to
# run_mimic_lure() or mimic_build_gym_dat(); output_4hourly is a column of the CSV.
MIMIC_REDUCED_DROP <- c(
  "PT", "CO2_mEqL", "SGOT", "paO2", "DiaBP", "SysBP", "SIRS",
  "SGPT", "Albumin", "Magnesium", "Calcium", "Ionised_Ca", "PTT", "Chloride",
  "Arterial_BE", "paCO2", "FiO2_1", "output_total", "re_admission"
)
MIMIC_COMPACT_COLS <- c(
  "SOFA", "MeanBP", "HR", "Shock_Index", "RR", "SpO2", "Temp_C", "PaO2_FiO2",
  "mechvent", "Arterial_lactate", "output_4hourly", "age", "gender", "elixhauser"
)

mimic_covariate_set <- function(name = c("full", "reduced", "compact")) {
  name <- match.arg(name)
  switch(name,
    full    = unique(c("SOFA", setdiff(MIMIC_CURATED_STATE_COLS, "re_admission"))),
    reduced = c(unique(c("SOFA", setdiff(MIMIC_CURATED_STATE_COLS, MIMIC_REDUCED_DROP))),
                "output_4hourly"),
    compact = MIMIC_COMPACT_COLS
  )
}

mimic_load_panel <- function(file_path = file.path(mimic_dir, "sepsis_processed_state_action.csv"),
                             horizon = 20L,
                             id_col = "icustayid",
                             time_col = "bloc",
                             action_col = "iv_input",
                             action_threshold = 1) {
  file_path <- normalizePath(file_path, mustWork = TRUE)
  ds <- mimic_read_csv(file_path)

  required_cols <- c(id_col, time_col, action_col)
  missing_cols <- setdiff(required_cols, names(ds))
  if (length(missing_cols) > 0L) {
    stop("Missing required columns: ", paste(missing_cols, collapse = ", "))
  }

  stay_counts <- table(ds[[id_col]])
  valid_ids <- names(stay_counts[stay_counts == horizon])
  ds <- ds[ds[[id_col]] %in% valid_ids, , drop = FALSE]
  ds <- ds[order(ds[[id_col]], ds[[time_col]]), , drop = FALSE]
  ds[[action_col]] <- as.integer(ds[[action_col]] >= action_threshold)
  ds
}

mimic_build_gym_dat <- function(ds,
                                state_cols = NULL,
                                horizon = 20L,
                                id_col = "icustayid",
                                time_col = "bloc",
                                action_col = "iv_input",
                                reward_col = "SOFA",
                                reward_transform = identity,
                                reward_timing = c("next", "current"),
                                drop_constant_states = TRUE) {
  reward_timing <- match.arg(reward_timing)
  if (is.null(state_cols)) {
    state_cols <- mimic_default_state_cols(
      ds,
      id_col = id_col,
      time_col = time_col,
      action_col = action_col,
      reward_col = reward_col
    )
  }

  required_cols <- unique(c(id_col, time_col, state_cols, action_col, reward_col))
  missing_cols <- setdiff(required_cols, names(ds))
  if (length(missing_cols) > 0L) {
    stop("Missing required columns: ", paste(missing_cols, collapse = ", "))
  }

  ds <- ds[, required_cols, drop = FALSE]
  ds <- ds[stats::complete.cases(ds), , drop = FALSE]
  ds <- ds[order(ds[[id_col]], ds[[time_col]]), , drop = FALSE]

  stay_counts <- table(ds[[id_col]])
  valid_ids <- names(stay_counts[stay_counts == horizon])
  ds <- ds[ds[[id_col]] %in% valid_ids, , drop = FALSE]
  ds <- ds[order(ds[[id_col]], ds[[time_col]]), , drop = FALSE]

  # Drop state variables with no variation in the analysis cohort. Such columns are
  # collinear with the intercept, which makes the linear design exactly rank
  # deficient and leaves the ridge-regularised omega/MIS solves ill-posed.
  dropped_states <- character(0)
  if (isTRUE(drop_constant_states) && nrow(ds) > 1L) {
    state_sd <- vapply(
      state_cols,
      function(nm) stats::sd(as.numeric(ds[[nm]])),
      numeric(1)
    )
    constant_cols <- state_cols[!is.finite(state_sd) | state_sd < 1e-12]
    keep_cols <- setdiff(constant_cols, reward_col)
    if (length(keep_cols) > 0L) {
      dropped_states <- keep_cols
      state_cols <- setdiff(state_cols, keep_cols)
      message(
        "Dropping ", length(keep_cols),
        " constant state variable(s): ", paste(keep_cols, collapse = ", ")
      )
    }
    if (length(constant_cols) > length(keep_cols)) {
      warning(
        "Reward column '", reward_col,
        "' is constant in this cohort; it was retained as a state variable."
      )
    }
  }
  if (length(state_cols) == 0L) {
    stop("No state variables remain after dropping constant columns.")
  }

  stays <- split(ds, ds[[id_col]])
  n_traj <- length(stays)
  if (n_traj == 0L) {
    stop("No complete ICU stays remain after filtering.")
  }

  TT <- horizon - 1L
  if (TT < 1L) {
    stop("horizon must be at least 2.")
  }

  state_dim <- length(state_cols)
  S <- array(NA_real_, dim = c(n_traj, TT, state_dim),
             dimnames = list(NULL, NULL, state_cols))
  Sp <- array(NA_real_, dim = c(n_traj, TT, state_dim),
              dimnames = list(NULL, NULL, state_cols))
  Atilde <- matrix(NA_integer_, nrow = n_traj, ncol = TT)
  R <- matrix(NA_real_, nrow = n_traj, ncol = TT)
  init_states <- matrix(NA_real_, nrow = n_traj, ncol = state_dim,
                        dimnames = list(NULL, state_cols))

  for (i in seq_along(stays)) {
    stay <- stays[[i]]
    if (nrow(stay) != horizon) {
      stop("Stay ", names(stays)[i], " does not have exactly ", horizon, " rows.")
    }

    state_mat <- as.matrix(stay[, state_cols, drop = FALSE])
    storage.mode(state_mat) <- "double"

    # reward_timing = "next"    : R_t = f(reward_col at t+1), the outcome the action at
    #                             time t can actually influence.
    # reward_timing = "current" : R_t = f(reward_col at t). Reproduces the published
    #                             Table 1, but makes R_t an exact copy of a coordinate
    #                             of S_t whenever reward_col is in state_cols, which
    #                             collapses theta_R(s, 0) and theta_R(s, 1) and hence
    #                             nulls the LURE correction term.
    reward_idx <- if (reward_timing == "next") seq_len(TT) + 1L else seq_len(TT)
    reward_vec <- as.numeric(reward_transform(as.numeric(stay[[reward_col]][reward_idx])))
    if (length(reward_vec) != TT) {
      stop("reward_transform must return a vector of length ", TT, ".")
    }

    S[i, , ] <- state_mat[seq_len(TT), , drop = FALSE]
    Sp[i, , ] <- state_mat[seq_len(TT) + 1L, , drop = FALSE]
    Atilde[i, ] <- as.integer(stay[[action_col]][seq_len(TT)])
    R[i, ] <- reward_vec
    init_states[i, ] <- state_mat[1L, ]
  }

  list(
    S = S,
    Sp = Sp,
    Atilde = Atilde,
    R = R,
    init_states = init_states,
    state_names = state_cols,
    stay_ids = names(stays),
    original_horizon = horizon,
    transition_horizon = TT,
    reward_timing = reward_timing,
    reward_col = reward_col,
    dropped_states = dropped_states
  )
}

mimic_fit_target_policy <- function(dat,
                                    policy_type = c("high_dose", "low_dose", "sofa_11"),
                                    sofa_cutoff = 3) {
  policy_type <- mimic_normalize_policy_type(policy_type)
  state_names <- dat$state_names

  if (policy_type == "high_dose") {
    return(list(
      policy_name = policy_type,
      fit = NULL,
      treat_rate = 1,
      pi_func = function(state_mat) rep(1, nrow(as.matrix(state_mat)))
    ))
  }

  if (policy_type == "low_dose") {
    return(list(
      policy_name = policy_type,
      fit = NULL,
      treat_rate = 0,
      pi_func = function(state_mat) rep(0, nrow(as.matrix(state_mat)))
    ))
  }

  if (policy_type == "sofa_11") {
    if (!("SOFA" %in% state_names)) {
      stop("SOFA must be included in state_cols when policy_type = 'sofa_11'.")
    }
    return(list(
      policy_name = policy_type,
      fit = NULL,
      treat_rate = NA_real_,
      pi_func = function(state_mat) {
        state_mat <- mimic_named_state_mat(state_mat, state_names)
        as.numeric(state_mat[, "SOFA"] >= sofa_cutoff)
      }
    ))
  }

  stop("Unhandled policy_type: ", policy_type)
}

generate_mimic_dgp <- function(dat, pi_func, bridge_index = NULL) {
  list(
    env_name = "MIMIC",
    state_dim = length(dat$state_names),
    bridge_index = bridge_index,
    init_states = dat$init_states,
    pi_func = pi_func
  )
}

mimic_empirical_policy_features <- function(state_mat, pi_func, specs) {
  policy_features_fn <- get("gym_poly_policy_features", mode = "function")
  policy_features_fn(state_mat, pi_func, specs = specs)
}

mimic_solve_omega <- function(em_out, dat, dgp, gamma, ridge = 0.001) {
  S_mat <- gym_flatten_states(dat$S)
  Sp_mat <- gym_flatten_states(dat$Sp)
  eta <- em_out$eta
  n <- nrow(S_mat)

  feat_fit <- gym_poly_features(S_mat, return_specs = TRUE)
  basis_specs <- feat_fit$specs
  phi0 <- gym_action_feature_matrix(feat_fit$x_mat, rep(0L, n))
  phi1 <- gym_action_feature_matrix(feat_fit$x_mat, rep(1L, n))
  phi_pi_sp <- mimic_empirical_policy_features(Sp_mat, dgp$pi_func, specs = basis_specs)

  d_feat <- ncol(phi0)
  a_mat <- matrix(0, d_feat, d_feat)
  for (a in 0:1) {
    phi_a <- if (a == 0) phi0 else phi1
    eta_a <- eta[, a + 1]
    diff_a <- phi_a - gamma * phi_pi_sp
    a_mat <- a_mat + crossprod(diff_a, phi_a * eta_a) / n
  }

  b_vec <- (1 - gamma) * colMeans(
    mimic_empirical_policy_features(dgp$init_states, dgp$pi_func, specs = basis_specs)
  )

  beta_hat <- solve(a_mat + ridge * diag(d_feat), b_vec)

  om0 <- drop(phi0 %*% beta_hat)
  om1 <- drop(phi1 %*% beta_hat)
  norm_c <- mean(eta[, 1] * om0 + eta[, 2] * om1)
  if (abs(norm_c) > 1e-6) {
    beta_hat <- beta_hat / norm_c
  }

  predict_omega <- function(state_new, action_vec) {
    basis_new <- gym_poly_features(state_new, specs = basis_specs)
    phi_new <- gym_action_feature_matrix(basis_new, action_vec)
    drop(phi_new %*% beta_hat)
  }

  list(
    beta = beta_hat,
    predict_omega = predict_omega,
    basis_specs = basis_specs,
    omega_all = cbind(
      predict_omega(S_mat, rep(0L, n)),
      predict_omega(S_mat, rep(1L, n))
    )
  )
}

mimic_naive_mis <- function(dat, dgp, gamma, ridge = 0.001) {
  flatten_states <- get("gym_flatten_states", mode = "function")

  S_mat <- flatten_states(dat$S)
  Sp_mat <- flatten_states(dat$Sp)
  At_vec <- as.vector(dat$Atilde)
  R_vec <- as.vector(dat$R)
  n <- nrow(S_mat)

  feat_fit <- gym_poly_features(S_mat, return_specs = TRUE)
  basis_specs <- feat_fit$specs
  phi_obs <- gym_action_feature_matrix(feat_fit$x_mat, At_vec)
  phi_pi_sp <- mimic_empirical_policy_features(Sp_mat, dgp$pi_func, specs = basis_specs)
  diff_mat <- phi_obs - gamma * phi_pi_sp
  a_mat <- crossprod(diff_mat, phi_obs) / n

  b_vec <- (1 - gamma) * colMeans(
    mimic_empirical_policy_features(dat$init_states, dgp$pi_func, specs = basis_specs)
  )
  beta_hat <- tryCatch(
    solve(a_mat + ridge * diag(ncol(phi_obs)), b_vec),
    error = function(e) {
      stop("MIS failed with ridge = ", ridge, ": ", conditionMessage(e))
    }
  )

  omega_hat <- drop(phi_obs %*% beta_hat)
  V_hat <- mean(omega_hat * R_vec) / (1 - gamma)

  list(
    V_hat = V_hat,
    omega_hat = omega_hat,
    beta = beta_hat,
    specs = basis_specs
  )
}

mimic_mr_components <- function(dat, dgp, gamma, cross_fit = TRUE) {
  n_traj <- dim(dat$S)[1]
  TT <- dim(dat$S)[2]
  state_dim <- dim(dat$S)[3]
  bridge_clip_q <- 0.98
  bridge_abs_cap <- 10
  omega_clip_q <- 0.98
  omega_abs_cap <- 10
  bridge_out <- select_bridge_index_gym(dat)
  bridge_scores <- bridge_out$bridge_scores
  bridge_index <- bridge_out$bridge_index

  if (!is.null(dgp$bridge_index)) {
    bridge_index <- as.integer(dgp$bridge_index)[1]
  }
  if (!is.finite(bridge_index) || bridge_index < 1L || bridge_index > state_dim) {
    stop("bridge_index must be between 1 and ", state_dim, ".")
  }

  # cross_fit = TRUE  : the published 2-fold scheme. Nuisances are fit on one half and
  #                     evaluated on the other, so nuisance error is orthogonalised, but
  #                     the estimate inherits the variability of a single random split.
  # cross_fit = FALSE : fit the nuisances once on all trajectories and evaluate on the
  #                     same data. Removes split variability and halves the noise in
  #                     every nuisance, at the cost of the orthogonality guarantee.
  fold_specs <- if (isTRUE(cross_fit)) {
    fold_ids <- sample(rep(1:2, length.out = n_traj))
    lapply(1:2, function(k) {
      list(train = which(fold_ids != k), test = which(fold_ids == k))
    })
  } else {
    list(list(train = seq_len(n_traj), test = seq_len(n_traj)))
  }

  direct_terms <- rep(NA_real_, n_traj)
  correction_terms <- rep(NA_real_, n_traj)

  for (spec in fold_specs) {
    train_idx <- spec$train
    test_idx <- spec$test

    dat_train <- gym_subset_dat(dat, train_idx)
    S_te <- gym_flatten_states(dat$S[test_idx, , , drop = FALSE])
    At_te <- as.vector(dat$Atilde[test_idx, , drop = FALSE])
    R_te <- as.vector(dat$R[test_idx, , drop = FALSE])
    Sp_te <- gym_flatten_states(dat$Sp[test_idx, , , drop = FALSE])
    n_test <- nrow(S_te)

    em <- em_gym(dat_train, gamma)
    omega_out <- mimic_solve_omega(em, dat_train, dgp, gamma)
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

    direct_terms[test_idx] <- fqe$predict_V(dat$init_states[test_idx, , drop = FALSE])

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
    phi_fold <- T1 / (1 - gamma) + T2 * gamma / (1 - gamma)

    correction_terms[test_idx] <- rowMeans(
      matrix(phi_fold, nrow = length(test_idx), ncol = TT)
    )
  }

  if (anyNA(direct_terms) || anyNA(correction_terms)) {
    stop("Failed to compute trajectory-level direct and correction terms for MIMIC MR.")
  }

  list(
    n_traj = n_traj,
    TT = TT,
    direct_terms = direct_terms,
    correction_terms = correction_terms,
    psi = direct_terms + correction_terms,
    bridge_index = bridge_index,
    bridge_scores = bridge_scores,
    cross_fit = isTRUE(cross_fit),
    gamma = gamma
  )
}

summarize_mimic_mr_components <- function(comps) {
  psi <- comps$psi
  V_hat <- mean(psi)
  se <- if (length(psi) > 1L) stats::sd(psi) / sqrt(comps$n_traj) else 0

  list(
    direct = mean(comps$direct_terms),
    correction = mean(comps$correction_terms),
    V_hat = V_hat,
    se = se,
    ci_lo = V_hat - 1.96 * se,
    ci_hi = V_hat + 1.96 * se,
    psi = psi,
    direct_terms = comps$direct_terms,
    correction_terms = comps$correction_terms,
    bridge_index = comps$bridge_index,
    bridge_scores = comps$bridge_scores,
    cross_fit = comps$cross_fit
  )
}

mimic_mr_estimator <- function(dat, dgp, gamma, cross_fit = TRUE) {
  comps <- mimic_mr_components(dat, dgp, gamma, cross_fit = cross_fit)
  summarize_mimic_mr_components(comps)
}

mimic_drl_estimate <- function(dat, dgp, gamma,
                               fqe_ridge = 0.001,
                               mis_ridge = 0.001) {
  naive_fqe <- get("naive_fqe_gym", mode = "function")
  flatten_states <- get("gym_flatten_states", mode = "function")

  n_traj <- dim(dat$S)[1]
  TT <- dim(dat$S)[2]

  fqe <- tryCatch(
    naive_fqe(dat, dgp, gamma, ridge = fqe_ridge),
    error = function(e) {
      stop("FQE failed with ridge = ", fqe_ridge, ": ", conditionMessage(e))
    }
  )

  mis <- mimic_naive_mis(dat, dgp, gamma, ridge = mis_ridge)

  S_mat <- flatten_states(dat$S)
  Sp_mat <- flatten_states(dat$Sp)
  At_vec <- as.vector(dat$Atilde)
  R_vec <- as.vector(dat$R)

  Q_obs <- ifelse(
    At_vec == 0L,
    fqe$fit_Q0$predict(S_mat),
    fqe$fit_Q1$predict(S_mat)
  )
  V_sp <- fqe$predict_V(Sp_mat)
  dr_inner <- R_vec + gamma * V_sp - Q_obs
  direct_terms <- fqe$predict_V(dat$init_states)
  correction_terms <- rowMeans(
    matrix(mis$omega_hat * dr_inner / (1 - gamma), nrow = n_traj, ncol = TT)
  )
  psi <- direct_terms + correction_terms
  V_hat <- mean(psi)
  se <- if (length(psi) > 1L) {
    stats::sd(psi) / sqrt(n_traj)
  } else {
    0
  }

  list(
    V_hat = V_hat,
    V_fqe = mean(direct_terms),
    V_mis = mis$V_hat,
    direct = mean(direct_terms),
    direct_terms = direct_terms,
    correction = mean(correction_terms),
    correction_terms = correction_terms,
    psi = psi,
    se = se,
    ci_lo = V_hat - 1.96 * se,
    ci_hi = V_hat + 1.96 * se,
    fqe_ridge = fqe_ridge,
    mis_ridge = mis_ridge
  )
}

mimic_summary_table <- function(mr_out, drl_out = NULL) {
  out <- data.frame(
    method = "LURE",
    estimate = as.numeric(mr_out$V_hat),
    ci_lo = as.numeric(mr_out$ci_lo),
    ci_hi = as.numeric(mr_out$ci_hi),
    se = as.numeric(mr_out$se),
    stringsAsFactors = FALSE
  )

  if (!is.null(drl_out)) {
    out <- rbind(
      out,
      data.frame(
        method = "DRL",
        estimate = as.numeric(drl_out$V_hat),
        ci_lo = as.numeric(drl_out$ci_lo),
        ci_hi = as.numeric(drl_out$ci_hi),
        se = as.numeric(drl_out$se),
        stringsAsFactors = FALSE
      )
    )
  }

  rownames(out) <- NULL
  out
}

mimic_format_table_number <- function(x, digits = 2) {
  x <- as.numeric(x)[1]
  if (is.na(x)) {
    return("NA")
  }
  formatC(x, format = "f", digits = digits)
}

mimic_escape_latex_cell <- function(x) {
  x <- as.character(x)
  x <- gsub("&", "\\\\&", x, fixed = TRUE)
  x <- gsub("%", "\\\\%", x, fixed = TRUE)
  x <- gsub("_", "\\\\_", x, fixed = TRUE)
  x
}

mimic_policy_value_table <- function(result_list,
                                     policy_labels = names(result_list),
                                     digits = 2) {
  if (length(result_list) != length(policy_labels)) {
    stop("result_list and policy_labels must have the same length.")
  }

  table_rows <- Map(function(result, policy_label) {
    lure_ci_text <- if (anyNA(c(result$ci_lo, result$ci_hi))) {
      "NA"
    } else {
      paste0(
        "[",
        mimic_format_table_number(result$ci_lo, digits),
        ", ",
        mimic_format_table_number(result$ci_hi, digits),
        "]"
      )
    }

    drl_ci_text <- if (anyNA(c(result$drl_ci_lo, result$drl_ci_hi))) {
      "NA"
    } else {
      paste0(
        "[",
        mimic_format_table_number(result$drl_ci_lo, digits),
        ", ",
        mimic_format_table_number(result$drl_ci_hi, digits),
        "]"
      )
    }

    data.frame(
      Policy = policy_label,
      `LURE estimate` = mimic_format_table_number(result$estimate, digits),
      `LURE 95% CI` = lure_ci_text,
      `DRL estimate` = mimic_format_table_number(result$drl_estimate, digits),
      `DRL 95% CI` = drl_ci_text,
      stringsAsFactors = FALSE,
      check.names = FALSE
    )
  }, result_list, policy_labels)

  do.call(rbind, table_rows)
}

mimic_policy_value_latex <- function(policy_table,
                                     caption = paste(
                                       "Estimated policy values for the MIMIC-III sepsis analysis.",
                                       "Smaller values indicate lower expected SOFA scores."
                                     ),
                                     label = "tab:mimic") {
  latex_table <- as.data.frame(
    lapply(policy_table, mimic_escape_latex_cell),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  line_break <- strrep("\\", 2)

  body_lines <- apply(latex_table, 1, function(row) {
    paste0(paste(row, collapse = " & "), " ", line_break)
  })

  paste(
    c(
      "\\begin{table}[t]",
      "\\centering",
      paste0("\\caption{", caption, "}"),
      paste0("\\label{", label, "}"),
      "\\begin{tabular}{lcccc}",
      "\\toprule",
      paste0(
        "Policy & LURE estimate & LURE 95\\% CI & DRL estimate & DRL 95\\% CI ",
        line_break
      ),
      "\\midrule",
      body_lines,
      "\\bottomrule",
      "\\end{tabular}",
      "\\end{table}"
    ),
    collapse = "\n"
  )
}

run_mimic_lure <- function(file_path = file.path(mimic_dir, "sepsis_processed_state_action.csv"),
                           gamma = 0.9,
                           horizon = 20L,
                           id_col = "icustayid",
                           time_col = "bloc",
                           action_col = "iv_input",
                           reward_col = "SOFA",
                           action_threshold = 1,
                           state_cols = NULL,
                           policy_type = c("high_dose", "low_dose", "sofa_11"),
                           custom_pi_func = NULL,
                           bridge_index = NULL,
                           max_stays = NULL,
                           reward_transform = identity,
                           reward_timing = c("next", "current"),
                           drop_constant_states = TRUE,
                           cross_fit = TRUE) {
  policy_type <- mimic_normalize_policy_type(policy_type)
  reward_timing <- match.arg(reward_timing)

  ds <- mimic_load_panel(
    file_path = file_path,
    horizon = horizon,
    id_col = id_col,
    time_col = time_col,
    action_col = action_col,
    action_threshold = action_threshold
  )

  if (!is.null(max_stays)) {
    keep_ids <- unique(ds[[id_col]])
    keep_ids <- keep_ids[seq_len(min(length(keep_ids), as.integer(max_stays)))]
    ds <- ds[ds[[id_col]] %in% keep_ids, , drop = FALSE]
  }

  if (is.null(state_cols)) {
    state_cols <- mimic_default_state_cols(
      ds,
      id_col = id_col,
      time_col = time_col,
      action_col = action_col,
      reward_col = reward_col
    )
  }

  if (identical(policy_type, "sofa_11") && !is.null(custom_pi_func)) {
    stop("Use either policy_type = 'sofa_11' or custom_pi_func, not both.")
  }

  if (identical(policy_type, "sofa_11") && !(reward_col %in% state_cols)) {
    state_cols <- unique(c(reward_col, state_cols))
  }

  dat <- mimic_build_gym_dat(
    ds,
    state_cols = state_cols,
    horizon = horizon,
    id_col = id_col,
    time_col = time_col,
    action_col = action_col,
    reward_col = reward_col,
    reward_transform = reward_transform,
    reward_timing = reward_timing,
    drop_constant_states = drop_constant_states
  )

  policy <- if (is.null(custom_pi_func)) {
    mimic_fit_target_policy(dat, policy_type = policy_type)
  } else {
    list(
      policy_name = "custom",
      fit = NULL,
      treat_rate = NA_real_,
      pi_func = function(state_mat) {
        custom_pi_func(mimic_named_state_mat(state_mat, dat$state_names))
      }
    )
  }
  dgp <- generate_mimic_dgp(dat, pi_func = policy$pi_func, bridge_index = bridge_index)
  mr_out <- mimic_mr_estimator(dat, dgp, gamma, cross_fit = cross_fit)
  drl_error <- NULL
  drl_out <- tryCatch(
    mimic_drl_estimate(dat, dgp, gamma),
    error = function(e) {
      drl_error <<- conditionMessage(e)
      list(
        V_hat = NA_real_,
        V_fqe = NA_real_,
        V_mis = NA_real_,
        se = NA_real_,
        ci_lo = NA_real_,
        ci_hi = NA_real_
      )
    }
  )

  bridge_idx <- as.integer(mr_out$bridge_index)
  bridge_state <- if (is.finite(bridge_idx) && bridge_idx >= 1L && bridge_idx <= length(dat$state_names)) {
    dat$state_names[bridge_idx]
  } else {
    NA_character_
  }

  bridge_scores <- mr_out$bridge_scores
  if (is.null(bridge_scores) || length(bridge_scores) != length(dat$state_names)) {
    bridge_scores <- stats::setNames(rep(NA_real_, length(dat$state_names)), dat$state_names)
  }

  list(
    summary = mimic_summary_table(mr_out, drl_out),
    estimate = mr_out$V_hat,
    se = mr_out$se,
    ci_lo = mr_out$ci_lo,
    ci_hi = mr_out$ci_hi,
    mr_out = mr_out,
    drl_estimate = drl_out$V_hat,
    drl_se = drl_out$se,
    drl_ci_lo = drl_out$ci_lo,
    drl_ci_hi = drl_out$ci_hi,
    drl_out = drl_out,
    drl_error = drl_error,
    bridge_state = bridge_state,
    bridge_scores = bridge_scores,
    policy = policy,
    gamma = gamma,
    n_stays = dim(dat$S)[1],
    n_transitions = dim(dat$S)[2],
    state_cols = dat$state_names,
    reward_timing = reward_timing,
    cross_fit = isTRUE(cross_fit),
    dropped_states = dat$dropped_states,
    dat = dat,
    dgp = dgp
  )
}
