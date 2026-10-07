#!/usr/bin/env Rscript

# DRL at gamma = 0.6 under simulated action misclassification, for three
# definitions of the binary action: A = 1{iv_input >= k}, k = 1, 2, 3.
#
# Estimator plotted: linear FQE + marginalised importance weights (omega) on
# quadratic state features -- a DR design whose correction does not cancel.
# The matched-feature DRL (linear omega, which coincides with FQE) is computed
# alongside as a check and saved, not plotted.
#
# Noise: independent symmetric flips of the binary action at rate tau, one
# Uniform draw per action per replication with seed 20260921 + rep -- identical
# to the earlier noise studies -- and the SAME draws for every cut-off, so the
# three panels differ only in how the action is defined.
#
# Covariates: COVARIATES = full (default, the 45-variable state in res.RData),
# reduced (28) or compact (14); see mimic_covariate_set() in MIMIC.R. Non-full sets
# are rebuilt from the CSV through the production pipeline and checked against the
# full data (same stays, rewards and actions; shared state columns identical).
#
# Outputs: figures/drl_noise_cutoffs_gamma06[_<set>].png, .csv (table view), and
#          drl_noise_cutoffs[_<set>].rds   (no suffix for the full set)
#
# Usage from Code/MIMIC:
#   env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
#     [COVARIATES=reduced] Rscript study_drl_noise_cutoffs.R

script_path <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])
script_dir <- if (is.na(script_path)) normalizePath(getwd()) else dirname(normalizePath(script_path))
source(file.path(script_dir, "MIMIC.R"))
suppressPackageStartupMessages({ library(parallel); library(ggplot2) })

GAMMA   <- 0.6
CUTOFFS <- c(1L, 2L, 3L)
TAUS    <- c(0, 0.05, 0.10, 0.20, 0.30, 0.40)
N_REP   <- 5L
N_WORKERS <- as.integer(Sys.getenv("DRL_WORKERS", "6"))
POLICIES <- c(high_dose = "Always IV", sofa_11 = "SOFA-tailored", low_dose = "No IV")
COVSET <- Sys.getenv("COVARIATES", "full")
if (!COVSET %in% c("full", "reduced", "compact")) stop("COVARIATES must be full, reduced or compact")
suffix <- if (COVSET == "full") "" else paste0("_", COVSET)
fig_dir <- file.path(script_dir, "figures"); dir.create(fig_dir, showWarnings = FALSE)
out_rds <- file.path(script_dir, paste0("drl_noise_cutoffs", suffix, ".rds"))

baseline <- new.env()
load(file.path(script_dir, "res.RData"), envir = baseline)
dat <- baseline$results_high$dat
n_traj <- dim(dat$S)[1]; TT <- dim(dat$S)[2]

if (COVSET != "full") {
  # Same cohort as run_mimic_lure(max_stays = 500): the first 500 stay IDs.
  cat("Building the", COVSET, "covariate set from the CSV ...\n"); flush.console()
  ds <- mimic_load_panel(file.path(script_dir, "sepsis_processed_state_action.csv"), horizon = 20L)
  keep_ids <- unique(ds$icustayid)[seq_len(n_traj)]
  ds <- ds[ds$icustayid %in% keep_ids, , drop = FALSE]
  dat_full <- dat
  dat <- mimic_build_gym_dat(ds, state_cols = mimic_covariate_set(COVSET), horizon = 20L,
                             reward_timing = "next", drop_constant_states = TRUE)
  rm(ds)
  shared <- intersect(dat$state_names, dat_full$state_names)
  stopifnot(identical(dat$stay_ids, dat_full$stay_ids),
            identical(dat$R, dat_full$R),
            identical(dat$Atilde, dat_full$Atilde),
            isTRUE(all.equal(unname(dat$S[, , shared]), unname(dat_full$S[, , shared]), tolerance = 0)),
            isTRUE(all.equal(unname(dat$Sp[, , shared]), unname(dat_full$Sp[, , shared]), tolerance = 0)))
  cat(sprintf("  d = %d (%d shared with the full state, identical values; added: %s)\n",
              length(dat$state_names), length(shared),
              paste(setdiff(dat$state_names, dat_full$state_names), collapse = ", ")))
  rm(dat_full)
}
S <- gym_flatten_states(dat$S); Sp <- gym_flatten_states(dat$Sp)
R <- as.vector(dat$R); S0 <- dat$init_states; nm <- dat$state_names

# Raw dose codes for the analysis transitions, in the same (stay, time) order as dat.
cat("Reading dose codes ...\n"); flush.console()
raw <- mimic_read_csv(file.path(script_dir, "sepsis_processed_state_action.csv"))
raw <- raw[raw$icustayid %in% as.numeric(dat$stay_ids), c("icustayid", "bloc", "iv_input")]
raw <- raw[order(raw$icustayid, raw$bloc), ]
iv_mat <- do.call(rbind, lapply(dat$stay_ids, function(id) {
  raw$iv_input[raw$icustayid == as.numeric(id)][seq_len(TT)]
}))
stopifnot(identical(dim(iv_mat), dim(dat$Atilde)),
          all((iv_mat >= 1) == (dat$Atilde == 1L)))   # k = 1 reproduces production
rm(raw)

n_unique <- apply(S, 2, function(x) length(unique(x)))
quad_cols <- which(n_unique > 2)
phi_lin  <- function(X) X
phi_quad <- function(X) cbind(X, X[, quad_cols, drop = FALSE]^2)
PI <- lapply(names(POLICIES), function(p) mimic_fit_target_policy(dat, policy_type = p)$pi_func)
names(PI) <- names(POLICIES)

fqe_fit <- function(pi_func, A, lambda = 0.001) {
  n_iter <- ceiling(log(1e-8) / log(GAMMA))
  pi_sp <- pi_func(Sp); i0 <- which(A == 0L); i1 <- which(A == 1L)
  q0 <- q1 <- rep(0, nrow(S))
  for (it in seq_len(n_iter)) {
    y <- R + GAMMA * ((1 - pi_sp) * q0 + pi_sp * q1)
    f0 <- gym_weighted_ridge_fit(S[i0, , drop = FALSE], y[i0], ridge = lambda)
    f1 <- gym_weighted_ridge_fit(S[i1, , drop = FALSE], y[i1], ridge = lambda)
    q0 <- f0$predict(Sp); q1 <- f1$predict(Sp)
  }
  Q <- function(X, a) if (a == 0) f0$predict(X) else f1$predict(X)
  list(Q = Q, V = function(X) { p <- pi_func(X); (1 - p) * Q(X, 0) + p * Q(X, 1) })
}

mis_fit <- function(pi_func, A, phi, lambda = 0.001) {
  FS <- phi(S); mu <- colMeans(FS)
  sdv <- sqrt(colMeans(sweep(FS, 2, mu)^2)); sdv[!is.finite(sdv) | sdv < 1e-8] <- 1
  z <- function(X) cbind(1, sweep(sweep(phi(X), 2, mu), 2, sdv, "/"))
  phi_obs <- gym_action_feature_matrix(z(S), A)
  phi_pi_sp <- gym_policy_feature_matrix(z(Sp), pi_func(Sp))
  a_mat <- crossprod(phi_obs - GAMMA * phi_pi_sp, phi_obs) / nrow(S)
  b_vec <- (1 - GAMMA) * colMeans(gym_policy_feature_matrix(z(S0), pi_func(S0)))
  drop(phi_obs %*% solve(a_mat + lambda * diag(ncol(phi_obs)), b_vec))
}

perturb <- function(A_mat, rep_id, tau) {
  if (tau == 0) return(as.vector(A_mat))
  set.seed(20260921L + rep_id)
  u <- matrix(runif(length(A_mat)), nrow = nrow(A_mat), ncol = ncol(A_mat))
  A_mat[u < tau] <- 1L - A_mat[u < tau]
  as.vector(A_mat)
}

run_one <- function(job) {
  A_mat <- matrix(as.integer(iv_mat >= job$cutoff), nrow = nrow(iv_mat))
  A <- perturb(A_mat, job$rep, job$tau)
  rows <- list()
  for (pol in names(POLICIES)) {
    fq <- fqe_fit(PI[[pol]], A)
    delta <- R + GAMMA * fq$V(Sp) - ifelse(A == 0L, fq$Q(S, 0), fq$Q(S, 1))
    direct_terms <- fq$V(S0)
    for (des in c("quadratic", "linear")) {
      omega <- mis_fit(PI[[pol]], A, if (des == "quadratic") phi_quad else phi_lin)
      p <- direct_terms + rowMeans(matrix(omega * delta / (1 - GAMMA), nrow = n_traj, ncol = TT))
      V <- mean(p); se <- stats::sd(p) / sqrt(n_traj)
      rows[[length(rows) + 1L]] <- data.frame(
        cutoff = job$cutoff, rep = job$rep, tau = job$tau, omega = des,
        policy = pol, V = V, ci_lo = V - 1.96 * se, ci_hi = V + 1.96 * se,
        p_treated = mean(A), stringsAsFactors = FALSE)
    }
  }
  do.call(rbind, rows)
}

jobs <- list()
for (k in CUTOFFS) {
  jobs[[length(jobs) + 1L]] <- list(cutoff = k, rep = 0L, tau = 0)
  for (tau in TAUS[TAUS > 0]) for (r in seq_len(N_REP)) {
    jobs[[length(jobs) + 1L]] <- list(cutoff = k, rep = r, tau = tau)
  }
}
cat(sprintf("gamma = %.1f | cut-offs %s | %d fits on %d workers\n",
            GAMMA, paste(CUTOFFS, collapse = ","), length(jobs), N_WORKERS)); flush.console()
t0 <- proc.time()[["elapsed"]]
out <- mclapply(jobs, run_one, mc.cores = N_WORKERS, mc.preschedule = FALSE)
bad <- vapply(out, function(o) !is.data.frame(o), logical(1))
if (any(bad)) stop(sum(bad), " fits failed: ", paste(unique(unlist(out[bad])), collapse = "; "))
vals <- do.call(rbind, out)
cat(sprintf("done in %.0fs\n", proc.time()[["elapsed"]] - t0))
saveRDS(vals, out_rds)

# ---- summary: mean over replications -------------------------------------------
agg <- aggregate(cbind(V, ci_lo, ci_hi) ~ cutoff + tau + omega + policy, data = vals, FUN = mean)
share <- vapply(CUTOFFS, function(k) mean(iv_mat >= k), numeric(1)); names(share) <- CUTOFFS
cut_lab <- c(`1` = "any fluid", `2` = "≥ 45 mL / 4 h", `3` = "≥ 150 mL / 4 h")

tab <- agg[agg$omega == "quadratic", ]
tab <- tab[order(tab$cutoff, match(tab$policy, names(POLICIES)), tab$tau), ]
csv <- data.frame(cutoff = paste0("iv_input >= ", tab$cutoff), policy = unname(POLICIES[tab$policy]),
                  tau = tab$tau, value = round(tab$V, 3), ci_lo = round(tab$ci_lo, 3), ci_hi = round(tab$ci_hi, 3))
csv_file <- file.path(fig_dir, paste0("drl_noise_cutoffs_gamma06", suffix, ".csv"))
write.csv(csv, csv_file, row.names = FALSE)

cat(sprintf("\n=== DRL values and 95%% CIs (gamma = 0.6, %s state, d = %d, mean over 5 noise draws) ===\n",
            COVSET, length(nm)))
for (k in CUTOFFS) {
  cat(sprintf("\nCut-off iv_input >= %d (%s, %.1f%% treated)\n", k, cut_lab[[as.character(k)]], 100 * share[[as.character(k)]]))
  cat(sprintf("%6s | %-22s | %-22s | %-22s\n", "tau", POLICIES[[1]], POLICIES[[2]], POLICIES[[3]]))
  for (tau in TAUS) {
    cells <- vapply(names(POLICIES), function(pol) {
      s <- tab[tab$cutoff == k & tab$tau == tau & tab$policy == pol, ]
      sprintf("%.2f [%.2f, %.2f]", s$V, s$ci_lo, s$ci_hi)
    }, character(1))
    cat(sprintf("%5.0f%% | %-22s | %-22s | %-22s\n", 100 * tau, cells[1], cells[2], cells[3]))
  }
}
chk <- merge(agg[agg$omega == "quadratic", ], agg[agg$omega == "linear", ],
             by = c("cutoff", "tau", "policy"))
cat(sprintf("\nCheck: quadratic-omega vs linear-omega DRL, max |difference in value| = %.3f\n",
            max(abs(chk$V.x - chk$V.y))))

# ---- figure --------------------------------------------------------------------
tok <- list(surface = "#fcfcfb", text = "#0b0b0b", text2 = "#52514e", muted = "#898781",
            grid = "#e1e0d9", axis = "#c3c2b7")
series <- c("Always IV" = "#2a78d6", "SOFA-tailored" = "#eb6834", "No IV" = "#1baf7a")

pd <- tab
pd$policy_label <- factor(unname(POLICIES[pd$policy]), levels = unname(POLICIES))
pd$panel <- factor(pd$cutoff, levels = CUTOFFS, labels = vapply(CUTOFFS, function(k) sprintf(
  "Cut-off iv_input ≥ %d: %s\n%.1f%% of actions treated", k, cut_lab[[as.character(k)]],
  100 * share[[as.character(k)]]), character(1)))
pd$x <- 100 * pd$tau
dodge <- position_dodge(width = 2.4)

g <- ggplot(pd, aes(x = x, y = V, colour = policy_label, group = policy_label)) +
  geom_line(position = dodge, linewidth = 0.55, lineend = "round", linejoin = "round") +
  geom_linerange(aes(ymin = ci_lo, ymax = ci_hi), position = dodge, linewidth = 0.45,
                 show.legend = FALSE) +
  geom_point(aes(fill = policy_label), position = dodge, shape = 21, size = 2.3,
             stroke = 0.6, colour = tok$surface, show.legend = FALSE) +
  facet_wrap(~panel, nrow = 1) +
  scale_colour_manual(values = series, name = NULL) +
  scale_fill_manual(values = series, name = NULL) +
  scale_x_continuous(breaks = 100 * TAUS, labels = paste0(100 * TAUS, "%"),
                     expand = expansion(add = 2.5)) +
  labs(title = "DRL policy values under simulated action misclassification",
       subtitle = paste0("γ = 0.6  ·  ",
                         if (COVSET == "full") "" else sprintf("%s state (%d covariates)  ·  ", COVSET, length(nm)),
                         "points are the mean over 5 noise draws, bars are 95% CIs  ·  lower SOFA is better"),
       x = "Share of recorded actions flipped (τ)",
       y = "Estimated policy value\n(discounted SOFA)",
       caption = paste0("DRL = linear FQE with marginalised importance weights on quadratic state features.  ",
                        "“IV” means an IV dose at or above the panel’s cut-off.")) +
  theme_minimal(base_size = 11) +
  theme(
    plot.background = element_rect(fill = tok$surface, colour = NA),
    panel.background = element_rect(fill = tok$surface, colour = NA),
    panel.grid.major.y = element_line(colour = tok$grid, linewidth = 0.3),
    panel.grid.major.x = element_blank(), panel.grid.minor = element_blank(),
    axis.line.x = element_line(colour = tok$axis, linewidth = 0.3),
    axis.ticks = element_blank(),
    axis.text = element_text(colour = tok$text2, size = 9),
    axis.title = element_text(colour = tok$text2, size = 10),
    strip.text = element_text(colour = tok$text, size = 10, face = "bold", hjust = 0, lineheight = 1.1),
    plot.title = element_text(colour = tok$text, face = "bold", size = 13),
    plot.subtitle = element_text(colour = tok$text2, size = 10, margin = margin(b = 8)),
    plot.caption = element_text(colour = tok$muted, size = 8.5, hjust = 0, margin = margin(t = 8)),
    plot.title.position = "plot", plot.caption.position = "plot",
    legend.position = "top", legend.justification = "left",
    legend.text = element_text(colour = tok$text, size = 10),
    legend.margin = margin(0, 0, 0, 0), legend.key.width = unit(18, "pt"),
    panel.spacing = unit(18, "pt"), plot.margin = margin(14, 16, 10, 12)
  ) +
  guides(colour = guide_legend(override.aes = list(linewidth = 0.55)), fill = "none")

png_file <- file.path(fig_dir, paste0("drl_noise_cutoffs_gamma06", suffix, ".png"))
ggsave(png_file, g, width = 11.5, height = 4.9, dpi = 200, bg = tok$surface)
cat("\nFigure:", png_file, "\nTable: ", csv_file, "\n")
