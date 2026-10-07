#!/usr/bin/env Rscript

# Why is DRL insensitive to misclassification of the observed action?
#
# Five checks, none of which refit LURE:
#   1. Is there any action effect in the data, conditional on the state?
#      (regression of the reward on S with and without A~)
#   2. Do all target policies collapse to the behaviour policy's own value?
#   3. Does relabelling actions change the fitted Q-functions at all?
#      (FQE at tau = 0 vs tau = 0.4, same noise draw as the sensitivity study)
#   4. Which DRL component, direct or correction, moves with tau?
#   5. Is the correction a genuine doubly robust correction, or does it vanish once
#      FQE has converged?
#
# Usage from Code/MIMIC:
#   Rscript study_drl_insensitivity.R    (reads drl_action_noise.rds for check 4)

script_path <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])
script_dir <- if (is.na(script_path)) normalizePath(getwd()) else dirname(normalizePath(script_path))
source(file.path(script_dir, "MIMIC.R"))

baseline <- new.env()
load(file.path(script_dir, "res.RData"), envir = baseline)
dat <- baseline$results_high$dat
bridge_index <- as.integer(baseline$results_high$mr_out$bridge_index)
n_traj <- dim(dat$S)[1]; TT <- dim(dat$S)[2]
S <- gym_flatten_states(dat$S); A <- as.vector(dat$Atilde); R <- as.vector(dat$R)
X <- as.data.frame(S); names(X) <- paste0("f", seq_len(ncol(S)))

cat("=== 1. Is there an action effect given the state? (reward = SOFA_{t+1}) ===\n\n")
fit_s  <- lm(R ~ ., data = cbind(R = R, X))
fit_sa <- lm(R ~ ., data = cbind(R = R, X, A = A))
co <- summary(fit_sa)$coefficients["A", ]
r2_s <- summary(fit_s)$r.squared; r2_sa <- summary(fit_sa)$r.squared
cat(sprintf("  R^2, reward on state          : %.4f\n", r2_s))
cat(sprintf("  R^2, reward on state + A~     : %.4f   (gain %.2e)\n", r2_sa, r2_sa - r2_s))
cat(sprintf("  coefficient on A~             : %+.4f  (se %.4f, t = %.2f)\n", co[1], co[2], co[3]))
cat(sprintf("  residual sd                   : %.4f\n", summary(fit_sa)$sigma))
cat(sprintf("  unadjusted: mean R | A~=1 - mean R | A~=0 = %+.4f\n", mean(R[A == 1]) - mean(R[A == 0])))
cat(sprintf("  arm sizes: A~=0 %d (%.1f%%), A~=1 %d (%.1f%%)\n\n",
            sum(A == 0), 100 * mean(A == 0), sum(A == 1), 100 * mean(A == 1)))

cat("=== 2. Do all policies collapse to the behaviour policy's value? ===\n")
cat("    behaviour value = empirical discounted return of the observed data, per step\n\n")
drl_sweep <- readRDS(file.path(script_dir, "drl_gamma_sweep.rds"))$summary
Rm <- dat$R
cat(sprintf("%6s %12s | %10s %10s %10s | %12s\n", "gamma", "behaviour", "Always IV", "SOFA", "No IV", "max |gap|"))
for (gam in c(0.4, 0.5, 0.7, 0.9)) {
  w <- gam^(seq_len(TT) - 1)
  beh <- (1 - gam) * mean(Rm %*% w) / sum((1 - gam) * w)   # renormalised for truncation at T
  s <- drl_sweep[drl_sweep$gamma == gam, ]
  v <- setNames(s$per_step_V, s$policy)
  cat(sprintf("%6.1f %12.4f | %10.4f %10.4f %10.4f | %12.4f\n", gam, beh,
              v[["high_dose"]], v[["sofa_11"]], v[["low_dose"]], max(abs(v - beh))))
}

cat("\n=== 3. Does relabelling actions change the fitted Q-functions? ===\n")
cat("    FQE at tau = 0 vs tau = 0.4 (noise draw of replication 1), gamma = 0.5\n\n")
set.seed(20260921L + 1L)
u <- matrix(runif(length(dat$Atilde)), nrow = nrow(dat$Atilde), ncol = ncol(dat$Atilde))
dat40 <- dat; dat40$Atilde[u < 0.4] <- 1L - dat40$Atilde[u < 0.4]
cat(sprintf("    labels changed on %.1f%% of transitions\n\n", 100 * mean(dat40$Atilde != dat$Atilde)))
for (pol in c("high_dose", "low_dose")) {
  pf <- mimic_fit_target_policy(dat, policy_type = pol)
  dgp <- generate_mimic_dgp(dat, pi_func = pf$pi_func, bridge_index = bridge_index)
  set.seed(23); f0 <- naive_fqe_gym(dat,   dgp, 0.5, ridge = 0.001)
  set.seed(23); f4 <- naive_fqe_gym(dat40, dgp, 0.5, ridge = 0.001)
  q00 <- f0$fit_Q0$predict(S); q01 <- f0$fit_Q1$predict(S)
  q40 <- f4$fit_Q0$predict(S); q41 <- f4$fit_Q1$predict(S)
  cat(sprintf("  policy %-10s sd(Q)=%.3f | Q0 shift: mean %+.4f sd %.4f | Q1 shift: mean %+.4f sd %.4f\n",
              pol, sd(q00), mean(q40 - q00), sd(q40 - q00), mean(q41 - q01), sd(q41 - q01)))
  cat(sprintf("  %-17s Q1-Q0 at tau=0: mean %+.4f | at tau=0.4: mean %+.4f\n", "",
              mean(q01 - q00), mean(q41 - q40)))
  vi0 <- f0$predict_V(dat$init_states); vi4 <- f4$predict_V(dat$init_states)
  cat(sprintf("  %-17s value at initial states: tau=0 %.4f, tau=0.4 %.4f\n\n", "", mean(vi0), mean(vi4)))
}

cat("=== 4. Which DRL component moves with tau? (gamma = 0.5, mean over reps) ===\n\n")
noise <- readRDS(file.path(script_dir, "drl_action_noise.rds"))$values
cat(sprintf("%6s %-15s %10s %12s %10s\n", "tau", "policy", "direct", "correction", "V"))
for (tau in sort(unique(noise$tau))) {
  for (pol in c("high_dose", "low_dose")) {
    s <- noise[noise$gamma == 0.5 & noise$tau == tau & noise$policy == pol, ]
    cat(sprintf("%5.0f%% %-15s %10.4f %12.4f %10.4f\n", 100 * tau, s$policy_label[1],
                mean(s$direct), mean(s$correction), mean(s$V)))
  }
}

cat("\n=== 5. Is DRL's correction a real doubly robust correction? ===\n")
cat("    With linear FQE and linear MIS on the same features, the correction cancels at\n")
cat("    the FQE fixed point: the TD residual is orthogonal to the features omega is built\n")
cat("    from. Any nonzero correction should then shrink as FQE iterates to convergence.\n\n")
Sp <- gym_flatten_states(dat$Sp)
pf <- mimic_fit_target_policy(dat, policy_type = "high_dose")
dgp <- generate_mimic_dgp(dat, pi_func = pf$pi_func, bridge_index = bridge_index)
mis <- mimic_naive_mis(dat, dgp, 0.9, ridge = 0.001)
cat("  Always IV, gamma = 0.9 (production uses n_iter = 20; unconverged share 0.9^20 = 12%)\n")
for (it in c(20, 50, 100, 200)) {
  set.seed(23); fq <- naive_fqe_gym(dat, dgp, 0.9, n_iter = it, ridge = 0.001)
  q_obs <- ifelse(A == 0L, fq$fit_Q0$predict(S), fq$fit_Q1$predict(S))
  td <- R + 0.9 * fq$predict_V(Sp) - q_obs
  dir <- mean(fq$predict_V(dat$init_states)); corr <- mean(mis$omega_hat * td) / (1 - 0.9)
  cat(sprintf("    n_iter = %3d   direct = %8.3f   correction = %+8.4f   DRL = %8.3f\n",
              it, dir, corr, dir + corr))
}
