#!/usr/bin/env Rscript

# What is LURE's latent "action"? Diagnostics on the full-sample EM fit.
#
#   1. Agreement of the EM latent class with the recorded action A~ (vs chance),
#      and the measurement-model gap mu1 - mu0 = P(A~=1 | S, latent=1) - P(... =0).
#   2. Which likelihood channel drives the E-step: log-likelihood ratio
#      (latent 1 vs 0) contributed by the prior b(s), A~, the reward, and the
#      45-dimensional next state.
#   3. Which next-state variables dominate that channel.
#   4. Agreement of the latent class with "liver labs carried forward" (SGPT and
#      SGOT unchanged from t to t+1), and how often labs vs vitals are unchanged.
#
# Uses lure_em_full.rds if present (written by study_lure_gamma.R); otherwise fits
# the same EM (seed 23, all 500 stays, ~4 min) without saving it.
#
# Usage from Code/MIMIC:
#   env OPENBLAS_NUM_THREADS=1 Rscript study_lure_latent_class.R

script_path <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])
script_dir <- if (is.na(script_path)) normalizePath(getwd()) else dirname(normalizePath(script_path))
source(file.path(script_dir, "MIMIC.R"))

baseline <- new.env()
load(file.path(script_dir, "res.RData"), envir = baseline)
dat <- baseline$results_high$dat

em_cache <- file.path(script_dir, "lure_em_full.rds")
if (file.exists(em_cache)) {
  em <- readRDS(em_cache)
} else {
  cat("No EM cache; fitting the full-sample EM (seed 23, ~4 min) ...\n"); flush.console()
  set.seed(23)
  em <- em_gym(dat, gamma = NA_real_)   # em_gym does not use gamma
}

S <- gym_flatten_states(dat$S); Sp <- gym_flatten_states(dat$Sp)
A <- as.vector(dat$Atilde); R <- as.vector(dat$R); nm <- dat$state_names
cls <- as.integer(em$eta[, 2] > 0.5)

cat("=== 1. Does the latent class track the recorded action? ===\n\n")
mu0 <- em$predict_mu(S, 0); mu1 <- em$predict_mu(S, 1)
cat(sprintf("  P(latent = 1) = %.3f   P(A~ = 1) = %.3f\n", mean(cls), mean(A)))
cat(sprintf("  agreement latent vs A~ = %.3f   (chance given the marginals = %.3f)\n",
            mean(cls == A), mean(cls) * mean(A) + (1 - mean(cls)) * (1 - mean(A))))
cat(sprintf("  mu1 - mu0 = %.3f   (mean mu0 = %.3f, mean mu1 = %.3f)\n",
            mean(mu1 - mu0), mean(mu0), mean(mu1)))
cat(sprintf("  posterior sharpness, mean max(eta) = %.3f\n\n", mean(pmax(em$eta[, 1], em$eta[, 2]))))

cat("=== 2. Which channel drives the E-step? (log-likelihood ratio, latent 1 vs 0) ===\n\n")
b <- em$predict_b(S)
tR0 <- em$predict_theta_R(S, 0); tR1 <- em$predict_theta_R(S, 1)
P0 <- em$predict_theta_Sp(S, 0); P1 <- em$predict_theta_Sp(S, 1)
llr <- list(
  "prior b(s)"            = log(b) - log(1 - b),
  "recorded action A~"    = dbinom(A, 1, mu1, log = TRUE) - dbinom(A, 1, mu0, log = TRUE),
  "reward"                = dnorm(R, tR1, em$sigma_R[2], log = TRUE) - dnorm(R, tR0, em$sigma_R[1], log = TRUE)
)
llr_j <- sapply(seq_along(nm), function(j) {
  dnorm(Sp[, j], P1[, j], em$sigma_Sp[2, j], log = TRUE) - dnorm(Sp[, j], P0[, j], em$sigma_Sp[1, j], log = TRUE)
})
llr[[sprintf("next state (%d vars)", length(nm))]] <- rowSums(llr_j)
total <- Reduce(`+`, llr)
for (k in names(llr)) {
  cat(sprintf("  %-24s mean |LLR| = %10.3f   cor with total = %.3f\n",
              k, mean(abs(llr[[k]])), cor(llr[[k]], total)))
}

cat("\n=== 3. Next-state variables that dominate (top 8 by sd of their LLR) ===\n\n")
v <- apply(llr_j, 2, sd); o <- order(v, decreasing = TRUE)[1:8]
for (k in o) {
  cat(sprintf("  %-18s sd(LLR) = %9.1f   sigma(latent 1)/sigma(latent 0) = %.2f\n",
              nm[k], v[k], em$sigma_Sp[2, k] / em$sigma_Sp[1, k]))
}

cat("\n=== 4. Is the latent class 'labs carried forward'? ===\n\n")
unchanged <- function(var) abs(Sp[, match(var, nm)] - S[, match(var, nm)]) < 1e-8
cf <- unchanged("SGPT") & unchanged("SGOT")
cat(sprintf("  SGPT and SGOT unchanged t -> t+1: %.1f%% of transitions\n", 100 * mean(cf)))
cat(sprintf("  agreement latent vs carried forward = %.3f   (vs A~: %.3f)\n\n", mean(cls == cf), mean(cls == A)))
print(table(latent = cls, liver_labs_carried_forward = cf))
cat("\n  share of transitions with the value unchanged t -> t+1:\n")
for (var in c("SGPT", "SGOT", "Albumin", "Total_bili", "BUN", "Platelets_count", "HR", "MeanBP", "RR", "SpO2")) {
  cat(sprintf("    %-16s %5.1f%%\n", var, 100 * mean(unchanged(var))))
}
cat(sprintf("\n  P(A~ = 1) when labs carried forward: %.3f   after a new lab draw: %.3f\n",
            mean(A[cf]), mean(A[!cf])))
