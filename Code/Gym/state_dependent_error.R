## Current-state-only recording errors. No reward or next state enters p(flip).
cartpole_error_probabilities <- function(states, tau, mode = "state_dependent",
                                          strength = .75, center = 0, scale = 1) {
  mode <- match.arg(mode, c("constant", "state_dependent"))
  if (length(tau) != 1L || !is.finite(tau) || tau < 0 || tau > 1 ||
      (mode == "state_dependent" && tau >= .5)) stop("Invalid error rate.")
  if (length(strength) != 1L || !is.finite(strength) || strength < 0 || strength >= 1 ||
      length(center) != 1L || !is.finite(center) || length(scale) != 1L ||
      !is.finite(scale) || scale <= 0) stop("Invalid error strength, center, or scale.")
  s <- as.matrix(states)
  if (ncol(s) != 4L || any(!is.finite(s))) stop("Need finite four-coordinate CartPole states.")
  amplitude <- if (mode == "constant") 0 else strength * min(tau, .5 - tau)
  tau + amplitude * tanh((s[, 1] - center) / scale)
}

cartpole_calibrate_error <- function(states) {
  x <- as.matrix(states)[, 1]
  if (any(!is.finite(x))) stop("Calibration states must be finite.")
  scale <- IQR(x)
  if (!is.finite(scale) || scale <= 0) stop("Calibration position IQR must be positive.")
  center <- uniroot(function(c) mean(tanh((x - c) / scale)),
                    interval = range(x), tol = 1e-12)$root
  list(center = center, scale = scale, calibration_n = length(x),
       mean_score = mean(tanh((x - center) / scale)))
}

cartpole_record_errors <- function(dat, tau, uniforms, mode = "state_dependent",
                                   strength = .75, center = 0, scale = 1) {
  if (!identical(dat$env_name, "CartPole-v1")) stop("CartPole only.")
  if (!identical(dim(uniforms), dim(dat$A)) || any(!is.finite(uniforms)) ||
      any(uniforms < 0 | uniforms >= 1)) stop("Provide one uniform in [0,1) per action.")
  p <- matrix(cartpole_error_probabilities(gym_flatten_states(dat$S), tau, mode,
               strength, center, scale), nrow = nrow(dat$A), ncol = ncol(dat$A))
  dat$Atilde <- ifelse(uniforms < p, 1L - dat$A, dat$A)
  dat$misclassification_prob <- p
  dat$misclassification <- list(mode = mode, tau = tau, state_strength = strength,
    error_center = center, error_scale = scale, expected_rate = mean(p),
    realized_rate = mean(dat$A != dat$Atilde))
  dat
}
