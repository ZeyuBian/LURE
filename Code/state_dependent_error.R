## Shared bounded recording-error utilities. No estimator or IF changes.
lure_error_amplitude <- function(epsilon, misclassification = "constant", state_strength = .75) {
  misclassification <- match.arg(misclassification, c("constant", "state_dependent"))
  if (length(epsilon) != 1L || !is.finite(epsilon) || epsilon < 0 || epsilon > 1) {
    stop("epsilon must be a finite probability in [0,1].")
  }
  if (length(state_strength) != 1L || !is.finite(state_strength) ||
      state_strength < 0 || state_strength >= 1) stop("state_strength must be in [0,1).")
  if (misclassification == "constant") return(0)
  if (epsilon >= .5) stop("State-dependent errors require epsilon < 0.5.")
  state_strength * min(epsilon, .5 - epsilon)
}

lure_error_probabilities <- function(score, epsilon, misclassification = "constant", state_strength = .75) {
  amplitude <- lure_error_amplitude(epsilon, misclassification, state_strength)
  if (any(!is.finite(score)) || any(abs(score) > 1)) stop("State scores must be finite and in [-1,1].")
  epsilon + amplitude * score
}

lure_error_metadata <- function(probabilities, score, original_labels, recorded_labels,
                                 epsilon, misclassification, state_strength) {
  list(mode = misclassification, nominal_rate = epsilon, state_strength = state_strength,
    expected_rate = mean(probabilities), realized_rate = mean(original_labels != recorded_labels),
    min_probability = min(probabilities), max_probability = max(probabilities),
    mean_score = mean(score), n_transitions = length(probabilities))
}

lure_error_diagnostic_row <- function(metadata) {
  as.data.frame(metadata[c("mode", "nominal_rate", "state_strength", "expected_rate",
    "realized_rate", "min_probability", "max_probability", "mean_score", "n_transitions")],
    stringsAsFactors = FALSE)
}
