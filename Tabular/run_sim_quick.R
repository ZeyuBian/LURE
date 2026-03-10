

source("Methods.R")
library(rpart)

# Source function definitions only
lines <- readLines("simulation_tabular.R")
main_idx <- grep("^dgp\\s*<-\\s*generate_dgp", lines)[1]
eval(parse(text = lines[1:(main_idx - 1)]))

dgp   <- generate_dgp()
gamma <- 0.6
n_rep <- 20
V_true <- compute_true_value(dgp, gamma)$V_value
cat("True V(pi) =", V_true, "\n")

eps <- 0.25
methods <- c("FQE","SIS","MIS","DRL","MR")
res <- matrix(NA, n_rep, 5, dimnames = list(NULL, methods))

for (rep in 1:n_rep) {
  set.seed(rep)
  if (rep %% 10 == 0) cat("  rep", rep, "/", n_rep, "\n")
  r <- one_rep(dgp, N = 20, TT = 30, epsilon = eps, gamma = gamma)
  res[rep, ] <- r[methods]
}

bias <- colMeans(res, na.rm = TRUE) - V_true
rmse <- sqrt(colMeans((res - V_true)^2, na.rm = TRUE))
cat("\nBias:\n"); print(round(bias, 3))
cat("\nRMSE:\n"); print(round(rmse, 3))

## ---- Boxplot ----
cols <- c("#E41A1C", "#FF7F00", "#984EA3", "#377EB8", "#4DAF4A")

par(mar = c(4.5, 4.5, 2.5, 1))
boxplot(res, col = cols, names = methods,
        ylab = expression("Estimated " * V(pi)),
        main = bquote(n[rep] == .(n_rep) ~ "Replications," ~ epsilon == .(eps)),
        cex.lab = 1.2, cex.axis = 1.1, outline = FALSE)

abline(h = V_true, lty = 2, lwd = 2, col = "black")

text(5.3, V_true,
     bquote(V(pi) == .(round(V_true, 2))),
     pos = 3, cex = 0.9)