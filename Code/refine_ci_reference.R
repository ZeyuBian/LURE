#!/usr/bin/env Rscript
# Refine only the shared continuous baseline target, without adding estimator replications.
root <- dirname(normalizePath(sub("^--file=","",grep("^--file=",commandArgs(),value=TRUE)[1])))
setwd(root)
source("convergence_diagnostics.R")
# Use exactly the validated target-return function from the CI runner.
expressions <- as.list(parse("run_ci_study.R"))
hit <- vapply(expressions,function(x) is.call(x) && identical(x[[1]],as.name("<-")) &&
  identical(x[[2]],as.name("target_returns")),logical(1))
stopifnot(sum(hit)==1L)
eval(expressions[[which(hit)]])
out <- file.path(root,"simulation_results/ci_20_20260913")
path <- file.path(out,"references/Continuous_baseline.rds")
old <- readRDS(path)
if (old$n < 10000000L) {
  stopifnot(old$n==1000000L)
  prior_path <- file.path(out,"references/Continuous_baseline_initial_1000000.rds")
  if (!file.exists(prior_path)) stopifnot(file.copy(path,prior_path))
  e <- diag_load(root,"Continuous")
  seeds <- 54000000L+1:9
  batches <- parallel::mclapply(seeds,function(seed) {
    set.seed(seed)
    ret <- unlist(lapply(1:10,function(i) target_returns(e,old$dgp,old$gamma,100000L,100L)),use.names=FALSE)
    cat("Reference batch",seed,"mean",mean(ret),"\n");flush.console()
    ret
  },mc.cores=2L,mc.set.seed=FALSE)
  stopifnot(all(vapply(batches,length,integer(1))==1000000L))
  refined <- old
  refined$returns <- c(old$returns,unlist(batches,use.names=FALSE))
  refined$n <- length(refined$returns)
  refined$value <- mean(refined$returns)
  refined$mcse <- sd(refined$returns)/sqrt(refined$n)
  refined$additional_batch_seeds <- seeds
  refined$reason <- "One CI endpoint lay inside the initial target estimate's 95% Monte Carlo interval; reference refined to a fixed 10,000,000 total paths."
  saveRDS(refined,paste0(path,".partial"))
  stopifnot(file.rename(paste0(path,".partial"),path))
  file.copy("refine_ci_reference.R",file.path(out,"source_snapshot/refine_ci_reference.R"),overwrite=TRUE)
  print(refined[c("value","mcse","n")])
} else print(old[c("value","mcse","n")])
