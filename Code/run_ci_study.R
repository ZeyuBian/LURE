#!/usr/bin/env Rscript
# Rscript run_ci_study.R validate|reference|run|report [CORES]
# Current production MR estimator, 20 saved data sets per setting/condition.
args <- commandArgs(TRUE)
stopifnot(length(args) >= 1L, args[1] %in% c("validate", "reference", "run", "report"))
root <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(), value=TRUE)[1])))
setwd(root)
source("convergence_diagnostics.R")
input <- file.path(root, "simulation_results/convergence_20_20260913")
out <- file.path(root, "simulation_results/ci_20_20260913")
dir.create(out, recursive=TRUE, showWarnings=FALSE)
settings <- c("Tabular", "Continuous")
scenarios <- c("typical", "hard", "weak_anchor", "weak_proxy", "state_dependent")
cores <- if (length(args) >= 2L) as.integer(args[2]) else 2L
sources <- c("Tabular/Methods.R", "Continuous/Methods_continuous.R", "state_dependent_error.R")
for (f in sources) {
  stopifnot(unname(tools::md5sum(f)) == unname(tools::md5sum(file.path(input, "source_snapshot", f))))
}
atomic_save <- function(x, path) {
  dir.create(dirname(path), recursive=TRUE, showWarnings=FALSE)
  saveRDS(x, paste0(path, ".partial"))
  stopifnot(file.rename(paste0(path, ".partial"), path))
}
load_case <- function(setting, scenario, rep) {
  readRDS(file.path(input, "main", setting, scenario, sprintf("rep_%03d.rds", rep)))
}
value_seed <- function(setting, rep) 43000000L + (setting == "Continuous") * 100000L + rep * 100L

# Only log EM updates; retain the original two-start wrapper, RNG stream,
# fold allocation, restart selection, relabeling, and downstream estimator.
instrument_continuous <- function(e) {
  e$.ci_trace <- list()
  e$.em_single_run <- function(...) {
    final <- NULL
    hook <- function(fr) {
      final <<- data.frame(n_iter=fr$iter, loglik=fr$loglik,
        last_delta=max(abs(fr$eta-fr$eta_old)), tol=fr$tol,
        converged=max(abs(fr$eta-fr$eta_old)) < fr$tol)
    }
    raw <- do.call(e$.diagnostic_single, c(list(...), list(diag_hook=hook)))
    id <- length(e$.ci_trace) + 1L
    e$.ci_trace[[id]] <- cbind(data.frame(fold=(id-1L) %/% 2L+1L, restart=(id-1L) %% 2L+1L), final)
    raw
  }
  invisible(e)
}

fit_case <- function(setting, scenario, rep, original=FALSE) {
  x <- load_case(setting, scenario, rep)
  e <- diag_load(root, setting)
  if (setting == "Tabular" && !original) {
    cached <- diag_tab_build(x$runs[[2]]$fit$raw, x$dat)
    e$em_tabular <- function(...) cached
  }
  if (setting == "Continuous" && !original) instrument_continuous(e)
  seed <- value_seed(setting, rep)
  set.seed(seed)
  warning_messages <- character()
  error <- NULL
  start <- proc.time()[["elapsed"]]
  result <- tryCatch(withCallingHandlers(e$.original_value(x$dat, x$spec$dgp, x$spec$gamma),
    warning=function(w) { warning_messages <<- c(warning_messages, conditionMessage(w)); invokeRestart("muffleWarning") }),
    error=function(err) { error <<- conditionMessage(err); NULL })
  diagnostics <- NULL
  if (!original) {
    if (setting == "Tabular") {
      f <- x$runs[[2]]$fit
      diagnostics <- data.frame(fold=1L, restart=1L, n_iter=f$raw$n_iter,
        loglik=f$raw$loglik, last_delta=tail(f$trace$delta_max,1), tol=f$tol,
        converged=f$status == "converged", selected=TRUE)
    } else if (length(e$.ci_trace)) {
      diagnostics <- do.call(rbind, e$.ci_trace)
      diagnostics$selected <- FALSE
      for (fold in unique(diagnostics$fold)) {
        ix <- which(diagnostics$fold == fold)
        diagnostics$selected[ix[which.max(diagnostics$loglik[ix])]] <- TRUE
      }
    }
  }
  list(setting=setting, scenario=scenario, rep=rep, spec=x$spec,
       data_seed=attr(x$dat, "diagnostic_seed"), value_seed=seed,
       result=result, diagnostics=diagnostics, error=error,
       warnings=unique(warning_messages), warning_count=length(warning_messages),
       elapsed=proc.time()[["elapsed"]]-start, complete=TRUE)
}

# Independent target-policy paths; vectorization is over trajectories.
# Real reward noise is included, exactly as in the production rollout.
# base1/base2 also handle the altered dynamics of the weak-proxy condition.
target_returns <- function(e, dgp, gamma, n, horizon=100L) {
  s <- e$draw_initial_states(dgp, n)
  s1 <- s[,1]; s2 <- s[,2]
  value <- numeric(n)
  discount <- 1
  for (t in seq_len(horizon)) {
    a <- rbinom(n, 1L, dgp$pi_func(s1, s2))
    sp1 <- dgp$base1*s1 + dgp$tr_shift*(2*a-1) + dgp$s1_a_int*s1*a + rnorm(n,0,dgp$sigma_tr)
    sp2 <- dgp$base2*s2 - dgp$tr_shift*(2*a-1) + dgp$s2_a_int*s2*a + rnorm(n,0,dgp$sigma_tr)
    reward <- 1+s1+.5*s2+1.5*a + rnorm(n,0,dgp$sigma_R)
    value <- value + discount*reward
    discount <- discount*gamma
    s1 <- sp1; s2 <- sp2
  }
  value
}

if (args[1] == "validate") {
  checks <- character()
  for (scenario in scenarios) {
    a <- fit_case("Tabular",scenario,1L)
    b <- fit_case("Tabular",scenario,1L,original=TRUE)
    stopifnot(is.null(a$error),is.null(b$error),isTRUE(all.equal(a$result,b$result,tolerance=1e-10)))
    checks <- c(checks,paste("Cached Tabular fit reproduces original estimator:",scenario))
  }
  for (scenario in c("hard","weak_proxy")) {
    a <- fit_case("Continuous",scenario,1L)
    b <- fit_case("Continuous",scenario,1L,original=TRUE)
    stopifnot(is.null(a$error),is.null(b$error),identical(a$result,b$result),nrow(a$diagnostics)==4L)
    checks <- c(checks,paste("Continuous instrumentation is bitwise identical to original:",scenario))
  }
  e <- diag_load(root,"Continuous")
  for (scenario in c("typical","weak_proxy")) {
    spec <- diag_spec(e,"Continuous",scenario)
    f <- e$compute_true_value_continuous
    body(f) <- diag_walk(body(f),function(x) {
      if (identical(x,quote((1/2) * s1))) quote(dgp$base1 * s1)
      else if (identical(x,quote((1/3) * s2))) quote(dgp$base2 * s2) else x
    })
    for (seed in 1:5) {
      set.seed(seed); a <- target_returns(e,spec$dgp,spec$gamma,1L,100L)
      set.seed(seed); b <- f(spec$dgp,spec$gamma,n_mc=1L,TT_mc=100L)$V_value
      stopifnot(isTRUE(all.equal(a,b,tolerance=1e-12)))
    }
    checks <- c(checks,paste("Vectorized target rollout matches scalar path for five seeds:",scenario))
  }
  writeLines(c(checks,capture.output(sessionInfo())),file.path(out,"validation.txt"))
  cat(paste(checks,collapse="\n"),"\n")
}

if (args[1] == "reference") {
  references <- list()
  for (setting in settings) for (scenario in scenarios) {
    key <- paste(setting, if (scenario=="weak_proxy") "weak_proxy" else "baseline",sep="_")
    path <- file.path(out,"references",paste0(key,".rds"))
    if (is.null(references[[key]])) {
      if (file.exists(path)) references[[key]] <- readRDS(path)
      else {
        e <- diag_load(root,setting); spec <- diag_spec(e,setting,scenario)
        if (setting == "Tabular") {
          r <- list(value=e$compute_true_value(spec$dgp,spec$gamma)$V_value, mcse=0, n=NA_integer_, horizon=Inf, method="Exact Bellman solve")
        } else {
          seed <- 53000000L + (scenario=="weak_proxy")*10000L
          set.seed(seed)
          # 1,000,000 reference trajectories do not increase the 20 evaluation replications.
          returns <- unlist(lapply(1:10,function(i) target_returns(e,spec$dgp,spec$gamma,100000L,100L)),use.names=FALSE)
          r <- list(value=mean(returns),mcse=sd(returns)/sqrt(length(returns)),n=length(returns),
            horizon=100L, seed=seed, returns=returns, method="Independent target-policy Monte Carlo")
        }
        r$dgp <- spec$dgp; r$gamma <- spec$gamma
        atomic_save(r,path); references[[key]] <- r
        cat(key,"truth",r$value,"MCSE",r$mcse,"\n");flush.console()
      }
    }
  }
}

if (args[1] == "run") {
  stopifnot(file.exists(file.path(out,"validation.txt")))
  all_sources <- c(sources,"convergence_diagnostics.R","run_ci_study.R")
  for (f in all_sources) {
    dest <- file.path(out,"source_snapshot",f)
    dir.create(dirname(dest),recursive=TRUE,showWarnings=FALSE)
    if (!file.exists(dest)) stopifnot(file.copy(f,dest))
    else stopifnot(unname(tools::md5sum(f)) == unname(tools::md5sum(dest)))
  }
  atomic_save(list(n_rep=20L,N=50L,T=50L,gamma=.7,settings=settings,scenarios=scenarios,
    source_md5=tools::md5sum(all_sources),input=input,estimator="Current production MR; original 95% IF CI; no convergence filtering",
    created=Sys.time(),session=sessionInfo()),file.path(out,"config.rds"))
  jobs <- expand.grid(rep=1:20,scenario=scenarios,setting=settings,stringsAsFactors=FALSE)
  answers <- parallel::mclapply(seq_len(nrow(jobs)),function(i) {
    j <- jobs[i,]
    path <- file.path(out,"replications",j$setting,j$scenario,sprintf("rep_%03d.rds",j$rep))
    if (file.exists(path)) return(TRUE)
    x <- fit_case(j$setting,j$scenario,j$rep)
    atomic_save(x,path)
    cat(sprintf("%s %s rep=%02d: %s, %.2fs\n",j$setting,j$scenario,j$rep,
      if(is.null(x$error)) sprintf("V=%.5f, SE=%.5f",x$result$V_hat,x$result$se) else x$error,x$elapsed))
    flush.console()
    TRUE
  },mc.cores=cores,mc.preschedule=FALSE,mc.set.seed=FALSE)
  stopifnot(all(vapply(answers,isTRUE,logical(1))))
}

if (args[1] == "report") {
  source("report_ci_study.R")
}
