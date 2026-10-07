# Instrumentation for the convergence study. Original estimator files are sourced
# into isolated environments; this file does not modify them.

diag_walk <- function(x, transform) {
  if (is.call(x)) for (i in seq_along(x)[-1L]) x[i] <- list(diag_walk(x[[i]], transform))
  transform(x)
}

diag_load <- function(root, setting) {
  e <- new.env(parent = globalenv())
  file <- if (setting == "Tabular") "Tabular/Methods.R" else "Continuous/Methods_continuous.R"
  source(file.path(root, file), local = e)
  e$.original_single <- if (setting == "Continuous") e$.em_single_run else NULL
  e$.original_wrapper <- if (setting == "Continuous") e$em_continuous else e$em_tabular
  e$.original_value <- if (setting == "Continuous") e$mr_estimator_continuous else e$mr_estimator
  if (setting == "Continuous") {
    f <- e$.original_single
    formals(f) <- c(formals(f), alist(diag_hook = NULL))
    body(f) <- diag_walk(body(f), function(x) {
      if (is.call(x) && identical(x[[1]], as.name("if")) &&
          identical(x[[2]], quote(max(abs(eta - eta_old)) < tol))) {
        substitute({ if (!is.null(diag_hook)) diag_hook(environment()); OLD }, list(OLD = x))
      } else x
    })
    e$.diagnostic_single <- f
    builder <- e$.original_wrapper
    formals(builder) <- c(formals(builder), alist(.diag_raw = NULL))
    b <- as.list(body(builder))
    hits <- vapply(b, function(x) is.call(x) && identical(x[[1]], as.name("for")) &&
                     identical(x[[2]], as.name("rst")), logical(1))
    stopifnot(sum(hits) == 1L)
    b[[which(hits)]] <- quote(best <- .diag_raw)
    body(builder) <- as.call(b)
    e$.builder <- builder
  }
  e
}

diag_spec <- function(e, setting, scenario) {
  tau <- switch(scenario, typical = .1, hard = .3, weak_anchor = .45,
                weak_proxy = .3, state_dependent = .3, diagnostic_seed12 = .3)
  mode <- if (scenario %in% c("state_dependent", "diagnostic_seed12")) "state_dependent" else "constant"
  if (setting == "Tabular") {
    dgp <- e$generate_dgp()
    if (scenario == "weak_proxy") {
      mid <- (dgp$P_trans[, , 1] + dgp$P_trans[, , 2]) / 2
      for (a in 1:2) dgp$P_trans[, , a] <- mid + .25 * (dgp$P_trans[, , a] - mid)
    }
  } else {
    policy <- function(s1, s2) as.numeric(s1 >= .25 & s2 >= -.10)
    environment(policy) <- baseenv()
    dgp <- e$generate_dgp_continuous(s1_a_int = .4, s2_a_int = -.3,
      init_mean = c(.25, .05), init_sd = c(.75, .75), pi_func = policy)
    dgp$base1 <- .5; dgp$base2 <- 1/3
    if (scenario == "weak_proxy") {
      dgp$base1 <- .5 + .375 * .4; dgp$base2 <- 1/3 + .375 * (-.3)
      dgp$tr_shift <- .25 * dgp$tr_shift
      dgp$s1_a_int <- .25 * dgp$s1_a_int; dgp$s2_a_int <- .25 * dgp$s2_a_int
    }
  }
  list(setting = setting, scenario = scenario, tau = tau, mode = mode, dgp = dgp,
       N = 50L, TT = 50L, gamma = .7, strength = .75)
}

diag_data <- function(e, spec, rep, evaluation = FALSE, pilot = FALSE) {
  env_id <- if (spec$setting == "Tabular") 1L else 2L
  seed <- 13000000L + env_id * 100000L + rep * 1000L +
    if (evaluation) 4000000L else 0L
  if (pilot) seed <- seed + 6000000L
  if (spec$scenario == "diagnostic_seed12" && !evaluation) seed <- 12L
  set.seed(seed)
  N <- if (evaluation) 20L else spec$N
  if (spec$setting == "Tabular") {
    dat <- e$generate_data(spec$dgp, N, spec$TT, spec$tau, spec$mode, spec$strength)
  } else {
    f <- e$generate_data_continuous
    if (spec$scenario == "weak_proxy") {
      body(f) <- diag_walk(body(f), function(x) {
        # Restrict the replacement to transition terms: (1/2) also occurs
        # in the reward, whose state coefficient must remain unchanged.
        if (identical(x, quote((1/2) * s1))) quote(dgp$base1 * s1)
        else if (identical(x, quote((1/3) * s2))) quote(dgp$base2 * s2) else x
      })
    }
    dat <- f(spec$dgp, N, spec$TT, spec$tau, spec$mode, spec$strength)
  }
  attr(dat, "diagnostic_seed") <- seed
  dat
}

diag_start_names <- c("surrogate_55", "default_70", "surrogate_90", "perturbed_1",
                      "perturbed_2", "random_1", "random_2", "reversed_70")
diag_init <- function(dat, start, rep, setting, fold = 0L) {
  at <- as.vector(dat$Atilde); n <- length(at)
  old_rng <- if (exists(".Random.seed", .GlobalEnv)) get(".Random.seed", .GlobalEnv) else NULL
  on.exit(if (!is.null(old_rng)) assign(".Random.seed", old_rng, .GlobalEnv), add = TRUE)
  set.seed(23000000L + (setting == "Continuous") * 100000L + rep * 100L + start + fold * 1000000L)
  if (start <= 3L) p <- rep(c(.55, .70, .90)[start], n)
  else if (start <= 5L) p <- runif(n, .55, .85)
  else if (start <= 7L) return(cbind(1 - (p <- runif(n, .10, .90)), p))
  else p <- rep(.30, n)
  p1 <- ifelse(at == 1L, p, 1-p)
  cbind(1-p1, p1)
}

diag_tab_single <- function(dat, init_eta, max_iter = 100L, tol = 1e-4,
                            clip_bound = .01, diag_hook = NULL) {
  S <- as.vector(dat$S); At <- as.vector(dat$Atilde); R <- as.vector(dat$R)
  Sp <- as.vector(dat$Sprime); n <- length(S); nS <- 3L; eta <- init_eta
  cell <- lapply(seq_len(nS), function(s) which(S == s))
  for (iter in seq_len(max_iter)) {
    eta_old <- eta
    b_hat <- numeric(nS); mu_hat <- matrix(.5, nS, 2)
    theta_R_hat <- matrix(0, nS, 2); sigma_R_hat <- matrix(1, nS, 2)
    P_hat <- array(1/nS, c(nS, nS, 2))
    for (s in seq_len(nS)) {
      idx <- cell[[s]]; if (!length(idx)) next
      b_hat[s] <- mean(eta[idx, 2])
      for (a in 1:2) {
        w <- eta[idx, a]; sw <- sum(w); if (sw < 1e-10) next
        mu_hat[s, a] <- sum(w * At[idx]) / sw
        theta_R_hat[s, a] <- sum(w * R[idx]) / sw
        sigma_R_hat[s, a] <- sqrt(max(sum(w * (R[idx] - theta_R_hat[s,a])^2)/sw, .01))
        for (sp in seq_len(nS)) P_hat[s, sp, a] <- sum(w * (Sp[idx] == sp))
        tot <- sum(P_hat[s, , a]); if (tot > 1e-10) P_hat[s, , a] <- P_hat[s, , a]/tot
      }
    }
    b_hat <- pmax(pmin(b_hat, 1-clip_bound), clip_bound)
    mu_hat <- pmax(pmin(mu_hat, 1-clip_bound), clip_bound)
    log_eta <- matrix(NA_real_, n, 2)
    for (a in 1:2) log_eta[, a] <- log(if (a == 2) b_hat[S] else 1-b_hat[S]) +
      dbinom(At, 1, mu_hat[cbind(S, a)], log=TRUE) +
      dnorm(R, theta_R_hat[cbind(S,a)], sigma_R_hat[cbind(S,a)], log=TRUE) +
      log(pmax(P_hat[cbind(S,Sp,a)],1e-10))
    mx <- pmax(log_eta[,1],log_eta[,2]); ex <- exp(log_eta-mx)
    loglik <- sum(log(rowSums(ex))+mx); eta <- ex/rowSums(ex)
    if (!is.null(diag_hook)) diag_hook(environment())
    if (max(abs(eta-eta_old)) < tol) break
  }
  list(eta=eta,loglik=loglik,n_iter=iter,b_hat=b_hat,mu_hat=mu_hat,
       theta_R_hat=theta_R_hat,sigma_R_hat=sigma_R_hat,P_hat=P_hat)
}

diag_tab_build <- function(raw, dat) {
  at <- as.vector(dat$Atilde); s <- as.vector(dat$S); sp <- as.vector(dat$Sprime)
  em <- raw
  em$label_swapped <- logical(3)
  for (ss in 1:3) if (em$mu_hat[ss,1] > em$mu_hat[ss,2]) {
    ii <- which(s==ss)
    em$eta[ii,] <- em$eta[ii,2:1,drop=FALSE]
    em$b_hat[ss] <- 1-em$b_hat[ss]
    for (nm in c("mu_hat","theta_R_hat","sigma_R_hat"))
      em[[nm]][ss,] <- em[[nm]][ss,2:1]
    em$P_hat[ss,,] <- em$P_hat[ss,,2:1,drop=FALSE]
    em$label_swapped[ss] <- TRUE
  }
  em$theta_Sp_hat <- apply(em$P_hat,c(1,3),function(p)sum(1:3*p))
  em$theta_At_hat <- em$mu_hat
  em
}

diag_clip_function <- function(f, bound) {
  body(f) <- diag_walk(body(f), function(x) {
    if (is.call(x) && identical(x[[1]],as.name("clip")) && length(x)==4L &&
        identical(x[[3]], .02) && identical(x[[4]], .98)) {
      x[[3]] <- bound; x[[4]] <- 1-bound
    }
    x
  }); f
}

diag_build <- function(e, setting, raw, dat, clip_factor=1) {
  if (setting=="Tabular") diag_tab_build(raw,dat)
  else diag_clip_function(e$.builder,.02*clip_factor)(dat,.7,.diag_raw=raw)
}

diag_capture_raw <- function(fr, setting) {
  if (setting=="Tabular") {
    nm <- c("eta","loglik","b_hat","mu_hat","theta_R_hat","sigma_R_hat","P_hat")
  } else nm <- c("eta","loglik","sigma_R","sigma_Sp1","sigma_Sp2",
    "fit_R0","fit_R1","fit_Sp1_0","fit_Sp1_1","fit_Sp2_0","fit_Sp2_1","fit_mu0","fit_mu1","fit_b")
  out <- mget(nm,envir=fr,inherits=FALSE); out$n_iter <- fr$iter
  out
}

diag_fit <- function(e, setting, dat, init, cap=NULL, tol=NULL, clip_factor=1,
                      checkpoints=FALSE) {
  if (is.null(cap)) cap <- 100L
  if (is.null(tol)) tol <- if(setting=="Tabular") 1e-4 else 1e-3
  trace <- vector("list",cap); snaps <- list(); first_hits <- list(); last_ll <- NA_real_
  streak <- setNames(integer(3),c("0.001","1e-04","1e-05"))
  hooks <- function(fr) {
    k <- fr$iter; delta <- fr$eta-fr$eta_old; avgll <- fr$loglik/nrow(fr$eta)
    dll <- avgll-last_ll
    trace[[k]] <<- data.frame(iter=k,avg_loglik=avgll,delta_ll=dll,
      delta_max=max(abs(delta)),delta_rms=sqrt(mean(delta^2)))
    last_ll <<- avgll
    if (checkpoints) {
      keys <- character()
      for (tt in c(1e-3,1e-4,1e-5)) {
        key <- format(tt,scientific=TRUE)
        a <- paste0("posterior_",key)
        if (max(abs(delta))<tt && is.null(first_hits[[a]])) {
          first_hits[[a]] <<- k; keys <- c(keys,a)
        }
        sk <- as.character(tt)
        streak[sk] <<- if (max(abs(delta))<tt && is.finite(dll) && abs(dll)<1e-6) streak[sk]+1L else 0L
        b <- paste0("joint_",key)
        if(streak[sk]>=3 && is.null(first_hits[[b]])) {
          first_hits[[b]] <<- k; keys <- c(keys,b)
        }
      }
      if(k %in% c(100,300,500)) keys <- c(keys,paste0("cap_",k))
      if(length(keys)) snaps[[as.character(k)]] <<- list(keys=keys,raw=diag_capture_raw(fr,setting))
    }
  }
  messages <- character(); error <- NULL; time0 <- proc.time()[["elapsed"]]
  raw <- tryCatch(withCallingHandlers({
    if(setting=="Tabular") diag_tab_single(dat,init,cap,tol,.01*clip_factor,hooks)
    else {
      f <- diag_clip_function(e$.diagnostic_single,.02*clip_factor)
      f(as.vector(dat$S1),as.vector(dat$S2),as.vector(dat$Atilde),as.vector(dat$R),
        as.vector(dat$Sp1),as.vector(dat$Sp2),init,max_iter=cap,tol=tol,diag_hook=hooks)
    }
  },warning=function(w) {messages <<- c(messages,conditionMessage(w));invokeRestart("muffleWarning")}),
  error=function(err) {error <<- conditionMessage(err);NULL})
  tr <- do.call(rbind,trace)
  list(raw=raw,trace=tr,snapshots=snaps,first_hits=first_hits,warnings=unique(messages),
       warning_count=length(messages),error=error,elapsed=proc.time()[["elapsed"]]-time0,
       status=if(!is.null(error)) "error" else if(tail(tr$delta_max,1)<tol) "converged" else "iteration_cap",
       cap=cap,tol=tol,clip_factor=clip_factor)
}

diag_truth <- function(spec,dat) {
  s <- if(spec$setting=="Tabular") as.vector(dat$S) else cbind(as.vector(dat$S1),as.vector(dat$S2))
  at<-as.vector(dat$Atilde); r<-as.vector(dat$R); pp<-as.vector(dat$misclassification_prob)
  mu<-cbind(pp,1-pp); d<-spec$dgp; n<-length(r); logs<-matrix(0,n,2)
  if(spec$setting=="Tabular") {
    b<-d$b_policy[s]; rr<-d$theta_R[s,,drop=FALSE]
    trans<-lapply(1:2,function(a) d$P_trans[s,,a]); spmean<-sapply(trans,function(p) as.vector(p %*% 1:3))
    for(a in 1:2) logs[,a]<-log(if(a==2)b else 1-b)+dbinom(at,1,mu[,a],log=TRUE)+
      dnorm(r,rr[,a],d$sigma_R,log=TRUE)+log(d$P_trans[cbind(s,as.vector(dat$Sprime),a)])
  } else {
    b<-rep(d$b_prob,n); rr<-cbind(1+s[,1]+.5*s[,2],2.5+s[,1]+.5*s[,2])
    spmean<-lapply(0:1,function(a) cbind(d$base1*s[,1]+d$tr_shift*(2*a-1)+d$s1_a_int*s[,1]*a,
      d$base2*s[,2]-d$tr_shift*(2*a-1)+d$s2_a_int*s[,2]*a))
    for(a in 1:2) logs[,a]<-log(if(a==2)b else 1-b)+dbinom(at,1,mu[,a],log=TRUE)+
      dnorm(r,rr[,a],d$sigma_R,log=TRUE)+
      dnorm(as.vector(dat$Sp1),spmean[[a]][,1],d$sigma_tr,log=TRUE)+
      dnorm(as.vector(dat$Sp2),spmean[[a]][,2],d$sigma_tr,log=TRUE)
    trans<-NULL
  }
  mx<-pmax(logs[,1],logs[,2]); z<-exp(logs-mx); eta<-z/rowSums(z)
  list(b=b,mu=mu,reward=rr,spmean=spmean,trans=trans,eta=eta)
}

diag_predict <- function(e,spec,em,dat) {
  if(spec$setting=="Tabular") {
    s<-as.vector(dat$S); at<-as.vector(dat$Atilde); r<-as.vector(dat$R); sp<-as.vector(dat$Sprime)
    b<-em$b_hat[s]; mu<-em$mu_hat[s,,drop=FALSE]; rr<-em$theta_R_hat[s,,drop=FALSE]
    spmean<-em$theta_Sp_hat[s,,drop=FALSE]
    trans<-lapply(1:2,function(a) em$P_hat[s,,a]); logeta<-matrix(0,length(s),2)
    for(a in 1:2) logeta[,a]<-log(if(a==2)b else 1-b)+dbinom(at,1,mu[,a],log=TRUE)+
      dnorm(r,rr[,a],em$sigma_R_hat[cbind(s,a)],log=TRUE)+log(pmax(em$P_hat[cbind(s,sp,a)],1e-10))
    mx<-pmax(logeta[,1],logeta[,2]); eta<-exp(logeta-mx);eta<-eta/rowSums(eta)
  } else {
    s1<-as.vector(dat$S1);s2<-as.vector(dat$S2)
    b<-em$predict_b(s1,s2);mu<-sapply(0:1,function(a) em$predict_mu(s1,s2,a))
    rr<-sapply(0:1,function(a) em$predict_theta_R(s1,s2,a))
    spmean<-lapply(0:1,function(a) cbind(em$predict_theta_Sp1(s1,s2,a),em$predict_theta_Sp2(s1,s2,a)))
    trans<-NULL
    eta<-e$compute_eta_outfold(em,s1,s2,as.vector(dat$Atilde),as.vector(dat$R),as.vector(dat$Sp1),as.vector(dat$Sp2))
  }
  list(b=b,mu=mu,reward=rr,spmean=spmean,trans=trans,eta=eta)
}

diag_metrics <- function(e,spec,fit,train,test,truth=NULL) {
  if(is.null(fit$raw)) return(list(metrics=data.frame(status=fit$status,error=fit$error),pred=NULL))
  if(is.null(truth)) truth<-diag_truth(spec,test)
  em<-diag_build(e,spec$setting,fit$raw,train,fit$clip_factor)
  pred<-diag_predict(e,spec,em,test); p<-pred$eta[,2]; p0<-truth$eta[,2]
  loss<-mean((p-p0)^2); reversed_loss<-mean((1-p-p0)^2)
  mu_gap<-pred$mu[,2]-pred$mu[,1]
  if(spec$setting=="Tabular") {
    spgap<-pred$spmean[,2]-pred$spmean[,1]
    sp_rmse<-sqrt(mean((pred$spmean-truth$spmean)^2))
    p_rmse<-sqrt(mean((unlist(pred$trans)-unlist(truth$trans))^2))
    local_loss<-vapply(1:3,function(s) {
      ii<-as.vector(test$S)==s
      mean((1-p[ii]-p0[ii])^2)-mean((p[ii]-p0[ii])^2)
    },numeric(1))
    local_wrong<-mean(local_loss < -1e-8)
  } else {
    bi<-e$select_bridge_index_continuous(train)$bridge_index
    spgap<-pred$spmean[[2]][,bi]-pred$spmean[[1]][,bi]
    sp_rmse<-sqrt(mean((unlist(pred$spmean)-unlist(truth$spmean))^2))
    p_rmse<-NA_real_;local_wrong<-NA_real_
  }
  rawm<-colSums(fit$raw$eta * as.vector(train$Atilde))/colSums(fit$raw$eta)
  code_swap<-rawm[1]>rawm[2]
  # Common-state manuscript orientation, evaluated on TRAINING states only.
  ptrain<-diag_predict(e,spec,em,train)
  manuscript_disagree<-mean(ptrain$mu[,2]-ptrain$mu[,1]) < 0
  tr<-fit$trace
  metrics<-data.frame(status=fit$status,n_iter=fit$raw$n_iter,elapsed=fit$elapsed,
    avg_loglik=fit$raw$loglik/length(train$Atilde),last_delta=tail(tr$delta_max,1),
    objective_decreases=sum(tr$delta_ll < -1e-8,na.rm=TRUE),
    warning_count=fit$warning_count,raw_swap=code_swap,
    orientation_wrong=reversed_loss+1e-8<loss,orientation_ambiguous=abs(reversed_loss-loss)<=1e-8,
    orientation_loss_gap=reversed_loss-loss,local_wrong_fraction=local_wrong,
    manuscript_rule_disagrees=manuscript_disagree,
    eta_rmse=sqrt(loss),eta_rmse_best_global=sqrt(min(loss,reversed_loss)),
    b_rmse=sqrt(mean((pred$b-truth$b)^2)),mu_rmse=sqrt(mean((pred$mu-truth$mu)^2)),
    reward_rmse=sqrt(mean((pred$reward-truth$reward)^2)),transition_mean_rmse=sp_rmse,
    transition_probability_rmse=p_rmse,
    mu_gap_min=min(mu_gap),mu_gap_q05=unname(quantile(mu_gap,.05)),
    mu_gap_negative=mean(mu_gap<=0),mu_gap_near_zero=mean(abs(mu_gap)<.05),
    proxy_gap_abs_q05=unname(quantile(abs(spgap),.05)),
    observed_error=mean(train$Atilde!=train$A),expected_error=mean(train$misclassification_prob))
  list(metrics=metrics,pred=pred)
}

diag_compact_raw <- function(raw) {
  if(is.null(raw)) return(raw)
  for(nm in names(raw)) if(inherits(raw[[nm]],"lm")) {
    raw[[nm]]$model<-NULL;raw[[nm]]$y<-NULL;raw[[nm]]$data<-NULL
    if(!is.null(raw[[nm]]$terms)) attr(raw[[nm]]$terms,".Environment")<-baseenv()
    if(!is.null(raw[[nm]]$formula)) environment(raw[[nm]]$formula)<-baseenv()
    # Prediction uses coefficients, terms, contrasts and QR. Removing these
    # training-only arrays avoids retaining duplicate data and call frames.
    for(k in c("residuals","fitted.values","effects","weights","prior.weights","linear.predictors"))
      raw[[nm]][[k]]<-NULL
  }
  raw
}

diag_bind_rows <- function(xs) {
  xs<-Filter(function(x)!is.null(x)&&nrow(x)>0,xs)
  if(!length(xs)) return(data.frame())
  nm<-unique(unlist(lapply(xs,names)))
  xs<-lapply(xs,function(x){for(k in setdiff(nm,names(x))) x[[k]]<-NA; x[,nm,drop=FALSE]})
  do.call(rbind,xs)
}
