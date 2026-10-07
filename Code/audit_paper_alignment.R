#!/usr/bin/env Rscript
# Read-only audit of production estimators against 2607.25241v1.pdf.
# Alternative formulas below are applied only to isolated function copies.
root <- dirname(normalizePath(sub("^--file=","",grep("^--file=",commandArgs(),value=TRUE)[1])))
setwd(root)
source("convergence_diagnostics.R")
out <- file.path(root,"simulation_results/paper_alignment_audit_20260914")
dir.create(out,recursive=TRUE,showWarnings=FALSE)
input <- file.path(root,"simulation_results/convergence_20_20260913/main")
sources <- c("Tabular/Methods.R","Continuous/Methods_continuous.R",
  "Tabular/simulation_tabular.R","Continuous/simulation_continuous.R","../2607.25241v1.pdf")
hashes <- tools::md5sum(sources)
load_case <- function(setting,scenario,rep) readRDS(file.path(input,setting,scenario,sprintf("rep_%03d.rds",rep)))
fix_prior_copy <- function(f) {
  hits <- 0L
  body(f) <- diag_walk(body(f),function(x) {
    if (identical(x,quote(ifelse(a == 1, log(b_hat), log(1 - b_hat))))) {
      hits <<- hits+1L
      quote(if (a == 1) log(b_hat) else log(1-b_hat))
    } else x
  })
  stopifnot(hits==1L)
  f
}

# Equation (8) must preserve a varying prior when the component likelihoods
# for the observed outcomes are equal. Row order must not change that result.
e <- diag_load(root,"Continuous")
stub <- list(predict_b=function(s1,s2) s1,
  predict_mu=function(s1,s2,a) rep(.5,length(s1)),
  predict_theta_R=function(s1,s2,a) rep(a,length(s1)),
  predict_theta_Sp1=function(s1,s2,a) rep(a,length(s1)),
  predict_theta_Sp2=function(s1,s2,a) rep(a,length(s1)),
  sigma_R=c(1,1),sigma_Sp1=c(1,1),sigma_Sp2=c(1,1))
original <- e$compute_eta_outfold(stub,c(.2,.8),c(0,0),c(0,0),c(.5,.5),c(.5,.5),c(.5,.5))[,2]
fixed_fn <- fix_prior_copy(e$compute_eta_outfold)
fixed <- fixed_fn(stub,c(.2,.8),c(0,0),c(0,0),c(.5,.5),c(.5,.5),c(.5,.5))[,2]
permuted <- e$compute_eta_outfold(stub,c(.8,.2),c(0,0),c(0,0),c(.5,.5),c(.5,.5),c(.5,.5))[2:1,2]
stopifnot(max(abs(original-c(.2,.2)))<1e-12,max(abs(fixed-c(.2,.8)))<1e-12,
  max(abs(permuted-c(.8,.8)))<1e-12)
posterior_check <- data.frame(observation=1:2,prior=c(.2,.8),original_posterior=original,
  expected_posterior=c(.2,.8),corrected_copy=fixed,original_after_row_permutation=permuted)
write.csv(posterior_check,file.path(out,"posterior_bayes_counterexample.csv"),row.names=FALSE)

# These distinct state weightings can reverse the label selector despite
# strictly positive conditional surrogate contrast at every state.
ps <- c(.5,.5); b <- c(.99,.01); mu <- cbind(c(.1,.8),c(.2,.9))
paper_scores <- colSums(mu*ps)
code_scores <- c(sum(ps*(1-b)*mu[,1])/sum(ps*(1-b)),sum(ps*b*mu[,2])/sum(ps*b))
stopifnot(all(mu[,2]>mu[,1]),which.max(paper_scores)==2L,which.max(code_scores)==1L)
label_check <- data.frame(component=0:1,paper_common_state_mean=paper_scores,code_class_conditional_mean=code_scores)
write.csv(label_check,file.path(out,"label_rule_counterexample.csv"),row.names=FALSE)

# Independent vectorized implementation of Equation (3) for Tabular.
x <- load_case("Tabular","typical",1L); et <- diag_load(root,"Tabular")
em <- diag_tab_build(x$runs[[2]]$fit$raw,x$dat)
et$em_tabular <- function(...) em
production <- et$mr_estimator(x$dat,x$spec$dgp,.7,K=2L)
production_k5 <- et$mr_estimator(x$dat,x$spec$dgp,.7,K=5L)
stopifnot(identical(production,production_k5))
om <- et$solve_omega_tabular(em,x$dat,x$spec$dgp,.7)
q <- et$solve_Q_tabular(em,x$spec$dgp,.7)
v <- rowSums(q*cbind(1-x$spec$dgp$pi_policy,x$spec$dgp$pi_policy))
mv <- cbind(em$P_hat[,,1]%*%v,em$P_hat[,,2]%*%v)
s <- as.vector(x$dat$S); sp <- as.vector(x$dat$Sprime)
rr <- as.vector(x$dat$R); at <- as.vector(x$dat$Atilde)
correction <- numeric(length(s))
for(a in 1:2) {
  other<-3L-a
  br_a <- (at-em$mu_hat[s,other])/(em$mu_hat[s,a]-em$mu_hat[s,other])
  br_r <- (rr-em$theta_R_hat[s,other])/(em$theta_R_hat[s,a]-em$theta_R_hat[s,other])
  br_sp <- (sp-em$theta_Sp_hat[s,other])/(em$theta_Sp_hat[s,a]-em$theta_Sp_hat[s,other])
  correction <- correction + om[s,a]*(br_a*br_sp*(rr-em$theta_R_hat[s,a])+.7*br_a*br_r*(v[sp]-mv[s,a]))/.3
}
direct <- sum(x$spec$dgp$p_e*v)
value <- direct+mean(correction)
n <- length(s)
se_eq2 <- sqrt(mean((direct+correction-value)^2)/n)
tab_check <- list(value_production=production$V_hat,value_equation3=value,
  se_production=production$se,se_centered_equation2=se_eq2,
  se_ratio=production$se/se_eq2,expected_bessel_factor=sqrt(n/(n-1)),
  bellman_residual=max(abs(q-em$theta_R_hat-.7*mv)),K_argument_ignored=identical(production,production_k5))
stopifnot(abs(value-production$V_hat)<1e-12,tab_check$bellman_residual<1e-12,
  abs(tab_check$se_ratio-tab_check$expected_bessel_factor)<1e-12)
saveRDS(tab_check,file.path(out,"tabular_equation_checks.rds"))

# Expose raw fold correction vectors without changing any fitted value or RNG.
expose_if_copy <- function(f) {
  expressions <- as.list(body(f)); last <- expressions[[length(expressions)]]
  expressions[[length(expressions)]] <- substitute({
    answer <- ORIGINAL
    answer$audit <- list(direct_list=direct_list,phi_list=phi_list,fold_ids=fold_ids)
    answer
  },list(ORIGINAL=last))
  body(f) <- as.call(expressions)
  f
}

audit_case <- function(scenario,rep) {
  x <- load_case("Continuous",scenario,rep)
  seed <- 43100000L+rep*100L
  outputs <- list(); rows <- list()
  for (variant in c("current","correct_prior_only","equation3_unclipped_only")) {
    e <- diag_load(root,"Continuous")
    if(variant=="correct_prior_only") {
      e$.em_single_run <- fix_prior_copy(e$.em_single_run)
      e$compute_eta_outfold <- fix_prior_copy(e$compute_eta_outfold)
    }
    f <- expose_if_copy(e$mr_estimator_continuous)
    if(variant=="equation3_unclipped_only") {
      e$safe_ratio <- function(num,denom,...) num/denom
      e$clip_abs_quantile <- function(x,...) x
      body(f)<-diag_walk(body(f),function(z) {
        if (identical(z,quote(quantile(c(abs(om0), abs(om1)), 0.97)))) quote(Inf) else z
      })
    }
    set.seed(seed)
    z <- f(x$dat,x$spec$dgp,x$spec$gamma)
    if(variant=="current") {
      old<-readRDS(file.path(root,"simulation_results/ci_20_20260913/replications/Continuous",scenario,sprintf("rep_%03d.rds",rep)))
      stopifnot(identical(z[names(old$result)],old$result))
    }
    full_if <- unlist(lapply(1:2,function(k) z$audit$direct_list[k]+z$audit$phi_list[[k]]-z$V_hat))
    rows[[variant]] <- data.frame(scenario=scenario,rep=rep,variant=variant,
      estimate=z$V_hat,se=z$se,full_fold_centered_se=sqrt(mean(full_if^2)/length(full_if)),
      bridge_index=z$bridge_index,direct_fold_difference=diff(z$audit$direct_list))
    outputs[[variant]] <- z
  }
  path <- file.path(out,"cases",scenario,sprintf("rep_%03d.rds",rep))
  dir.create(dirname(path),recursive=TRUE,showWarnings=FALSE)
  saveRDS(outputs,path)
  cat(scenario,"rep",rep,"complete\n");flush.console()
  do.call(rbind,rows)
}
# First five saved datasets in every prespecified condition: fixed diagnostic subset.
jobs <- expand.grid(scenario=c("typical","hard","weak_anchor","weak_proxy","state_dependent"),rep=1:5,stringsAsFactors=FALSE)
records <- parallel::mclapply(seq_len(nrow(jobs)),function(i) audit_case(jobs$scenario[i],jobs$rep[i]),
  mc.cores=2L,mc.set.seed=FALSE)
stopifnot(all(vapply(records,is.data.frame,logical(1))))
records <- do.call(rbind,records)
write.csv(records,file.path(out,"isolated_formula_comparisons.csv"),row.names=FALSE)
summary_rows <- list()
for(scenario in unique(jobs$scenario)) for(variant in c("correct_prior_only","equation3_unclipped_only")) {
  base <- records[records$scenario==scenario & records$variant=="current",]
  changed <- records[records$scenario==scenario & records$variant==variant,]
  changed <- changed[match(base$rep,changed$rep),]
  delta <- changed$estimate-base$estimate
  summary_rows[[length(summary_rows)+1L]] <- data.frame(scenario=scenario,variant=variant,n=length(delta),
    median_abs_value_change=median(abs(delta)),max_abs_value_change=max(abs(delta)),
    median_se_ratio=median(changed$se/base$se),max_abs_change_in_current_se=max(abs(delta)/base$se))
}
summary <- do.call(rbind,summary_rows)
write.csv(summary,file.path(out,"isolated_formula_summary.csv"),row.names=FALSE)
stopifnot(identical(hashes,tools::md5sum(sources)))
writeLines(c("Production code and source PDF unchanged.",capture.output(hashes),
  "Tabular Equation (3), Bellman equation, CI centering/Bessel factor, and ignored K verified.",
  "Continuous varying-prior and row-order counterexamples verified; corrected function copies satisfy Bayes formula.",
  "Current continuous outputs reproduce saved CI results exactly on all 25 diagnostic datasets.",
  capture.output(tab_check),capture.output(summary),capture.output(sessionInfo())),file.path(out,"validation.txt"))
saveRDS(list(source_md5=hashes,tabular=tab_check,posterior=posterior_check,labels=label_check,summary=summary),file.path(out,"audit.rds"))
print(posterior_check);print(label_check);print(tab_check);print(summary)
