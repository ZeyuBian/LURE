# Reproduce and isolate tabular issues using the existing 20-dataset study.
# Original estimator files are read only; candidate changes are applied to copies.
source("convergence_diagnostics.R")
suppressPackageStartupMessages(library(dplyr))
root<-normalizePath(".")
study<-file.path(root,"simulation_results/convergence_20_20260913")
out<-file.path(root,"simulation_results/tabular_code_audit_20260913")
dir.create(out,recursive=TRUE,showWarnings=FALSE)
e<-diag_load(root,"Tabular")
methods_hash<-tools::md5sum("Tabular/Methods.R")
writeLines(as.character(methods_hash),file.path(out,"methods_md5.txt"))

raw_loglik<-function(raw,dat) {
  s<-as.vector(dat$S);at<-as.vector(dat$Atilde);r<-as.vector(dat$R);sp<-as.vector(dat$Sprime)
  logp<-sapply(1:2,function(a)log(if(a==2)raw$b_hat[s] else 1-raw$b_hat[s])+
    dbinom(at,1,raw$mu_hat[cbind(s,a)],log=TRUE)+
    dnorm(r,raw$theta_R_hat[cbind(s,a)],raw$sigma_R_hat[cbind(s,a)],log=TRUE)+
    log(pmax(raw$P_hat[cbind(s,sp,a)],1e-10)))
  mx<-pmax(logp[,1],logp[,2]);sum(mx+log(rowSums(exp(logp-mx))))
}
permute_states<-function(raw,dat,states) {
  for(s in states) {
    ii<-as.vector(dat$S)==s;raw$eta[ii,]<-raw$eta[ii,2:1,drop=FALSE]
    raw$b_hat[s]<-1-raw$b_hat[s]
    for(nm in c("mu_hat","theta_R_hat","sigma_R_hat"))raw[[nm]][s,]<-raw[[nm]][s,2:1]
    raw$P_hat[s,,]<-raw$P_hat[s,,2:1]
  };raw
}
coherent_mean<-function(em) {
  em$theta_Sp_hat<-sapply(1:2,function(a)as.vector(em$P_hat[,,a]%*%1:3));em
}
pointwise_build<-diag_tab_build
body(pointwise_build)<-diag_walk(body(pointwise_build),function(x) {
  if(is.call(x)&&identical(x[[1]],as.name("if"))&&identical(x[[2]],quote(m[1]>m[2])))x[[2]]<-FALSE
  x
})
value_from_em<-function(em,dat,dgp) {
  env<-new.env(parent=e);f<-e$.original_value;environment(f)<-env
  env$em_tabular<-function(...)em
  f(dat,dgp,.7)
}

# Constructive demonstration: a single-state label swap preserves the entire
# fitted observed likelihood, but the production global rule cannot repair it.
x<-readRDS(file.path(study,"main/Tabular/typical/rep_001.rds"))
raw<-x$runs[[2]]$fit$raw;swapped<-permute_states(raw,x$dat,3L)
original_em<-diag_tab_build(raw,x$dat);global_em<-diag_tab_build(swapped,x$dat)
statewise_raw<-permute_states(swapped,x$dat,which(swapped$mu_hat[,2]<swapped$mu_hat[,1]))
statewise_em<-pointwise_build(statewise_raw,x$dat)
stopifnot(abs(raw_loglik(raw,x$dat)-raw_loglik(swapped,x$dat))<1e-8,
          max(abs(original_em$P_hat-statewise_em$P_hat))<1e-10,
          max(abs(original_em$eta-statewise_em$eta))<1e-10)
permutation_demo<-data.frame(
  variant=c("Original","Swap only state 3, then production global alignment","Swap state 3, then statewise alignment"),
  loglik=c(raw_loglik(raw,x$dat),raw_loglik(swapped,x$dat),raw_loglik(statewise_raw,x$dat)),
  value=c(value_from_em(original_em,x$dat,x$spec$dgp)$V_hat,
          value_from_em(global_em,x$dat,x$spec$dgp)$V_hat,
          value_from_em(statewise_em,x$dat,x$spec$dgp)$V_hat),
  mu_gap_s1=c(diff(original_em$mu_hat[1,]),diff(global_em$mu_hat[1,]),diff(statewise_em$mu_hat[1,])),
  mu_gap_s2=c(diff(original_em$mu_hat[2,]),diff(global_em$mu_hat[2,]),diff(statewise_em$mu_hat[2,])),
  mu_gap_s3=c(diff(original_em$mu_hat[3,]),diff(global_em$mu_hat[3,]),diff(statewise_em$mu_hat[3,])))
write.csv(permutation_demo,file.path(out,"statewise_permutation_counterexample.csv"),row.names=FALSE)
print(permutation_demo)

# Statewise trace of the unchanged update, with a larger diagnostic cap.
audit_one<-function(scenario,rep) {
  output<-file.path(out,sprintf("%s_%03d.rds",scenario,rep))
  if(file.exists(output))return(TRUE)
  x<-readRDS(file.path(study,"main/Tabular",scenario,sprintf("rep_%03d.rds",rep)))
  old<-x$runs[[2]]$fit
  s<-as.vector(x$dat$S);trace<-vector("list",5000);snapshots<-list();previous<-rep(NA_real_,3)
  hook<-function(fr) {
    k<-fr$iter;rows<-vector("list",3)
    for(ss in 1:3) {
      ii<-s==ss;d<-fr$eta[ii,]-fr$eta_old[ii,]
      mx<-apply(fr$log_eta[ii,,drop=FALSE],1,max)
      ll<-mean(mx+log(rowSums(exp(fr$log_eta[ii,,drop=FALSE]-mx))))
      rows[[ss]]<-data.frame(iter=k,state=ss,delta=max(abs(d)),rms=sqrt(mean(d^2)),
        avg_loglik=ll,delta_loglik=ll-previous[ss],mu_gap=diff(fr$mu_hat[ss,]))
      previous[ss]<<-ll
    }
    trace[[k]]<<-do.call(rbind,rows)
    if(k %in% c(old$raw$n_iter,100,500,1000))snapshots[[as.character(k)]]<<-diag_capture_raw(fr,"Tabular")
  }
  time0<-proc.time()[["elapsed"]]
  final_raw<-diag_tab_single(x$dat,diag_init(x$dat,2,rep,"Tabular"),max_iter=5000,tol=1e-6,diag_hook=hook)
  tr<-do.call(rbind,trace);snapshots[["final"]]<-final_raw
  stopifnot(max(abs(snapshots[[as.character(old$raw$n_iter)]]$eta-old$raw$eta))<1e-10)
  first<-function(tt,ss) {ii<-tr$iter[tr$state==ss & tr$delta<tt];if(length(ii))ii[1]else NA_integer_}
  state_summary<-data.frame(scenario=scenario,rep=rep,state=1:3,
    observations=as.numeric(table(factor(s,levels=1:3))),
    first_1e4=vapply(1:3,function(ss)first(1e-4,ss),integer(1)),
    first_1e6=vapply(1:3,function(ss)first(1e-6,ss),integer(1)),
    delta_at_100=tr$delta[tr$iter==100],delta_at_end=tr$delta[tr$iter==final_raw$n_iter])
  variants<-list(original_100=old$raw,extended=final_raw)
  rows<-list()
  for(name in names(variants))for(alignment in c("global","statewise"))for(moment in c("current","coherent")) {
    rr<-variants[[name]]
    if(alignment=="statewise")rr<-permute_states(rr,x$dat,which(rr$mu_hat[,2]<rr$mu_hat[,1]))
    em<-if(alignment=="statewise")pointwise_build(rr,x$dat)else diag_tab_build(rr,x$dat)
    coherent<-coherent_mean(em);inconsistency<-max(abs(em$theta_Sp_hat-coherent$theta_Sp_hat))
    if(moment=="coherent")em<-coherent
    val<-value_from_em(em,x$dat,x$spec$dgp)
    gap<-em$mu_hat[,2]-em$mu_hat[,1]
    rows[[length(rows)+1L]]<-data.frame(scenario=scenario,rep=rep,fit=name,alignment=alignment,moment=moment,
      iterations=rr$n_iter,converged=if(name=="original_100")old$status=="converged" else max(tail(tr$delta,3))<1e-6,
      last_delta=if(name=="original_100")tail(old$trace$delta_max,1) else max(tail(tr$delta,3)),
      avg_loglik=raw_loglik(rr,x$dat)/length(s),loglik_gain=(raw_loglik(rr,x$dat)-raw_loglik(old$raw,x$dat))/length(s),
      mean_inconsistency=inconsistency,min_abs_mu_gap=min(abs(gap)),
      value=val$V_hat,se=val$se,
      max_surrogate_bridge=max(vapply(1:3,function(ss) max(abs(c(0,1)-em$mu_hat[ss,1]))/abs(gap[ss]),numeric(1))))
  }
  saveRDS(list(state_summary=state_summary,variants=do.call(rbind,rows),trace=tr,snapshots=snapshots,
               elapsed=proc.time()[["elapsed"]]-time0),output)
  cat(scenario,rep,"iterations",final_raw$n_iter,"end delta",max(tail(tr$delta,3)),"\n");flush.console();TRUE
}
tasks<-expand.grid(scenario=c("typical","hard","state_dependent"),rep=1:20,stringsAsFactors=FALSE)
tasks<-rbind(tasks,data.frame(scenario="diagnostic_seed12",rep=12L))
status<-parallel::mclapply(seq_len(nrow(tasks)),function(i)audit_one(tasks$scenario[i],tasks$rep[i]),
                           mc.cores=2L,mc.set.seed=FALSE)
stopifnot(all(vapply(status,isTRUE,logical(1))))
files<-list.files(out,pattern="_[0-9]+\\.rds$",full.names=TRUE)
results<-lapply(files,readRDS)
states<-do.call(rbind,lapply(results,`[[`,"state_summary"))
variants<-do.call(rbind,lapply(results,`[[`,"variants"))
write.csv(states,file.path(out,"statewise_convergence.csv"),row.names=FALSE)
write.csv(variants,file.path(out,"ablation_values.csv"),row.names=FALSE)
regular<-variants %>% filter(scenario!="diagnostic_seed12",alignment=="global",moment=="current")
sumry<-regular %>% group_by(scenario,fit) %>% summarise(n=n(),converged=sum(converged),
  median_iterations=median(iterations),max_iterations=max(iterations),
  median_final_delta=median(last_delta),max_mean_inconsistency=max(mean_inconsistency),
  median_abs_gap=median(min_abs_mu_gap),min_abs_gap=min(min_abs_mu_gap),.groups="drop")
write.csv(sumry,file.path(out,"convergence_summary.csv"),row.names=FALSE)
state_sum<-states %>% filter(scenario!="diagnostic_seed12") %>% group_by(scenario,state) %>% summarise(
  median_n=median(observations),median_delta_at_100=median(delta_at_100),
  median_first_1e4=median(first_1e4),median_first_1e6=median(first_1e6),.groups="drop")
write.csv(state_sum,file.path(out,"statewise_summary.csv"),row.names=FALSE)
stopifnot(identical(unname(methods_hash),unname(tools::md5sum("Tabular/Methods.R"))))
print(sumry);print(state_sum)
cat("Audit complete. Original Tabular/Methods.R unchanged.\n")
