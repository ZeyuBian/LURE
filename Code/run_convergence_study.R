#!/usr/bin/env Rscript
# From Code: Rscript run_convergence_study.R PHASE SETTING OUTPUT [CORES]
# PHASE: pilot, main, stopping, tuning, value, case. SETTING: Tabular/Continuous/Both.
arg<-commandArgs(TRUE)
stopifnot(length(arg)>=3,arg[1] %in% c("pilot","main","stopping","tuning","value","case"),
          arg[2] %in% c("Tabular","Continuous","Both"))
root<-dirname(normalizePath(sub("^--file=","",grep("^--file=",commandArgs(),value=TRUE)[1])))
setwd(root);source("convergence_diagnostics.R")
phase<-arg[1];settings<-if(arg[2]=="Both") c("Tabular","Continuous") else arg[2]
dir.create(arg[3],recursive=TRUE,showWarnings=FALSE);out<-normalizePath(arg[3])
cores<-if(length(arg)>=4) as.integer(arg[4]) else 2L
sources<-c("convergence_diagnostics.R","run_convergence_study.R","state_dependent_error.R",
           "Tabular/Methods.R","Continuous/Methods_continuous.R")
for(file in sources) {
  dest<-file.path(out,"source_snapshot",file);dir.create(dirname(dest),recursive=TRUE,showWarnings=FALSE)
  if(!file.exists(dest)) stopifnot(file.copy(file,dest))
  else if(unname(tools::md5sum(file))!=unname(tools::md5sum(dest))) {
    revision<-file.path(out,"source_revisions",paste0(unname(tools::md5sum(file)),"_",basename(file)))
    dir.create(dirname(revision),recursive=TRUE,showWarnings=FALSE)
    if(!file.exists(revision)) stopifnot(file.copy(file,revision))
  }
}
saveRDS(tools::md5sum(sources),file.path(out,paste0("source_hashes_",phase,"_",arg[2],".rds")))
if(!file.exists(file.path(out,"config.rds"))) {
  saveRDS(list(n_rep=20L,N=50L,T=50L,gamma=.7,starts=diag_start_names,
    main=c("typical","hard"),stress=c("weak_anchor","weak_proxy","state_dependent"),
    stopping_n=10L,value_n=10L,settings=c("Tabular","Continuous"),excluded="Gym",
    baseline="Current production stopping and alignment; diagnostic comparisons are labeled separately",
    source_md5=tools::md5sum(sources),session=sessionInfo(),created=Sys.time()),file.path(out,"config.rds"))
}
file_main<-function(setting,scenario,rep,pilot=FALSE)
  file.path(out,if(pilot) "pilot" else "main",setting,scenario,sprintf("rep_%03d.rds",rep))
save_output<-function(x,file) {
  dir.create(dirname(file),recursive=TRUE,showWarnings=FALSE)
  tmp<-paste0(file,".partial");saveRDS(x,tmp,compress="gzip");stopifnot(file.rename(tmp,file))
}

main_one<-function(setting,scenario,rep,pilot=FALSE) {
  file<-file_main(setting,scenario,rep,pilot)
  if(file.exists(file)) return(TRUE)
  e<-diag_load(root,setting);spec<-diag_spec(e,setting,scenario)
  dat<-diag_data(e,spec,rep,pilot=pilot);test<-diag_data(e,spec,rep,TRUE,pilot)
  truth<-diag_truth(spec,test);runs<-vector("list",8);tables<-vector("list",8)
  for(st in 1:8) {
    fit<-diag_fit(e,setting,dat,diag_init(dat,st,rep,setting))
    diagnostic<-tryCatch(diag_metrics(e,spec,fit,dat,test,truth),error=function(err)
      list(metrics=data.frame(status="diagnostic_error",error=conditionMessage(err)),pred=NULL))
    tables[[st]]<-cbind(data.frame(setting=setting,scenario=scenario,rep=rep,start=diag_start_names[st],start_id=st),diagnostic$metrics)
    fit$raw<-diag_compact_raw(fit$raw)
    runs[[st]]<-list(fit=fit,pred=diagnostic$pred)
  }
  metrics<-diag_bind_rows(tables)
  save_output(list(spec=spec,rep=rep,dat=dat,test=test,truth=truth,runs=runs,metrics=metrics,complete=TRUE),file)
  cat(sprintf("%s %s %s rep=%02d: converged %d/8, wrong global orientation %d/8, %.2fs\n",
    if(pilot)"pilot" else "main",setting,scenario,rep,sum(metrics$status=="converged"),
    sum(metrics$orientation_wrong,na.rm=TRUE),sum(metrics$elapsed,na.rm=TRUE)));flush.console()
  TRUE
}

stopping_one<-function(setting,scenario,rep) {
  file<-file.path(out,"stopping",setting,scenario,sprintf("rep_%03d.rds",rep))
  if(file.exists(file)) return(TRUE)
  x<-readRDS(file_main(setting,scenario,rep));e<-diag_load(root,setting);runs<-list();rows<-list()
  for(st in c(2L,6L)) {
    fit<-diag_fit(e,setting,x$dat,diag_init(x$dat,st,rep,setting),cap=500L,tol=-1,checkpoints=TRUE)
    if(is.null(fit$raw)) {
      rows[[length(rows)+1]]<-data.frame(setting=setting,scenario=scenario,rep=rep,start=diag_start_names[st],status="error",error=fit$error)
      runs[[as.character(st)]]<-fit;next
    }
    final<-diag_metrics(e,x$spec,fit,x$dat,x$test,x$truth)
    for(key in names(fit$snapshots)) {
      snap<-fit$snapshots[[key]];f2<-fit;f2$raw<-snap$raw;f2$snapshots<-NULL
      f2$trace<-fit$trace[seq_len(as.integer(key)),,drop=FALSE]
      dd<-diag_metrics(e,x$spec,f2,x$dat,x$test,x$truth)
      row<-data.frame(setting=setting,scenario=scenario,rep=rep,start=diag_start_names[st],
        iteration=as.integer(key),criteria=paste(snap$keys,collapse=";"),
        posterior_drift=sqrt(mean((dd$pred$eta-final$pred$eta)^2)),
        mu_drift=sqrt(mean((dd$pred$mu-final$pred$mu)^2)),
        reward_drift=sqrt(mean((dd$pred$reward-final$pred$reward)^2)),
        eta_rmse=dd$metrics$eta_rmse,reference_delta=tail(fit$trace$delta_max,1),
        reference_stable=tail(fit$trace$delta_max,1)<1e-5,
        reference_orientation_wrong=final$metrics$orientation_wrong)
      rows[[length(rows)+1L]]<-row
      fit$snapshots[[key]]$raw<-diag_compact_raw(snap$raw)
    }
    fit$raw<-diag_compact_raw(fit$raw);runs[[as.character(st)]]<-fit
  }
  save_output(list(runs=runs,metrics=diag_bind_rows(rows),complete=TRUE),file)
  cat("stopping",setting,scenario,"rep",rep,"complete\n");flush.console();TRUE
}

tuning_one<-function(setting,scenario,rep) {
  file<-file.path(out,"tuning",setting,scenario,sprintf("rep_%03d.rds",rep))
  if(file.exists(file)) return(TRUE)
  x<-readRDS(file_main(setting,scenario,rep));e<-diag_load(root,setting);rows<-runs<-list()
  basepred<-x$runs[[2]]$pred
  for(cf in c(.5,2)) {
    fit<-diag_fit(e,setting,x$dat,diag_init(x$dat,2,rep,setting),clip_factor=cf)
    dd<-diag_metrics(e,x$spec,fit,x$dat,x$test,x$truth)
    row<-cbind(data.frame(setting=setting,scenario=scenario,rep=rep,clip_factor=cf),dd$metrics)
    row$posterior_difference<-if(is.null(dd$pred)||is.null(basepred)) NA_real_ else sqrt(mean((dd$pred$eta-basepred$eta)^2))
    rows[[length(rows)+1L]]<-row;fit$raw<-diag_compact_raw(fit$raw);runs[[as.character(cf)]]<-fit
  }
  save_output(list(runs=runs,metrics=diag_bind_rows(rows),complete=TRUE),file)
  cat("tuning",setting,scenario,"rep",rep,"complete\n");flush.console();TRUE
}

value_one<-function(setting,scenario,rep) {
  file<-file.path(out,"value",setting,scenario,sprintf("rep_%03d.rds",rep))
  if(file.exists(file)) return(TRUE)
  x<-readRDS(file_main(setting,scenario,rep));e<-diag_load(root,setting)
  cache<-list();current_start<-1L;fold_counter<-0L;fit_records<-list()
  # Replacement fits are solely plumbing: downstream Q, omega and IF are original.
  fetch<-function(dat,gamma,...) {
    fold_counter<<-fold_counter+1L;fold<-fold_counter
    if(setting=="Tabular") {
      available<-lapply(x$runs[1:7],function(z)z$fit$raw)
    } else {
      key<-as.character(fold)
      if(is.null(cache[[key]])) {
        candidate<-vector("list",7)
        for(st in 1:7) {
          f<-diag_fit(e,setting,dat,diag_init(dat,st,rep,setting,fold))
          candidate[[st]]<-diag_compact_raw(f$raw)
          fit_records[[paste(fold,st,sep="_")]]<<-list(status=f$status,trace=f$trace,error=f$error,warnings=f$warnings)
        }
        cache[[key]]<<-list(dat=dat,raw=candidate)
      } else stopifnot(identical(cache[[key]]$dat,dat))
      available<-cache[[key]]$raw
    }
    ids<-if(current_start<=7L) current_start else if(current_start==8L) c(2L,4L) else 1:7
    score<-vapply(available[ids],function(z)if(is.null(z)) -Inf else z$loglik,numeric(1))
    if(!any(is.finite(score))) stop("No finite candidate fit for this training fold")
    chosen<-ids[which.max(score)]
    diag_build(e,setting,available[[chosen]],dat)
  }
  if(setting=="Tabular") {
    e$em_tabular<-function(dat,nS,gamma,...) fetch(dat,gamma)
  } else e$em_continuous<-fetch
  rows<-list();strategies<-c(diag_start_names[1:7],"current_two_start","best_of_seven")
  for(st in seq_along(strategies)) {
    current_start<-st;fold_counter<-0L;messages<-character();failure<-NA_character_
    # Identical fold assignment and integration draws for each candidate.
    set.seed(33000000L+(setting=="Continuous")*100000L+rep*100L)
    time0<-proc.time()[["elapsed"]]
    val<-tryCatch(withCallingHandlers(e$.original_value(x$dat,x$spec$dgp,.7),
      warning=function(w){messages<<-c(messages,conditionMessage(w));invokeRestart("muffleWarning")}),
      error=function(err){failure<<-conditionMessage(err);list(V_hat=NA_real_,se=NA_real_,ci_lo=NA_real_,ci_hi=NA_real_)})
    rows[[st]]<-data.frame(setting=setting,scenario=scenario,rep=rep,strategy=strategies[st],
      estimate=val$V_hat,se=val$se,lower=val$ci_lo,upper=val$ci_hi,error=failure,
      warnings=paste(unique(messages),collapse=" | "),elapsed=proc.time()[["elapsed"]]-time0)
  }
  save_output(list(metrics=do.call(rbind,rows),fold_cache=cache,fit_records=fit_records,complete=TRUE),file)
  vals<-vapply(rows[1:7],function(z)z$estimate,numeric(1))
  cat(sprintf("value %s %s rep=%02d range=[%.5f, %.5f]\n",setting,scenario,rep,min(vals),max(vals)));flush.console();TRUE
}

scenarios<-if(phase=="main") c("typical","hard","weak_anchor","weak_proxy","state_dependent") else c("typical","hard")
reps<-if(phase=="main") 1:20 else if(phase=="pilot") 1:3 else 1:10
if(nzchar(Sys.getenv("LURE_DIAG_REPS",""))) reps<-as.integer(strsplit(Sys.getenv("LURE_DIAG_REPS"),",")[[1]])
if(phase=="case") {settings<-"Tabular";scenarios<-"diagnostic_seed12";reps<-12L}
tasks<-expand.grid(setting=settings,scenario=scenarios,rep=reps,stringsAsFactors=FALSE)
cat("Starting",phase,"with",nrow(tasks),"datasets and",cores,"workers at",format(Sys.time()),"\n")
status<-parallel::mclapply(seq_len(nrow(tasks)),function(i) {
  a<-tasks[i,]
  tryCatch({
    if(phase %in% c("main","pilot")) main_one(a$setting,a$scenario,a$rep,phase=="pilot")
    else if(phase=="stopping") stopping_one(a$setting,a$scenario,a$rep)
    else if(phase=="tuning") tuning_one(a$setting,a$scenario,a$rep)
    else if(phase=="value") value_one(a$setting,a$scenario,a$rep)
    else {main_one(a$setting,a$scenario,a$rep);value_one(a$setting,a$scenario,a$rep)}
  },error=function(err) {cat("WORKER ERROR",a$setting,a$scenario,a$rep,conditionMessage(err),"\n");FALSE})
},mc.cores=cores,mc.set.seed=FALSE)
stopifnot(all(vapply(status,isTRUE,logical(1))))
cat("Finished",phase,"at",format(Sys.time()),"\n")
