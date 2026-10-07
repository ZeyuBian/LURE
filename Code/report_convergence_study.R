#!/usr/bin/env Rscript
arg<-commandArgs(TRUE);stopifnot(length(arg)==1L)
source("convergence_diagnostics.R")
suppressPackageStartupMessages({library(dplyr);library(ggplot2)})
out<-normalizePath(arg[1]);dir.create(file.path(out,"figures"),showWarnings=FALSE)
scenes<-c("typical","hard","weak_anchor","weak_proxy","state_dependent")
labels<-c(typical="10% error",hard="30% error",weak_anchor="45% error / weak anchor",
          weak_proxy="Weak transition proxy",state_dependent="State-dependent error")
files<-list.files(file.path(out,"main"),pattern="\\.rds$",recursive=TRUE,full.names=TRUE)
metrics<-selected<-within<-traces<-list();case<-NULL
for(file in files) {
  x<-readRDS(file)
  if(x$spec$scenario=="diagnostic_seed12") {case<-x;next}
  stopifnot(isTRUE(x$complete),nrow(x$metrics)==8L)
  z<-x$metrics;metrics[[length(metrics)+1L]]<-z
  choices<-list(default=2L,current=if(x$spec$setting=="Tabular")2L else c(2L,4L),best7=1:7)
  for(strategy in names(choices)) {
    ids<-choices[[strategy]]; scores<-z$avg_loglik[ids];scores[!is.finite(scores)]<--Inf
    j<-ids[which.max(scores)]
    selected[[length(selected)+1L]]<-cbind(strategy=strategy,z[j,,drop=FALSE])
  }
  pred<-lapply(x$runs[1:7],function(r)if(is.null(r$pred))NULL else r$pred$eta[,2])
  valid<-which(vapply(pred,function(p)!is.null(p)&&all(is.finite(p)),logical(1)))
  pairs<-if(length(valid)>=2) combn(valid,2) else matrix(integer(),2,0)
  delta<-if(ncol(pairs)) apply(pairs,2,function(a)sqrt(mean((pred[[a[1]]]-pred[[a[2]]])^2))) else NA_real_
  informed_pairs<-combn(1:5,2)
  delta_informed<-if(all(1:5 %in% valid))apply(informed_pairs,2,function(a)sqrt(mean((pred[[a[1]]]-pred[[a[2]]])^2)))else NA_real_
  within[[length(within)+1L]]<-data.frame(setting=x$spec$setting,scenario=x$spec$scenario,rep=x$rep,
    max_pairwise_posterior_rmse=if(length(valid)==7)max(delta)else NA_real_,
    informed_max_pairwise_posterior_rmse=max(delta_informed),
    reversal_posterior_difference=if(!is.null(x$runs[[8]]$pred))sqrt(mean((x$runs[[2]]$pred$eta-x$runs[[8]]$pred$eta)^2))else NA_real_,
    valid_starts=length(valid))
  if(x$rep==1L && x$spec$scenario %in% c("typical","hard")) for(st in 1:8)
    traces[[length(traces)+1L]]<-cbind(setting=x$spec$setting,scenario=x$spec$scenario,
      start=diag_start_names[st],x$runs[[st]]$fit$trace)
}
m<-diag_bind_rows(metrics);sel<-diag_bind_rows(selected);w<-diag_bind_rows(within);tr<-diag_bind_rows(traces)
stopifnot(nrow(m)==1600L,nrow(w)==200L,all(table(m$setting,m$scenario)==160L))
write.csv(m,file.path(out,"all_nuisance_runs.csv"),row.names=FALSE)
write.csv(sel,file.path(out,"selected_nuisance_runs.csv"),row.names=FALSE)
write.csv(w,file.path(out,"within_dataset_stability.csv"),row.names=FALSE)
ci<-function(k,n) {b<-binom.test(k,n)$conf.int;sprintf("%.1f–%.1f%%",100*b[1],100*b[2])}
summary<-sel %>% group_by(setting,scenario,strategy) %>% summarise(
  datasets=n(),converged=sum(status=="converged"),capped=sum(status=="iteration_cap"),
  errors=sum(!status %in% c("converged","iteration_cap")),median_iterations=median(n_iter),
  orientation_wrong=sum(orientation_wrong),local_mixed=sum(local_wrong_fraction>0 & local_wrong_fraction<1,na.rm=TRUE),
  rule_disagreement=sum(manuscript_rule_disagrees),median_eta_rmse=median(eta_rmse),
  max_eta_rmse=max(eta_rmse),median_mu_rmse=median(mu_rmse),median_reward_rmse=median(reward_rmse),
  median_transition_rmse=median(transition_mean_rmse),mu_nonpositive_datasets=sum(mu_gap_negative>0),
  objective_decrease_datasets=sum(objective_decreases>0),.groups="drop")
summary$convergence_interval<-mapply(ci,summary$converged,summary$datasets)
summary$orientation_error_interval<-mapply(ci,summary$orientation_wrong,summary$datasets)
write.csv(summary,file.path(out,"nuisance_summary.csv"),row.names=FALSE)
by_start<-m %>% group_by(setting,scenario,start,start_id) %>% summarise(n=n(),converged=sum(status=="converged"),
  iterations=median(n_iter),eta_rmse=median(eta_rmse),orientation_wrong=sum(orientation_wrong),
  local_mixed=sum(local_wrong_fraction>0&local_wrong_fraction<1,na.rm=TRUE),.groups="drop")
write.csv(by_start,file.path(out,"initialization_summary.csv"),row.names=FALSE)
within_summary<-w %>% group_by(setting,scenario) %>% summarise(
  median_pairwise=median(max_pairwise_posterior_rmse),max_pairwise=max(max_pairwise_posterior_rmse),
  informed_median=median(informed_max_pairwise_posterior_rmse),informed_max=max(informed_max_pairwise_posterior_rmse),
  reversal_max=max(reversal_posterior_difference),.groups="drop")
write.csv(within_summary,file.path(out,"initialization_spread.csv"),row.names=FALSE)

stopfiles<-list.files(file.path(out,"stopping"),pattern="\\.rds$",recursive=TRUE,full.names=TRUE)
stopifnot(length(stopfiles)==40L)
stops<-policies<-list()
for(file in stopfiles) {
  x<-readRDS(file);stops[[length(stops)+1]]<-x$metrics
  for(st in c(2L,6L)) {
    f<-x$runs[[as.character(st)]];zz<-x$metrics[x$metrics$start==diag_start_names[st],]
    stopifnot(!is.null(f$raw),nrow(f$trace)==500L)
    for(type in c("posterior","joint")) for(tol in c(1e-3,1e-4,1e-5)) for(cap in c(100L,300L,500L)) {
      key<-paste0(type,"_",format(tol,scientific=TRUE));hit<-f$first_hits[[key]]
      met<-!is.null(hit)&&hit<=cap;iteration<-if(met)hit else cap
      rr<-zz[zz$iteration==iteration,,drop=FALSE];stopifnot(nrow(rr)==1L)
      policies[[length(policies)+1]]<-cbind(rule=type,tolerance=tol,cap=cap,criterion_met=met,rr)
    }
  }
}
stop_data<-diag_bind_rows(stops);stop_policies<-diag_bind_rows(policies)
write.csv(stop_data,file.path(out,"stopping_checkpoints.csv"),row.names=FALSE)
write.csv(stop_policies,file.path(out,"stopping_rules.csv"),row.names=FALSE)
stop_summary<-stop_policies %>% group_by(setting,scenario,start,rule,tolerance,cap) %>% summarise(
  n=n(),met=sum(criterion_met),median_iteration=median(iteration),median_drift=median(posterior_drift),
  max_drift=max(posterior_drift),reference_unstable=sum(!reference_stable),.groups="drop")
write.csv(stop_summary,file.path(out,"stopping_summary.csv"),row.names=FALSE)

tunefiles<-list.files(file.path(out,"tuning"),pattern="\\.rds$",recursive=TRUE,full.names=TRUE)
stopifnot(length(tunefiles)==40L)
tune<-diag_bind_rows(lapply(tunefiles,function(f)readRDS(f)$metrics))
write.csv(tune,file.path(out,"tuning_runs.csv"),row.names=FALSE)
tune_summary<-tune %>% group_by(setting,scenario,clip_factor) %>% summarise(n=n(),
  converged=sum(status=="converged"),median_posterior_difference=median(posterior_difference),
  max_posterior_difference=max(posterior_difference),eta_rmse=median(eta_rmse),.groups="drop")
write.csv(tune_summary,file.path(out,"tuning_summary.csv"),row.names=FALSE)

vfiles<-list.files(file.path(out,"value"),pattern="\\.rds$",recursive=TRUE,full.names=TRUE)
v<-diag_bind_rows(lapply(vfiles,function(f)readRDS(f)$metrics))
case_values<-v %>% filter(scenario=="diagnostic_seed12")
v<-v %>% filter(scenario %in% c("typical","hard"))
stopifnot(nrow(v)==360L)
write.csv(v,file.path(out,"all_value_estimates.csv"),row.names=FALSE)
vr<-v %>% group_by(setting,scenario,rep) %>% group_modify(function(x,key) {
  seven<-x$estimate[match(diag_start_names[1:7],x$strategy)]
  informed<-seven[1:5]
  current<-x[x$strategy==if(key$setting=="Tabular")"default_70" else "current_two_start",]
  best<-x[x$strategy=="best_of_seven",]
  data.frame(valid=sum(is.finite(seven)),range=diff(range(seven)),sd=sd(seven),
    informed_range=diff(range(informed)),current_value=current$estimate,current_se=current$se,
    range_over_se=if(is.finite(current$se)&&current$se>0)diff(range(seven))/current$se else NA_real_,
    best7_value=best$estimate,best7_se=best$se,best7_current_difference=best$estimate-current$estimate)
}) %>% ungroup()
write.csv(vr,file.path(out,"within_dataset_value_spread.csv"),row.names=FALSE)
value_summary<-vr %>% group_by(setting,scenario) %>% summarise(n=n(),fully_finite=sum(valid==7L),
  median_range=median(range),max_range=max(range),median_informed_range=median(informed_range),
  max_informed_range=max(informed_range),median_current_se=median(current_se),median_range_over_se=median(range_over_se),
  median_best_current_abs_difference=median(abs(best7_current_difference)),
  max_best_current_abs_difference=max(abs(best7_current_difference)),.groups="drop")
write.csv(value_summary,file.path(out,"value_summary.csv"),row.names=FALSE)

# Training-fold diagnostics establish whether a value outlier arose in the latent
# fit before Q, omega and bridge amplification. No additional fits are performed.
fold_rows<-list()
for(file in vfiles[grepl("/Continuous/",vfiles)]) {
  vv<-readRDS(file);z<-vv$metrics[1,];if(!z$scenario %in% c("typical","hard"))next
  xx<-readRDS(file.path(out,"main","Continuous",z$scenario,sprintf("rep_%03d.rds",z$rep)))
  e<-diag_load(normalizePath("."),"Continuous")
  for(fold in names(vv$fold_cache)) for(st in 1:7) {
    cache<-vv$fold_cache[[fold]];raw<-cache$raw[[st]];rec<-vv$fit_records[[paste(fold,st,sep="_")]]
    fit<-list(raw=raw,trace=rec$trace,status=rec$status,clip_factor=1,elapsed=NA_real_,warning_count=length(rec$warnings))
    dd<-diag_metrics(e,xx$spec,fit,cache$dat,xx$test,xx$truth)
    fold_rows[[length(fold_rows)+1]]<-cbind(setting="Continuous",scenario=z$scenario,rep=z$rep,
      fold=as.integer(fold),start=diag_start_names[st],dd$metrics)
  }
}
fold_metrics<-diag_bind_rows(fold_rows)
write.csv(fold_metrics,file.path(out,"value_training_fold_diagnostics.csv"),row.names=FALSE)

theme_set(theme_minimal(base_size=12)+theme(panel.grid.minor=element_blank(),legend.position="bottom"))
pal<-c(surrogate_55="#4477AA",default_70="#007766",surrogate_90="#AA3377",perturbed_1="#66CCEE",
       perturbed_2="#228833",random_1="#EE7733",random_2="#CC3311",reversed_70="#999999")
tr$condition<-factor(labels[tr$scenario],levels=labels[c("typical","hard")])
p<-ggplot(tr,aes(iter,pmax(delta_max,1e-10),color=start))+geom_line(linewidth=.65)+
  scale_y_log10()+scale_color_manual(values=pal)+facet_grid(setting~condition)+
  labs(x="Iteration",y="Maximum posterior change (log scale)",color="Initialization",
       title="Convergence paths for the first prespecified dataset",
       subtitle="Original stopping rules; all eight starts shown. Aggregate tables use all 20 datasets.")
ggsave(file.path(out,"figures","convergence_paths.png"),p,width=11,height=7,dpi=180)
plot_init<-by_start %>% filter(scenario %in% c("typical","hard"),start_id<=7) %>%
  mutate(condition=labels[scenario],start=factor(start,levels=diag_start_names[1:7]))
p<-ggplot(plot_init,aes(start,converged/20,fill=start))+geom_col(width=.72)+
  facet_grid(setting~condition)+scale_fill_manual(values=pal)+scale_y_continuous(limits=c(0,1),labels=scales::percent)+
  labs(x=NULL,y="Fraction meeting the original stopping rule",title="Convergence across 20 independent datasets per setting")+
  theme(legend.position="none",axis.text.x=element_text(angle=35,hjust=1))
ggsave(file.path(out,"figures","convergence_rates.png"),p,width=11,height=6.5,dpi=180)
plot_error<-sel %>% filter(strategy %in% c("current","best7")) %>%
  mutate(condition=factor(labels[scenario],levels=labels),strategy=recode(strategy,current="Current strategy",best7="Highest likelihood of seven"))
p<-ggplot(plot_error,aes(condition,eta_rmse,color=strategy))+geom_boxplot(outlier.shape=NA,position=position_dodge(.75))+
  geom_point(position=position_jitterdodge(jitter.width=.08,dodge.width=.75,seed=13),size=1.3,alpha=.55)+
  facet_wrap(~setting,ncol=1,scales="free_y")+scale_color_manual(values=c("#007766","#CC3311"))+
  labs(x=NULL,y="Posterior RMSE against simulation truth",color=NULL,title="Latent-action recovery after practical label alignment")+
  theme(axis.text.x=element_text(angle=15,hjust=1))
ggsave(file.path(out,"figures","nuisance_accuracy.png"),p,width=11,height=7,dpi=180)
plot_value<-v %>% filter(strategy %in% diag_start_names[1:7]) %>%
  mutate(strategy=factor(strategy,levels=diag_start_names[1:7]),condition=labels[scenario])
p<-ggplot(plot_value,aes(strategy,estimate,group=rep,color=factor(rep)))+
  geom_line(alpha=.5,linewidth=.45)+geom_point(size=1.7)+facet_wrap(~setting+condition,ncol=2,scales="free_y")+
  scale_y_continuous(trans=scales::pseudo_log_trans(sigma=1))+
  labs(x=NULL,y="LURE estimate (symmetric log scale)",title="Value sensitivity to initialization",
       subtitle="Ten fixed datasets per panel; a line connects starts on the same dataset. All estimates retained.")+
  theme(legend.position="none",axis.text.x=element_text(angle=35,hjust=1))
ggsave(file.path(out,"figures","value_sensitivity.png"),p,width=11,height=7,dpi=180)

mdtable<-function(df) {
  df<-as.data.frame(df);for(nm in names(df)) {
    if(is.numeric(df[[nm]]))df[[nm]]<-formatC(df[[nm]],digits=4,format="fg",flag="#")
    df[[nm]]<-gsub("\\|","/",as.character(df[[nm]]))
  }
  c(paste0("| ",paste(names(df),collapse=" | ")," |"),
    paste0("| ",paste(rep("---",ncol(df)),collapse=" | ")," |"),
    apply(df,1,function(r)paste0("| ",paste(r,collapse=" | ")," |")),"")
}
current<-summary %>% filter(strategy=="current") %>% arrange(setting,match(scenario,scenes))
main_table<-current %>% filter(scenario %in% c("typical","hard")) %>% transmute(
  Environment=setting,Condition=labels[scenario],Converged=paste0(converged,"/",datasets),
  `95% interval`=convergence_interval,`Median iterations`=median_iterations,
  `Median posterior RMSE`=median_eta_rmse,`Wrong global orientation`=paste0(orientation_wrong,"/",datasets),
  `Mixed state labels`=ifelse(setting=="Tabular",paste0(local_mixed,"/",datasets),"not tabulated"))
stress_table<-current %>% filter(!scenario %in% c("typical","hard")) %>% transmute(
  Environment=setting,Condition=labels[scenario],Converged=paste0(converged,"/",datasets),
  `Median posterior RMSE`=median_eta_rmse,`Wrong global orientation`=paste0(orientation_wrong,"/",datasets),
  `Mixed state labels`=ifelse(setting=="Tabular",paste0(local_mixed,"/",datasets),"not tabulated"))
spread_table<-within_summary %>% transmute(Environment=setting,Condition=labels[scenario],
  `Median posterior difference across 7 starts`=median_pairwise,
  `Maximum posterior difference across 7 starts`=max_pairwise,
  `Median across 5 informed starts`=informed_median,
  `Maximum across 5 informed starts`=informed_max)
value_table<-value_summary %>% transmute(Environment=setting,Condition=labels[scenario],Datasets=n,
  `Median range across 7 starts`=median_range,`Maximum range`=max_range,
  `Median range across 5 informed starts`=median_informed_range,
  `Median current IF SE`=median_current_se)
stop_table<-stop_summary %>% filter(start=="default_70",rule=="posterior",cap==100,
  tolerance==ifelse(setting=="Tabular",1e-4,1e-3)) %>% transmute(Environment=setting,Condition=labels[scenario],
    `Original criterion met`=paste0(met,"/",n),`Median drift to iteration 500`=median_drift,
    `Maximum drift`=max_drift,`Still above 1e-5 at iteration 500`=paste0(reference_unstable,"/",n))
extended_table<-stop_summary %>% filter(start=="default_70",rule=="joint",tolerance==1e-5,cap==500) %>%
  transmute(Environment=setting,Condition=labels[scenario],`Joint criterion met by 500`=paste0(met,"/",n),
            `Median stopping iteration`=median_iteration,`Maximum subsequent posterior drift`=max_drift)
rule_table<-current %>% transmute(Environment=setting,Condition=labels[scenario],
  `Code/manuscript global rule disagree`=paste0(rule_disagreement,"/",datasets),
  `Any nonpositive state-specific surrogate contrast`=paste0(mu_nonpositive_datasets,"/",datasets))
case_lines<-if(nrow(case_values)) c("Previously observed extreme tabular case (separate diagnostic, not part of the 20-dataset rates):","",
  mdtable(case_values %>% select(strategy,estimate,se))) else character()
fold_bad<-fold_metrics %>% filter(orientation_wrong | eta_rmse>.10 | status!="converged") %>%
  select(scenario,rep,fold,start,status,n_iter,eta_rmse,orientation_wrong,mu_gap_negative)
write.csv(fold_bad,file.path(out,"value_fold_flags.csv"),row.names=FALSE)
report<-c("# Algorithm convergence and initialization: 20-dataset study","",
  "Completed for Tabular and Continuous only. Gym was excluded at the user's request.","",
  "20 independent datasets in each of 10 environment/condition cells; N=50, T=50, gamma=0.7. Eight starts per dataset give 1,600 main nuisance fits. The 12-dataset pilot uses separate seeds and is excluded from the reported rates. The first 10 datasets in each of four main cells are used for stopping, tuning, and value sensitivity.","",
  "The main findings are: Continuous fits selected by the current surrogate-informed strategy met the stopping rule in all 20 datasets in every condition. Tabular fits were usually truncated at the 100-iteration cap. Unanchored random starts sometimes stabilized at substantially different latent fits in both environments; numerical stopping alone did not establish correct recovery. In the main conditions, the five surrogate-informed starts produced closely agreeing values, while random starts sometimes produced large value changes. Tabular weak-separation cases also exposed state-specific orientation problems.","",
  "The current strategy is the single 0.70 surrogate-informed start in Tabular and the highest working likelihood of the default and one perturbed start in Continuous. This matches the current restart structure. Original probability safeguards, stopping thresholds, label alignment, downstream estimators, and IF standard errors are retained. The Tabular E-step was vectorized and verified against the original implementation; Continuous uses instrumented copies of the original functions.","",
  "## Convergence under the current strategy","",mdtable(main_table),
  "Converged means the implemented posterior-change tolerance was met before or at the cap; reaching the cap alone does not count as convergence. Thresholds are 1e-4 (Tabular) and 1e-3 (Continuous), both with maximum 100 iterations. Intervals are exact binomial intervals over independent datasets.","",
  "![Convergence by initialization](figures/convergence_rates.png)","",
  "![Prespecified convergence trajectories](figures/convergence_paths.png)","",
  "## Agreement across initializations","",
  "For each dataset, compute the largest pairwise posterior RMSE between starts on the same independent evaluation sample. These are differences between fitted posteriors, not errors against truth. Compare all seven starts with the five surrogate-informed starts, then summarize across 20 datasets:","",mdtable(spread_table),
  "In particular, a Continuous fit can meet its stopping rule yet differ greatly from the well-recovered surrogate-informed solution. Such a case is not automatically a harmless global label swap. The current and best-of-seven selection summaries below should be read together with this all-start table.","",
  "## Weak separation and label alignment","",mdtable(stress_table),
  "Wrong global orientation means the reverse global permutation has smaller posterior loss against the true posterior on independent evaluation data, by more than 1e-8. Mixed state labels means this preferred orientation differs across the three tabular states. A single global relabeling cannot repair such local inconsistency. The deliberately reversed start is retained as a permutation check and excluded from the seven-start spread summaries.","",
  "The weak-anchor condition sets constant error to 0.45, retaining reward and transition signals. The weak-proxy condition reduces action-dependent transition differences to 25% while retaining the midpoint and noise. State-dependent error uses the previously specified strength 0.75 and reference rate 0.30; its actual rate is recorded in all_nuisance_runs.csv.","",
  "![Nuisance recovery accuracy](figures/nuisance_accuracy.png)","",
  "The manuscript's common-state average measurement contrast and the implemented posterior-weighted within-component surrogate rule can differ:","",mdtable(rule_table),
  "The Continuous alignment code also refits the behavior model after a reversal. The pilot reproduced the same raw mixture up to permutation but found a maximum posterior difference of about 1.8e-6 after this refit. This is recorded as an implementation detail, rather than hidden by changing the method.","",
  "## Stopping-rule and tuning checks","",
  "The same update paths were continued to 500 iterations from the default and one unanchored random start. The table below describes the default start. Drift is posterior RMSE on independent evaluation data from the original stopping point (or cap) to iteration 500.","",mdtable(stop_table),
  "Iteration 500 is a numerical reference, not truth. Runs still changing at that point are explicitly counted. Candidate joint stopping requires maximum posterior change below 1e-5 and absolute average working-log-likelihood change below 1e-6 for three consecutive iterations:","",mdtable(extended_table),
  "All comparisons of tolerances 1e-3/1e-4/1e-5, caps 100/300/500 and posterior-only/joint rules are in stopping_summary.csv. This is an empirical stopping assessment and does not establish a convergence theorem.","",
  "Probability clipping for the behavior and measurement models was varied to half and twice the original bound, holding other safeguards fixed:","",mdtable(tune_summary),
  "## Limited value-sensitivity check","",
  "Each dataset uses seven non-reversal starts; the five informed starts are confidence 0.55/0.70/0.90 plus two perturbed starts. The range is computed within the same dataset, with identical folds and downstream integration draws. Q and omega are recomputed for each fit. Tabular uses its current full-data estimator; Continuous uses its current two trajectory folds, so its nuisance training sample is smaller than in the standalone full-data nuisance experiment.","",mdtable(value_table),
  "![Value sensitivity](figures/value_sensitivity.png)","",
  "The saved current two-start and best-of-seven strategies select by working likelihood, never by the true policy value. The values and all errors are in all_value_estimates.csv. The IF SE is the implementation's reported SE, not an independently validated benchmark. Ten datasets per cell support a paired initialization diagnostic; they do not establish nominal coverage.","",
  sprintf("There were %d nonfinite estimates among %d value evaluations. No estimates or starts were trimmed.",sum(!is.finite(v$estimate)),nrow(v)),"",
  "Training-fold diagnostics for all 280 Continuous fits are provided in value_training_fold_diagnostics.csv. These distinguish instability in the latent fit from amplification by subsequent value estimation; full-data nuisance stability alone does not settle the latter.","",
  case_lines,
  "## Interpretation and reproducibility","",
  "For the reviewer response, the evidence supports emphasizing surrogate-informed initialization, explicitly reporting cap events, and retaining the small downstream value diagnostic. The Tabular implementation needs a revised stopping policy and scrutiny of state-specific orientation before claiming routine convergence. Increasing the cap to 500 helps but still leaves some paths changing. Selecting the highest likelihood across unrestricted starts can worsen Tabular latent recovery under weak separation, so that selection rule alone is not a remedy.","",
  "Numerical stopping, agreement across initializations, recovery of the true latent functions, and stability of the final value are separate outcomes. Report them separately in the response letter. Empirical numerical stability does not establish the nuisance-rate conditions or global convergence.","",
  "With 20 independent datasets, zero observed failures has a one-sided 95% binomial upper bound of about 13.9%. The eight starts within one dataset are dependent and cannot be counted as eight independent datasets for a rare-failure claim.","",
  "Data-generation seeds are separate from pilot, evaluation, initialization and downstream seeds. Error conditions share latent trajectories and random uniforms where permitted by the DGP. Every per-dataset RDS retains input data, fit parameters, iteration traces, diagnostics and warnings. Saved regression objects omit training-only arrays and formula call environments after prediction-equivalence checks; metrics are unchanged. source_snapshot, source_revisions, and phase source hashes document code versions.","",
  "The original estimator source files were not edited by this study. The original default-start value and IF SE were independently reproduced for both environments. Tests also verify nuisance fit equivalence, pure raw-component reversal, compact-fit prediction, and the weak-proxy generator.","",
  "Reproduce from Code with run_convergence_study.R PHASE Both OUTPUT 2, using phases pilot, main, stopping, tuning, value, and case; then run report_convergence_study.R OUTPUT. Existing completed datasets are skipped. The single seed-12 diagnostic is separate from the prespecified rates.","",
  "Simulation uncertainty is reported following [Morris, White, and Crowther (2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6492164/).")
if(file.exists(file.path(out,"WEAK_PROXY_CORRECTION.md"))) {
  report<-append(report,c("","Correction: the Continuous weak-proxy data-generation helper was corrected to keep rewards unchanged, and all 20 affected datasets and nuisance diagnostics were regenerated. See [correction and provenance](WEAK_PROXY_CORRECTION.md)."),after=1L)
}
writeLines(report,file.path(out,"REPORT.md"))
saveRDS(list(nuisance=summary,initialization=within_summary,stopping=stop_summary,tuning=tune_summary,
             value=value_summary),file.path(out,"summaries.rds"))
print(main_table);print(stress_table);print(value_table);print(stop_table)
cat("Report written:",file.path(out,"REPORT.md"),"\n")
