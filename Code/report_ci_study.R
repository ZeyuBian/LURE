# Sourced by run_ci_study.R report, after all 200 replications and references.
rows <- list(); diagnostics <- list()
for (setting in settings) for (scenario in scenarios) for (rep in 1:20) {
  path <- file.path(out,"replications",setting,scenario,sprintf("rep_%03d.rds",rep))
  stopifnot(file.exists(path))
  x <- readRDS(path)
  key <- paste(setting,if(scenario=="weak_proxy") "weak_proxy" else "baseline",sep="_")
  reference <- readRDS(file.path(out,"references",paste0(key,".rds")))
  z <- x$result
  finite <- !is.null(z) && all(is.finite(unlist(z[c("V_hat","se","ci_lo","ci_hi")])))
  if (is.null(z)) z <- list(V_hat=NA_real_,se=NA_real_,ci_lo=NA_real_,ci_hi=NA_real_)
  r <- data.frame(setting=setting,scenario=scenario,rep=rep,N=x$spec$N,T=x$spec$TT,
    tau=x$spec$tau,error_model=x$spec$mode,data_seed=x$data_seed,value_seed=x$value_seed,
    truth=reference$value,truth_mcse=reference$mcse,estimate=z$V_hat,se=z$se,
    ci_lo=z$ci_lo,ci_hi=z$ci_hi,width=z$ci_hi-z$ci_lo,finite=finite,
    covered=finite && z$ci_lo<=reference$value && z$ci_hi>=reference$value,
    covers_reference_band=finite && z$ci_lo<=reference$value-1.96*reference$mcse && z$ci_hi>=reference$value+1.96*reference$mcse,
    intersects_reference_band=finite && z$ci_lo<=reference$value+1.96*reference$mcse && z$ci_hi>=reference$value-1.96*reference$mcse,
    all_selected_em_converged=if(is.null(x$diagnostics)) NA else all(x$diagnostics$converged[x$diagnostics$selected]),
    max_selected_em_iterations=if(is.null(x$diagnostics)) NA else max(x$diagnostics$n_iter[x$diagnostics$selected]),
    bridge_index=if(is.null(z$bridge_index)) NA_integer_ else z$bridge_index,
    warning_count=x$warning_count,error=if(is.null(x$error)) "" else x$error,elapsed=x$elapsed)
  rows[[length(rows)+1L]] <- r
  if (!is.null(x$diagnostics)) diagnostics[[length(diagnostics)+1L]] <- cbind(r[,c("setting","scenario","rep")],x$diagnostics)
}
records <- do.call(rbind,rows)
stopifnot(nrow(records)==200L,!anyDuplicated(records[,c("setting","scenario","rep")]),
  all(records$width[records$finite]>=0),
  max(abs(records$ci_lo[records$finite]-(records$estimate[records$finite]-1.96*records$se[records$finite])))<1e-10,
  max(abs(records$ci_hi[records$finite]-(records$estimate[records$finite]+1.96*records$se[records$finite])))<1e-10)
write.csv(records,file.path(out,"individual_cis.csv"),row.names=FALSE)
write.csv(do.call(rbind,diagnostics),file.path(out,"em_diagnostics.csv"),row.names=FALSE)
summary_rows <- list()
for (setting in settings) for (scenario in scenarios) {
  d <- records[records$setting==setting & records$scenario==scenario,,drop=FALSE]
  stopifnot(nrow(d)==20L,length(unique(d$setting))==1L,length(unique(d$scenario))==1L)
  valid <- d[d$finite,]
  k <- sum(d$covered)
  coverage_ci <- binom.test(k,nrow(d))$conf.int
  summary_rows[[length(summary_rows)+1L]] <- data.frame(setting=setting,scenario=scenario,
    n_rep=nrow(d),n_finite=sum(d$finite),n_covered=k,coverage=k/nrow(d),
    coverage_binomial_ci_lo=coverage_ci[1],coverage_binomial_ci_hi=coverage_ci[2],
    truth=d$truth[1],truth_mcse=d$truth_mcse[1],mean_estimate=mean(valid$estimate),
    bias=mean(valid$estimate-valid$truth),rmse=sqrt(mean((valid$estimate-valid$truth)^2)),
    empirical_sd=sd(valid$estimate),mean_se=mean(valid$se),median_se=median(valid$se),
    mean_width=mean(valid$width),median_width=median(valid$width),max_width=max(valid$width),
    n_selected_em_converged=sum(d$all_selected_em_converged,na.rm=TRUE),
    coverage_count_min_in_reference_band=sum(d$covers_reference_band),
    coverage_count_max_in_reference_band=sum(d$intersects_reference_band))
}
summaries <- do.call(rbind,summary_rows)
stopifnot(sum(summaries$n_rep)==nrow(records),all(summaries$n_covered<=20L),
  sum(summaries$n_covered)==sum(records$covered))
write.csv(summaries,file.path(out,"ci_summary.csv"),row.names=FALSE)

labels <- c(typical="10% recording error",hard="30% recording error",weak_anchor="45% recording error",
  weak_proxy="Weak transition proxy (30%)",state_dependent="State-dependent error (30%)")
records$condition <- factor(labels[records$scenario],levels=labels)
records$setting <- factor(records$setting,levels=settings)
records$covered_label <- factor(ifelse(records$covered,"Contains truth","Misses truth"),levels=c("Contains truth","Misses truth"))
records$panel <- factor(paste(records$setting,records$condition,sep="\n"),
  levels=unlist(lapply(settings,function(s) paste(s,labels,sep="\n"))))
suppressPackageStartupMessages(library(ggplot2))
dir.create(file.path(out,"figures"),showWarnings=FALSE)
ci_plot <- ggplot(records,aes(x=rep,y=estimate-truth,color=covered_label))+
  geom_hline(yintercept=0,linetype="dashed",linewidth=.4,color="grey35")+
  geom_linerange(aes(ymin=ci_lo-truth,ymax=ci_hi-truth),linewidth=.5)+
  geom_point(size=1.3)+facet_wrap(~panel,ncol=5,scales="free_y")+
  scale_color_manual(values=c("Contains truth"="#236C94","Misses truth"="#C54B3C"))+
  scale_x_continuous(breaks=c(1,5,10,15,20))+
  labs(title="95% influence-function confidence intervals: 20 replications per condition",
    subtitle="Current estimator; 50 trajectories × 50 steps; all finite intervals retained. Vertical scales vary by panel.",
    x="Replication",y="Estimate and CI minus target value",color=NULL,
    caption="Dashed line: target value. Tabular targets are exact; continuous targets use independent Monte Carlo references (see report).")+
  theme_minimal(base_size=10)+theme(legend.position="bottom",panel.grid.minor=element_blank(),
    strip.text=element_text(face="bold",size=10),plot.title=element_text(face="bold"))
ggsave(file.path(out,"figures/individual_cis.png"),ci_plot,width=15,height=7.5,dpi=180)
ggsave(file.path(out,"figures/individual_cis.pdf"),ci_plot,width=15,height=7.5)
baseline <- subset(records,scenario %in% c("typical","hard"))
baseline$panel <- droplevels(baseline$panel)
baseline_plot <- ci_plot
baseline_plot$data <- baseline
baseline_plot <- baseline_plot + facet_wrap(~panel,ncol=2,scales="free_y")+
  labs(title="95% confidence intervals in the two main conditions")
ggsave(file.path(out,"figures/main_condition_cis.png"),baseline_plot,width=10,height=7,dpi=180)
width_plot <- ggplot(records,aes(x=condition,y=width,color=setting))+
  geom_boxplot(position=position_dodge(width=.8),outlier.shape=NA,width=.65)+
  geom_point(position=position_jitterdodge(jitter.width=.15,dodge.width=.8,seed=1),size=1.4,alpha=.7)+
  scale_y_log10()+scale_color_manual(values=c(Tabular="#A65A2B",Continuous="#236C94"))+
  labs(title="Interval widths across all 20 replications",x=NULL,y="95% CI width (log scale)",color=NULL)+
  theme_minimal(base_size=10)+theme(legend.position="bottom",axis.text.x=element_text(angle=15,hjust=1),
    panel.grid.minor=element_blank(),plot.title=element_text(face="bold"))
ggsave(file.path(out,"figures/ci_widths.png"),width_plot,width=11,height=5,dpi=180)

fmt <- function(x) formatC(x,format="f",digits=3)
table_lines <- function(d) c(
  "| Setting | Condition | Coverage | Mean width | Median width | Bias | Empirical SD | Mean IF SE | EM converged |",
  "|---|---|---:|---:|---:|---:|---:|---:|---:|",
  vapply(seq_len(nrow(d)),function(i) {
    r<-d[i,]
    sprintf("| %s | %s | %d/20 (%d%%) | %s | %s | %s | %s | %s | %d/20 |",
      r$setting,labels[r$scenario],r$n_covered,round(100*r$coverage),fmt(r$mean_width),
      fmt(r$median_width),fmt(r$bias),fmt(r$empirical_sd),fmt(r$mean_se),r$n_selected_em_converged)
  },character(1)))
reference_lines <- vapply(c("Tabular_baseline","Tabular_weak_proxy","Continuous_baseline","Continuous_weak_proxy"),function(key) {
  r <- readRDS(file.path(out,"references",paste0(key,".rds")))
  sprintf("- %s: %.6f (reference MC SE %.6f%s).",gsub("_"," ",key),r$value,r$mcse,
    if(is.na(r$n)) "" else paste0("; ",format(r$n,big.mark=",",scientific=FALSE,trim=TRUE)," paths"))
},character(1))
report <- c("# Current-estimator confidence intervals: 20 replications", "",
  "These are the production MR estimator's nominal 95% influence-function intervals, estimate ± 1.96 × SE. The study uses 20 independently generated data sets per condition, in each of Tabular and Continuous: 200 data sets total. Each data set has 50 trajectories and 50 steps, with discount 0.7. The same data sets as the convergence study are reused; no additional Gym experiments are included.","",
  "The estimator has not been repaired or tuned for this report. All completed estimates are retained, including iteration-cap fits and extreme intervals. Coverage is the fraction of the 20 intervals containing the reference target, not a confidence interval for the average estimate.","",
  "During CI validation, an error was found in the earlier Continuous weak-proxy data-generation helper: replacing every occurrence of (1/2) also changed a reward coefficient. The helper was corrected to change only transition terms. All 20 affected data sets and their eight-start nuisance diagnostics were regenerated with the original seeds; this CI report uses those corrected data. The earlier affected files are archived under the convergence study's superseded_reward_coefficient_bug folder. Other conditions and the production estimator files were unchanged.","",
  table_lines(summaries),"",
  sprintf("All %d/%d replications returned finite estimates and intervals.",sum(records$finite),nrow(records)),"",
  "EM converged means every selected nuisance fit in that replication met its production stopping tolerance. Tabular has one full-data fit; Continuous has two training-fold fits, each selected by likelihood from the original two starts. Convergence flags describe the EM update criterion only.","",
  "## Target values","",reference_lines, "",
  "Recording errors alone do not change the target value. The weak transition proxy changes the actual transition law, so it receives its own target. Continuous references use independent target-policy trajectories with a 100-step discounted horizon (0.7^100 ≈ 3.2e-16); they are computed once per transition model, separately from the 20 estimator replications. Each initially used 1,000,000 paths. One main-condition CI endpoint lay within the baseline reference's initial Monte Carlo interval, so that shared reference was refined to a fixed total of 10,000,000 paths. The weak-proxy reference remains at 1,000,000. The baseline and weak-proxy rollout formulas were checked against scalar production rollouts for five seeds each.","",
  "The CSV includes a coverage-count range allowing the reference target to vary over its Monte Carlo estimate ± 1.96 MC SE. This separates reference-value uncertainty from uncertainty due to only 20 estimator replications.","",
  "## Reproduction and limits","",
  "- The existing Tabular default-start nuisance fits are reused. Their source hashes match the current methods, and the complete value/SE/CI outputs were checked against fresh production fits in all five conditions.",
  "- Continuous uses the original two-start EM wrapper and original estimator, with fresh deterministic seeds for folds, restarts, and integration draws. Diagnostic instrumentation was checked to give bitwise identical estimator outputs in the hard and weak-proxy conditions.",
  "- Data seeds are those from the convergence study. Value seeds are 43,000,000 + 100,000 × I(Continuous) + 100 × replication. Settings are paired by replication across error conditions; they are not 100 independent replications of one condition.",
  "- Both production SEs use sqrt(var(IF)/(N×T)). Tabular uses the full data for fitting and correction; Continuous cross-fits by trajectory. These are the existing formulas, with no bootstrap or variance-formula change.",
  "- With 20 replications, coverage moves in 5 percentage-point increments. Exact binomial 95% intervals for the estimated coverage are included in ci_summary.csv. These results are a small simulation check, not a validation of asymptotic coverage.",
  "- For extreme intervals, interpret coverage alongside median/mean/maximum width, empirical estimate SD, and bias. No trimming or winsorization is applied in this report beyond the estimator's pre-existing clipping.","",
  "Run from Code:","","```sh","Rscript run_ci_study.R validate","Rscript run_ci_study.R reference",
  "Rscript run_ci_study.R run 2","Rscript refine_ci_reference.R","Rscript run_ci_study.R report","```","",
  "## Files","",
  "- ci_summary.csv: coverage, exact binomial uncertainty, interval widths, bias/RMSE, empirical SD, mean SE, and convergence counts.",
  "- individual_cis.csv: all 200 individual estimates, SEs, bounds, reference values, seeds, and diagnostics.",
  "- em_diagnostics.csv: selected/unselected restart iteration diagnostics.",
  "- replications/: individual reproducible output records.",
  "- references/: exact or Monte Carlo target records, including continuous simulated returns.",
  "- validation.txt and source_snapshot/: equivalence checks and unchanged production code snapshots.","",
  "![Individual intervals](figures/individual_cis.png)","","![Interval widths](figures/ci_widths.png)")
writeLines(report,file.path(out,"REPORT.md"))
file.copy("report_ci_study.R",file.path(out,"source_snapshot/report_ci_study.R"),overwrite=TRUE)
atomic_save(list(records=records,summary=summaries,source_md5=tools::md5sum(c(sources,"run_ci_study.R","report_ci_study.R","refine_ci_reference.R"))),file.path(out,"summaries.rds"))
print(summaries[,c("setting","scenario","n_covered","mean_width","median_width","bias","empirical_sd","mean_se","n_selected_em_converged")],row.names=FALSE)
cat("Reference uncertainty coverage bounds:\n")
print(summaries[,c("setting","scenario","coverage_count_min_in_reference_band","coverage_count_max_in_reference_band")],row.names=FALSE)
