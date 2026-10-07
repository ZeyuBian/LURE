suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(ggplot2))

report_state_dependent_example <- function(root) {
  config <- readRDS(file.path(root, "config.rds"))
  truth <- readRDS(file.path(root, "truth.rds"))
  calibration <- readRDS(file.path(root, "calibration.rds"))
  paths <- file.path(root, sprintf("rep_%03d.rds", seq_len(config$n_rep)))
  if (!all(file.exists(paths))) stop("Wait for all replications before reporting.")
  runs <- lapply(paths, readRDS)
  estimates <- bind_rows(lapply(runs, `[[`, "estimates"))
  intervals <- bind_rows(lapply(runs, `[[`, "intervals"))
  diagnostics <- bind_rows(lapply(runs, `[[`, "diagnostics"))
  warnings <- bind_rows(lapply(runs, `[[`, "warnings"))
  stopifnot(nrow(estimates) == config$n_rep * 8L * 7L,
    nrow(intervals) == config$n_rep * 8L,
    !anyDuplicated(estimates[c("rep","mode","rate","method")]),
    all(diagnostics$min_probability >= 0), all(diagnostics$max_probability < .5))
  for (z in runs) {
    for (rate in config$rates) {
      p <- z$diagnostics %>% filter(mode == "constant", .data$rate == .env$rate)
      stopifnot(nrow(p)==1L, abs(p$expected_rate-rate)<1e-12)
    }
  }
  summary <- estimates %>% group_by(mode, rate, method) %>% summarise(
    n = n(), n_finite = sum(is.finite(estimate)), mean = mean(estimate),
    bias = mean(estimate-V_true), rmse = sqrt(mean((estimate-V_true)^2)),
    empirical_sd = sd(estimate), .groups = "drop")
  intervals <- intervals %>% mutate(
    contains_target_mc_band = valid & lower <= V_true-qnorm(.975)*truth$mc_se &
      upper >= V_true+qnorm(.975)*truth$mc_se)
  coverage <- intervals %>% group_by(mode, rate) %>% summarise(
    n = n(), n_valid = sum(valid), n_covered = sum(covered), coverage = mean(covered),
    mean_se = mean(se), empirical_sd = sd(estimate), mean_width = mean(upper-lower),
    covers_entire_target_mc_band = sum(contains_target_mc_band), .groups = "drop") %>%
    rowwise() %>% mutate(coverage_lo = binom.test(n_covered,n)$conf.int[1],
                         coverage_hi = binom.test(n_covered,n)$conf.int[2]) %>% ungroup()
  error_rates <- diagnostics %>% group_by(mode, rate) %>% summarise(
    expected_rate = mean(expected_rate), actual_rate = mean(actual_rate),
    min_probability = min(min_probability), max_probability = max(max_probability),
    .groups = "drop")
  best <- summary %>% filter(!method %in% c("MR","DIRECT"), is.finite(rmse)) %>%
    group_by(mode, rate) %>% slice_min(rmse, n=1, with_ties=FALSE) %>% ungroup() %>%
    transmute(mode,rate,best_baseline=method,baseline_rmse=rmse)
  comparison <- summary %>% filter(method=="MR") %>%
    select(mode,rate,bias,rmse) %>% left_join(best,by=c("mode","rate")) %>%
    left_join(coverage,by=c("mode","rate")) %>% left_join(error_rates,by=c("mode","rate"))
  tables <- list(estimates=estimates, intervals=intervals, summary=summary, coverage=coverage,
                 error_rates=error_rates, comparison=comparison, diagnostics=diagnostics, warnings=warnings)
  for (nm in names(tables)) write.csv(tables[[nm]],file.path(root,paste0(nm,".csv")),row.names=FALSE)
  plot_data <- summary %>% filter(method!="DIRECT",is.finite(rmse),rmse>0) %>%
    mutate(method=ifelse(method=="MR","LURE",method),
           mode=ifelse(mode=="constant","Constant recording error","State-dependent recording error"))
  colors <- c(LURE="#006B9E",FQE="#59A14F",SIS="#B279A2",MIS="#E377C2",DRL="#E15759",LSTD="#F28E2B")
  p <- ggplot(plot_data,aes(rate,rmse,color=method,group=method)) + geom_line(linewidth=.8) +
    geom_point(size=2.2) + facet_wrap(~mode,nrow=1) + scale_y_log10() +
    scale_x_continuous(breaks=config$rates) + scale_color_manual(values=colors) +
    labs(title="CartPole: constant versus state-dependent recording errors",
      subtitle=sprintf("Updated Code/ pipeline; N=50, T=50; %d paired replications per rate",config$n_rep),
      x="Nominal error rate",y="RMSE (log scale)",color=NULL,
      caption="All finite estimates are included. Unavailable results, if any, are counted in the report.") +
    theme_minimal(base_size=12) + theme(legend.position="bottom",panel.grid.minor=element_blank(),
      plot.title=element_text(face="bold"),plot.caption=element_text(hjust=0))
  ggsave(file.path(root,"rmse_comparison.png"),p,width=11,height=5.5,dpi=180,bg="white")
  ci_plot <- intervals %>% filter(mode=="state_dependent",valid) %>%
    mutate(rate=factor(rate, levels=config$rates, labels=paste0("Error rate ",100*config$rates,"%")))
  q <- ggplot(ci_plot,aes(rep,estimate,color=covered)) +
    geom_hline(yintercept=truth$V_true,linetype="dashed",color="#444444") +
    geom_linerange(aes(ymin=lower,ymax=upper),linewidth=.7) + geom_point(size=2) +
    facet_wrap(~rate,nrow=1) + scale_x_continuous(breaks=unique(c(1L,ceiling(config$n_rep/2),config$n_rep))) +
    scale_color_manual(values=c(`FALSE`="#D1495B",`TRUE`="#006B9E")) +
    labs(title="LURE IF-based 95% confidence intervals",subtitle="State-dependent recording; dashed line is the Monte Carlo target",
      x="Replication",y="Policy value",color="Covers target") + theme_minimal(base_size=12) +
    theme(legend.position="bottom",panel.grid.minor=element_blank(),plot.title=element_text(face="bold"))
  ggsave(file.path(root,"if_intervals.png"),q,width=12,height=5,dpi=180,bg="white")
  fmt <- function(x,d=4) formatC(x,digits=d,format="f")
  report_rows <- vapply(seq_len(nrow(comparison)),function(i) {
    z <- comparison[i,]
    paste0("| ",z$mode," | ",fmt(z$rate,2)," | ",fmt(z$actual_rate,3)," | ",fmt(z$bias),
      " | ",fmt(z$rmse)," | ",z$best_baseline," (",fmt(z$baseline_rmse),") | ",
      z$n_covered,"/",z$n," | ",fmt(z$mean_se)," | ",fmt(z$empirical_sd)," |")
  },"")
  all_rows <- vapply(seq_len(nrow(summary)),function(i) {
    z<-summary[i,]
    paste0("| ",z$mode," | ",fmt(z$rate,2)," | ",ifelse(z$method=="MR","LURE",z$method),
      " | ",fmt(z$bias)," | ",fmt(z$rmse)," | ",z$n_finite,"/",z$n," |")
  },"")
  report <- c("# Updated-code CartPole example: state-dependent error", "",
    sprintf("%d paired replications per error rate and recording mechanism; N=50, T=50, gamma=0.7. LURE uses the updated code's influence-function CI, unchanged. No bootstrap was run.",config$n_rep),"",
    "## Main results", "",
    "| Recording | Nominal rate | Actual rate | LURE bias | LURE RMSE | Best baseline (RMSE) | IF coverage | Mean IF SE | Estimator SD |",
    "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",report_rows,"",
    "![RMSE comparison](rmse_comparison.png)","","![IF intervals](if_intervals.png)","",
    "## Fixed error mechanism", "",
    "p(flip | S) = tau + 0.75 min(tau, 0.5-tau) tanh((x-center)/scale). Only the current cart position enters this recording probability; true actions, dynamics, rewards, and next states are otherwise unchanged.","",
    sprintf("Independent calibration: center=%.8f, scale=%.8f (position IQR), mean score=%.3g, using 1000 behavior trajectories of length 50. The average probability on the calibration sample equals tau; the independent example's realized and expected rates fluctuate.",calibration$center,calibration$scale,calibration$mean_score),"",
    "The mechanism and strength were fixed before inspecting method performance. Constant and state-dependent cases share the same trajectories, recording uniforms, and fitting seeds. Parameters were not selected to improve coverage or outperform baselines.","",
    "## Target and implementation", "",
    sprintf("Target action 1 iff x > %g and theta < %g. This preserves the updated R policy and explicitly passes matching thresholds to Python; set LURE_CARTPOLE_TARGET_X and LURE_CARTPOLE_TARGET_THETA to select another matched policy.",config$target_x,config$target_theta),
    sprintf("Fresh Monte Carlo target = %.8f, MC SE = %.8f (%d trajectories, horizon %d, seed %d).",truth$V_true,truth$mc_se,truth$n,truth$horizon,truth$seed),"",
    "The Python-to-R loader was corrected to preserve JSON trajectory/time/coordinate ordering. The original loader reshaped row-major JSON directly into column-major R arrays. Tests verify identifiable coordinates and consecutive-state alignment.","",
    "The nuisance-fitting, bridge selection, clipping, regularization, and IF CI functions are the user's updated versions. The old initial-state convention (clean resets for estimation, noisy starts for rollouts) remains a limitation; this task does not silently change it.","",
    "## Limitations and reproducibility", "",
    "- Ten replications are a small diagnostic, not evidence that nominal 95% coverage has been achieved. Binomial reference ranges are in coverage.csv.",
    "- The target is estimated by Monte Carlo. intervals.csv also indicates whether an interval contains its entire approximate 95% MC reference band.",
    sprintf("- Unavailable point estimates: %d/%d; invalid LURE intervals: %d/%d; warning messages: %d. No failed fit or extreme estimate was discarded from the saved outputs.",sum(!is.finite(estimates$estimate)),nrow(estimates),sum(!intervals$valid),nrow(intervals),sum(diagnostics$warning_count)),
    "- Fresh rollout/reset seed ranges do not overlap across replications, calibration, or target MC. The same replication is paired across rates and mechanisms, so those comparisons are not independent.",
    "- Original saved datasets and res_cartpole.RData were not overwritten. No MountainCar simulations were run; Tabular, Continuous, and MIMIC files were not changed. The JSON alignment correction is in the shared Gym loader.",
    "- PLAN.md and config.rds record the design and seeds; calibration.rds and truth.rds record calibration and target uncertainty.",
    "- rep_*.rds contain all estimates, warnings, clean trajectories, and recording uniforms. source_snapshot/ and source_hashes.csv preserve the code used.","",
    "## All point-estimation results", "",
    "| Recording | Rate | Method | Bias | RMSE | Finite fits |",
    "| --- | ---: | --- | ---: | ---: | ---: |",all_rows,"")
  writeLines(report,file.path(root,"RESULTS.md"))
  print(as.data.frame(comparison))
  invisible(list(summary=summary,coverage=coverage,comparison=comparison))
}

if (sys.nframe()==0L) {
  args<-commandArgs(TRUE)
  if(length(args)!=1L) stop("Pass a completed example output folder.")
  report_state_dependent_example(args[1])
}
