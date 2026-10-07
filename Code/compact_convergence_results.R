args<-commandArgs(TRUE);stopifnot(length(args)==1L)
source("convergence_diagnostics.R")
compact<-function(x) {
  if(inherits(x,"lm")) return(diag_compact_raw(list(model=x))$model)
  if(is.list(x)&&!is.data.frame(x)) for(i in seq_along(x)) x[i]<-list(compact(x[[i]]))
  x
}
files<-unlist(lapply(c("pilot","main"),function(stage)
  list.files(file.path(args[1],stage),pattern="\\.rds$",recursive=TRUE,full.names=TRUE)))
saved<-0
for(file in files) {
  if(file.info(file)$size<1e7) next
  x<-readRDS(file);old_metrics<-x$metrics;y<-compact(x)
  stopifnot(identical(old_metrics,y$metrics))
  temp<-paste0(file,".compact");saveRDS(y,temp,compress="gzip")
  saved<-saved+file.info(file)$size-file.info(temp)$size
  stopifnot(file.rename(temp,file))
}
cat("Removed duplicated training arrays/call frames; saved",round(saved/1024^2),"MiB. Metrics unchanged.\n")
