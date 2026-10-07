#!/usr/bin/env Rscript

# Is the analysis cohort (the first 500 stay IDs) representative of the 3,708
# eligible stays? Compares it with the full eligible cohort and with 1,000 random
# 500-stay subsamples, on patient-level and bloc-level summaries.
#
# Usage from Code/MIMIC:
#   Rscript study_cohort.R          (~1 min, mostly reading the CSV)

script_path <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])
script_dir <- if (is.na(script_path)) normalizePath(getwd()) else dirname(normalizePath(script_path))
source(file.path(script_dir, "MIMIC.R"))

raw <- mimic_read_csv(file.path(script_dir, "sepsis_processed_state_action.csv"))
cnt <- table(raw$icustayid)
eligible <- sort(as.numeric(names(cnt[cnt == 20])))
raw <- raw[raw$icustayid %in% eligible, ]
first500 <- eligible[1:500]                      # what run_mimic_lure(max_stays = 500) keeps

stay <- aggregate(cbind(age, gender, elixhauser, mortality_90d, died_in_hosp) ~ icustayid,
                  data = raw, FUN = function(x) x[1])
bloc <- aggregate(cbind(SOFA = SOFA, iv1 = as.numeric(iv_input >= 1), iv3 = as.numeric(iv_input >= 3)) ~ icustayid,
                  data = raw, FUN = mean)
per <- merge(stay, bloc, by = "icustayid")
vars <- c(age = "age (years)", gender = "share with gender = 1", elixhauser = "Elixhauser",
          mortality_90d = "90-day mortality", died_in_hosp = "in-hospital mortality",
          SOFA = "mean SOFA per bloc", iv1 = "blocs with any IV (k >= 1)", iv3 = "blocs with IV >= 150 mL/4h (k >= 3)")

set.seed(2026)
rand <- replicate(1000, colMeans(per[per$icustayid %in% sample(eligible, 500), names(vars)]))

cat(sprintf("Eligible stays (exactly 20 blocs): %d.  First 500 IDs span %.0f-%.0f; all eligible IDs span %.0f-%.0f.\n\n",
            length(eligible), min(first500), max(first500), min(eligible), max(eligible)))
cat(sprintf("%-38s %10s %10s %24s %10s\n", "", "first 500", "all 3,708", "random 500s (2.5-97.5%)", "percentile"))
for (v in names(vars)) {
  a <- mean(per[per$icustayid %in% first500, v]); b <- mean(per[[v]])
  q <- quantile(rand[v, ], c(0.025, 0.975))
  cat(sprintf("%-38s %10.3f %10.3f %11.3f - %-10.3f %9.0f%%\n", vars[[v]], a, b, q[1], q[2],
              100 * mean(rand[v, ] <= a)))
}
