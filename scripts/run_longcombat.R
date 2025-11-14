#!/usr/bin/env Rscript
# run_longcombat.R
# Inputs:
#  - long_csv: rows = scans (subject_id, scanner_id, age_at_scan, sex, ROI_1, ROI_2, ...)
# Outputs:
#  - harmonized CSV
# Rscript scripts/run_longcombat.R \
#   --input derivatives/preproc/anat/roi_longitudinal.csv \
#   --output derivatives/harmonized/structural/roi_longcombat.csv \
#   --batch scanner_id \
#   --subj subject_id \
#   --time age_at_scan


library(optparse)
library(longCombat)  # if not available, install from CRAN / github
library(data.table)

option_list = list(
  make_option(c("-i","--input"), type="character"),
  make_option(c("-o","--output"), type="character"),
  make_option(c("--batch"), type="character", default="scanner_id"),
  make_option(c("--subj"), type="character", default="subject_id"),
  make_option(c("--time"), type="character", default="age_at_scan")
)
opt = parse_args(OptionParser(option_list=option_list))

df = fread(opt$input)
# build mod matrix (age and sex)
mod = model.matrix(~ df[[opt$time]] + df$sex)
features = names(df)[!(names(df) %in% c(opt$subj, opt$batch, opt$time, "sex"))]
X = t(as.matrix(df[, ..features])) # features x scans

res = longCombat(dat = X,
                 batch = df[[opt$batch]],
                 subj = df[[opt$subj]],
                 mod = mod,
                 verbose = TRUE)

harm = t(res$dat.combat)
df_out = cbind(df[, .(subject_id = get(opt$subj), scanner_id = get(opt$batch), age_at_scan = get(opt$time), sex = sex)], as.data.table(harm))
fwrite(df_out, opt$output)
cat("Wrote:", opt$output, "\n")
