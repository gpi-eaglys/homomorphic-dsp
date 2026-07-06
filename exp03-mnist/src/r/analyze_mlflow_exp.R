#!/usr/bin/env Rscript
# Usage:
#   Rscript analyze_runs.R [path/to/exp03_runs.csv]
# If no argument is given, uses build/exp03_runs.csv under the repo root.

# install.packages("rstudioapi")



script_dir <- dirname(normalizePath(rstudioapi::getActiveDocumentContext()$path, mustWork = FALSE))
#script_dir <- dirname(normalizePath(sys.frame(1)$ofile, mustWork = FALSE))
repo_dir   <- normalizePath(file.path(script_dir, "../../.."), mustWork = FALSE)
csv_path <- file.path(repo_dir, "build/exp03_runs.csv")


if (!file.exists(csv_path)) stop("File not found: ", csv_path)

df <- read.csv(csv_path, strip.white = TRUE)


###################################################################
### Handle duplicates
###################################################################
params <- c("lr", "dropout", "activation", "batch", "layers")

xtabs(~df$dropout + df$activation)

df = df[df$dropout != .5,]

hist(df$test_acc, breaks=100)


dup_counts <- aggregate(run_id ~ ., data = df[c(params, "run_id")], FUN = length)
names(dup_counts)[names(dup_counts) == "run_id"] <- "n_runs"
dup_counts <- dup_counts[dup_counts$n_runs > 1, ]
dup_counts <- dup_counts[order(-dup_counts$n_runs), ]

dup_counts

dup_rows <- merge(dup_counts[params], df, by = params)
dup_rows <- dup_rows[order(dup_rows[[params[1]]], dup_rows[[params[2]]], dup_rows[[params[3]]],
                           dup_rows[[params[4]]], dup_rows[[params[5]]], -dup_rows$test_acc), ]

dup_rows[c(params, "run_id", "test_acc", "train_acc")]


###################################################################
### Best test_acc by activation x layers
###################################################################
tbl <- 100*tapply(df$test_acc, list(df$layers, df$activation), max)
tbl <- tbl[order(tbl[, 2], decreasing=T),]

par(mar = c(5, 14, 4, 4))
image(1:ncol(tbl), 1:nrow(tbl), t(tbl),
      axes = FALSE, xlab = "", ylab = "", zlim = c(90, 99),
      col = colorRampPalette(c("white", "steelblue"))(20))
axis(1, at = 1:ncol(tbl), labels = colnames(tbl))
axis(2, at = 1:nrow(tbl), labels = rownames(tbl), las = 2)
title(main = "MNIST best ACC% on dev set")
for (i in 1:nrow(tbl)) {
  for (j in 1:ncol(tbl)) {
    if (!is.na(tbl[i, j])) {
      text(j, i, sprintf("%.1f", tbl[i, j]))
    }
  }
}


###################################################################
### Best test_acc by activation x layers
###################################################################
df.quad = df[df$activation == "Quadratic",]
df.quad$layers <- reorder(df.quad$layers, df.quad$test_acc, FUN = median)  # swap FUN = mean / max as needed
par(mar = c(14, 4, 4, 2))
boxplot(df.quad$test_acc ~ df.quad$layers, las = 2, col="lightblue")


df.quad <- df.quad[df.quad$test_acc > 0.8,]

boxplot(df.quad$test_acc ~ df.quad$dropout, col="lightblue", ylim=c(0.9, 1.0))
abline(h=seq(0, 1, 0.05), col="gray")
dev.off()


###################################################################
### Best test_acc by activation x layers
###################################################################



