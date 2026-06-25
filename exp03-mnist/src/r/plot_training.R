#!/usr/bin/env Rscript
# Usage:
#   Rscript plot_training.R [path/to/epochs.csv]
# If no argument is given, uses the latest run_* directory under build/mdl/exp03/.

script_dir <- dirname(normalizePath(sys.frame(1)$ofile, mustWork = FALSE))
repo_dir   <- normalizePath(file.path(script_dir, "../../.."), mustWork = FALSE)
mdl_root   <- file.path(repo_dir, "build/mdl/exp03")

args <- commandArgs(trailingOnly = TRUE)

if (length(args) >= 1) {
  csv_path <- args[1]
} else {
  run_dirs <- sort(list.dirs(mdl_root, recursive = FALSE, full.names = TRUE))
  run_dirs <- run_dirs[grepl("/run_", run_dirs)]
  if (length(run_dirs) == 0) stop("No run_* directories found under ", mdl_root)
  csv_path <- file.path(tail(run_dirs, 1), "epochs.csv")
  message("Using: ", csv_path)
}

if (!file.exists(csv_path)) stop("File not found: ", csv_path)

df <- read.csv(csv_path, strip.white = TRUE)

cat(sprintf("Epochs:         %d\n",   nrow(df)))
cat(sprintf("Best train_acc: %.5f  (epoch %d)\n", max(df$train_acc), df$epoch[which.max(df$train_acc)]))
cat(sprintf("Best dev_acc:   %.5f  (epoch %d)\n", max(df$dev_acc),   df$epoch[which.max(df$dev_acc)]))
cat(sprintf("Final train_acc: %.5f\n", tail(df$train_acc, 1)))
cat(sprintf("Final dev_acc:   %.5f\n", tail(df$dev_acc,   1)))

out_png <- file.path(dirname(csv_path), "training.png")
png(out_png, width = 1200, height = 800, res = 120)

par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))

# Loss
plot(df$epoch, df$train_loss, type = "l", col = "steelblue",
     xlab = "Epoch", ylab = "Loss", main = "Training Loss")

# Accuracy
plot(df$epoch, df$train_acc, type = "l", col = "steelblue",
     ylim = c(0, 1), xlab = "Epoch", ylab = "Accuracy", main = "Accuracy")
lines(df$epoch, df$dev_acc, col = "tomato")
legend("bottomright", legend = c("train", "dev"),
       col = c("steelblue", "tomato"), lty = 1, bty = "n")

dev.off()
message("Saved -> ", out_png)
