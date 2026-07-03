#!/usr/bin/env Rscript
# Usage:
#   Rscript plot_training.R [path/to/epochs.csv]
# If no argument is given, uses the latest run_* directory under build/mdl/exp03/.

script_dir <- "//wsl.localhost/Ubuntu-24.04/home/kinoko/GIT/eaglys/homomorphic-dsp/exp03-mnist/src/r"
#script_dir <- dirname(normalizePath(sys.frame(1)$ofile, mustWork = FALSE))
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

csv_path = paste(repo_dir, "build\\mdl\\exp03\\run_20260625-1412\\epochs.csv", sep="\\")

"/home/kinoko/GIT/eaglys/homomorphic-dsp/build/mdl/exp03/run_20260625-1412/epochs.csv"
csv_path =  "//wsl.localhost/Ubuntu-24.04/home/kinoko/GIT/eaglys/homomorphic-dsp/build/mdl/exp03/run_20260625-1412/epochs.csv"


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

dev.off()
# Loss
plot(df$epoch, df$train_loss, type = "l", col = "steelblue",
     xlab = "Epoch", ylab = "Loss", main = "Training Loss")

# Accuracy
plot(df$epoch, df$train_acc, 
     type = "l", col = "steelblue",
     ylim = c(0.5, 1), 
     xlab = "Epoch", ylab = "Accuracy", main = "Accuracy")


lines(df$epoch, df$dev_acc, col = "tomato")


################################################################################
### Dev vs Train ACC
################################################################################
{
  plot(df$train_acc, df$dev_acc,
       pch=20, cex=0.5, col="steelblue",
       xlim=c(0.75, 1), 
       ylim=c(0.75, 1), 
       xlab="Train ACC", ylab="Dev ACC")
  abline(h=seq(0, 1, 0.05), col="gray", lty=2)
  abline(h=seq(0, 1, 0.1), col="darkgray")
}

################################################################################
### Dev vs Train ACC
################################################################################

{
  plot(df$epoch, 1-df$train_acc, 
       type = "l", col = "steelblue",
       ylim = c(0.0, .25), 
       xlab = "Epoch", ylab = "Error", main = "Train vs Dev. Set Error ")
  abline(h=seq(0, 1, 0.05), col="lightgray", lty=2)
  abline(h=seq(0, 1, 0.1), col="gray")
  lines(df$epoch, 1-df$dev_acc, col = "tomato", lwd=2)
  lines(df$epoch, 1-df$train_acc, col= "steelblue", lwd=2)
}




################################################################################


legend("bottomright", legend = c("train", "dev"),
       col = c("steelblue", "tomato"), lty = 1, bty = "n")

dev.off()
message("Saved -> ", out_png)
