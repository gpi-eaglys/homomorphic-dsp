RESULTS_DIR <- file.path("..", "..", "..", "build", "results")

read_activations <- function(mode, feat) {
  fpath <- file.path(RESULTS_DIR, mode, paste0(feat, ".txt"))
  df <- read.table(fpath, header = FALSE, sep = "", stringsAsFactors = FALSE)
  n_classes <- ncol(df) - 1
  colnames(df) <- c("stem", paste0("class_", seq_len(n_classes) - 1))
  df
}

compare <- function(feat) {
  pt  <- read_activations("plaintext", feat)
  fhe <- read_activations("fhe",       feat)

  merged <- merge(pt, fhe, by = "stem", suffixes = c("_pt", "_fhe"))

  n_classes <- ncol(pt) - 1
  class_cols <- paste0("class_", seq_len(n_classes) - 1)
  diffs <- as.vector(as.matrix(merged[, paste0(class_cols, "_fhe")]) -
                     as.matrix(merged[, paste0(class_cols, "_pt")]))

  cat(sprintf("\n--- %s ---\n", feat))
  cat(sprintf("  samples  : %d\n",   nrow(merged)))
  cat(sprintf("  classes  : %d\n",   n_classes))
  cat(sprintf("  max |err|: %.3e\n", max(abs(diffs))))
  cat(sprintf("  mean|err|: %.3e\n", mean(abs(diffs))))
  cat(sprintf("  std  err : %.3e\n", sd(diffs)))

  diffs
}

feats <- c("esc10-mfb", "esc10-mfcc", "esc50-mfb", "esc50-mfcc")

for (feat in feats) {
  diffs <- compare(feat)
  outfile <- paste0(feat, "_activation_diff.png")
  png(outfile, width = 1400, height = 800, res = 150)
  par(mar = c(4, 4, 3, 1))
  hist(diffs, breaks = 80, main = paste("", feat),
       xlab = "Activation Diff: Enc-Plain ", col = "steelblue", border = "white")
  dev.off()
  cat(sprintf("Saved: %s\n", outfile))
}
