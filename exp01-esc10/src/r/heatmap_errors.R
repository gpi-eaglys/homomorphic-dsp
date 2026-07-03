RESULTS_DIR <- file.path("..", "..", "..", "build", "results")
ESC50_META  <- file.path("..", "..", "..", "assets", "esc-50",
                         "ESC-50-master", "meta", "esc50.csv")

meta          <- read.csv(ESC50_META, stringsAsFactors = FALSE)
meta$stem     <- sub("\\.wav$", "", meta$filename)
stem_to_cat   <- setNames(meta$category, meta$stem)
all_classes   <- sort(unique(meta$category))

read_results <- function(feat, mode = "plaintext") {
  fpath <- file.path(RESULTS_DIR, mode, paste0(feat, ".txt"))
  df    <- read.table(fpath, header = FALSE, sep = "", stringsAsFactors = FALSE)
  colnames(df) <- c("stem", paste0("c", seq_len(ncol(df) - 1)))
  df
}

build_confusion <- function(feat, classes, mode = "plaintext") {
  df     <- read_results(feat, mode)
  n      <- length(classes)
  cm     <- matrix(0L, n, n, dimnames = list(true = classes, pred = classes))
  s_cols <- paste0("c", seq_len(n))
  for (i in seq_len(nrow(df))) {
    true_cat <- stem_to_cat[df$stem[i]]
    if (is.na(true_cat) || !(true_cat %in% classes)) next
    pred_cat <- classes[which.max(as.numeric(df[i, s_cols]))]
    cm[true_cat, pred_cat] <- cm[true_cat, pred_cat] + 1L
  }
  cm
}

plot_heatmap <- function(cm, title, outfile) {
  # keep only rows that have at least one error
  error_rows <- rownames(cm)[diag(cm) < rowSums(cm)]
  if (length(error_rows) == 0) {
    cat(sprintf("No errors in %s — skipping\n", title))
    return(invisible(NULL))
  }
  # columns that received a misprediction from error rows
  error_cols <- colnames(cm)[colSums(cm[error_rows, , drop = FALSE]) > 0]
  # union: any category involved as true or predicted goes on both axes
  involved <- union(error_rows, error_cols)
  sub <- cm[involved, involved, drop = FALSE]

  nr <- nrow(sub)
  nc <- ncol(sub)

  # colour scale: white → steel blue → dark blue
  pal <- colorRampPalette(c("white", "#4682B4", "#00008B"))(64)

  cell_px <- 80
  pw <- max(600, nc * cell_px + 280)
  ph <- max(500, nr * cell_px + 220)
  png(outfile, width = pw, height = ph)
  par(mar = c(11, 14, 4, 2))

  # image() draws x left-right, y bottom-top; we flip rows so first is at top
  m_plot <- sub[nr:1, , drop = FALSE]          # flip row order
  image(seq_len(nc), seq_len(nr), t(m_plot),   # t(): nc x nr for image()
        col  = pal,
        xaxt = "n", yaxt = "n",
        xlab = "", ylab = "",
        main = title)

  # grid lines between cells
  abline(v = seq(0.5, nc + 0.5), col = "grey60", lwd = 0.8)
  abline(h = seq(0.5, nr + 0.5), col = "grey60", lwd = 0.8)

  axis(1, at = seq_len(nc), labels = colnames(sub),    las = 2, cex.axis = 1.666)
  axis(2, at = seq_len(nr), labels = rownames(m_plot), las = 1, cex.axis = 1.666)
  mtext("Predicted", side = 1, line = 9,  cex = 1.1, font = 2)
  mtext("True",      side = 2, line = 12, cex = 1.1, font = 2)

  # cell counts — white text on dark cells, black text on light cells
  threshold <- max(sub) * 0.5
  for (ri in seq_len(nr)) {
    true_cls <- rownames(m_plot)[ri]
    for (ci in seq_len(nc)) {
      v <- sub[true_cls, colnames(sub)[ci]]
      if (v > 0) {
        txt_col <- if (v >= threshold) "white" else "black"
        text(ci, ri, v, cex = 2, col = txt_col,
             font = if (colnames(sub)[ci] == true_cls) 2L else 1L)
      }
    }
  }

  dev.off()
  cat(sprintf("Saved: %s\n", outfile))
}

for (feat in c("esc50-mfb", "esc50-mfcc")) {
  cm <- build_confusion(feat, all_classes)
  plot_heatmap(cm, paste("Confusion — errors only:", feat),
               paste0(feat, "_errors_heatmap.png"))
}
