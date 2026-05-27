#!/usr/bin/env Rscript
# Run R SpaNorm on Visium SP1 and save model parameters.

suppressMessages({
  library(SpaNorm)
  library(SpatialExperiment)
  library(Matrix)
})

cat("Loading data...\n")
counts <- Matrix::readMM(gzfile("D:/桌面/myproject/data/GSE211956_RAW/GSM6506110_SP1_matrix.mtx.gz"))
features <- read.delim("D:/桌面/myproject/data/GSE211956_RAW/GSM6506110_SP1_features.tsv.gz", header = FALSE)
barcodes <- read.delim("D:/桌面/myproject/data/GSE211956_RAW/GSM6506110_SP1_barcodes.tsv.gz", header = FALSE)
rownames(counts) <- features$V2
colnames(counts) <- barcodes$V1

spatial <- read.csv(unz("D:/桌面/myproject/data/GSE211956_RAW/GSM6506110_SP1_spatial.zip", "spatial/tissue_positions_list.csv"), header = FALSE)
coords <- matrix(0, ncol(counts), 2)
for (i in seq_len(ncol(counts))) {
  bc <- colnames(counts)[i]
  idx <- which(spatial$V1 == bc)
  if (length(idx) > 0) coords[i, ] <- as.numeric(spatial[idx, c(5, 6)])
}

valid <- coords[, 1] > 0
counts <- counts[, valid]
coords <- coords[valid, ]
cat(sprintf("Loaded: %d genes x %d spots\n", nrow(counts), ncol(counts)))

# Filter
gene_expr <- Matrix::rowMeans(counts > 0)
counts <- counts[gene_expr >= 0.1, ]
cat(sprintf("After filter: %d genes\n", nrow(counts)))

# Top 100 genes (smaller for stability)
gene_var <- matrixStats::rowVars(as.matrix(counts))
counts <- counts[order(gene_var, decreasing = TRUE)[1:100], ]
cat(sprintf("Selected: %d genes\n", nrow(counts)))

# Create SPE
spe <- SpatialExperiment(assays = list(counts = counts), spatialCoords = coords)
ls_val <- colSums(as.matrix(counts))
ls_val <- ls_val / mean(ls_val)
SingleCellExperiment::sizeFactors(spe) <- ls_val

# Run SpaNorm
cat("Running SpaNorm (df.tps=2)...\n")
t_start <- proc.time()
spe <- SpaNorm(spe, sample.p = 0.25, df.tps = 2, tol = 1e-2, adj.method = "meanbio", verbose = FALSE)
t_r <- (proc.time() - t_start)["elapsed"]

fit <- S4Vectors::metadata(spe)$SpaNorm
lc <- as.matrix(logcounts(spe))

cat(sprintf("R time: %.3f s\n", t_r))
cat(sprintf("logcounts: [%.4f, %.4f]\n", min(lc), max(lc)))
cat(sprintf("W: %d x %d\n", nrow(fit$W), ncol(fit$W)))

# Save
out_dir <- "D:/test/SpaNorm-master/py-spanorm/data"
write.csv(as.data.frame(lc), file.path(out_dir, "visium100_r_lc.csv"), row.names = FALSE)
write.csv(as.data.frame(fit$W), file.path(out_dir, "visium100_r_W.csv"), row.names = FALSE)
write.csv(data.frame(gmean = fit$gmean), file.path(out_dir, "visium100_r_gmean.csv"), row.names = FALSE)
write.csv(as.data.frame(fit$alpha), file.path(out_dir, "visium100_r_alpha.csv"), row.names = FALSE)
write.csv(data.frame(psi = fit$psi), file.path(out_dir, "visium100_r_psi.csv"), row.names = FALSE)
writeLines(as.character(t_r), file.path(out_dir, "visium100_r_time.txt"))
cat("Saved\n")
