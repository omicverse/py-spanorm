#!/usr/bin/env Rscript
# Run R SpaNorm on real Visium SP1 data.

suppressMessages({
  library(SpaNorm)
  library(SpatialExperiment)
  library(Matrix)
})

cat("Loading Visium SP1 data...\n")

# Read 10x format
counts <- Matrix::readMM(gzfile("D:/桌面/myproject/data/GSE211956_RAW/GSM6506110_SP1_matrix.mtx.gz"))
features <- read.delim("D:/桌面/myproject/data/GSE211956_RAW/GSM6506110_SP1_features.tsv.gz", header = FALSE)
barcodes <- read.delim("D:/桌面/myproject/data/GSE211956_RAW/GSM6506110_SP1_barcodes.tsv.gz", header = FALSE)

rownames(counts) <- features$V2
colnames(counts) <- barcodes$V1

# Load spatial coords
spatial <- read.csv(unz("D:/桌面/myproject/data/GSE211956_RAW/GSM6506110_SP1_spatial.zip", "spatial/tissue_positions_list.csv"), header = FALSE)

# Match barcodes
coords <- matrix(0, ncol(counts), 2)
for (i in seq_len(ncol(counts))) {
  bc <- colnames(counts)[i]
  idx <- which(spatial$V1 == bc)
  if (length(idx) > 0) {
    coords[i, ] <- as.numeric(spatial[idx, c(5, 6)])
  }
}

# Filter valid spots
valid <- coords[, 1] > 0
counts <- counts[, valid]
coords <- coords[valid, ]

cat(sprintf("Data: %d genes x %d spots\n", nrow(counts), ncol(counts)))

# Filter lowly expressed genes
gene_expr <- Matrix::rowMeans(counts > 0)
keep <- gene_expr >= 0.1
counts <- counts[keep, ]
cat(sprintf("After filter: %d genes\n", nrow(counts)))

# Select top 300 variable genes
gene_var <- matrixStats::rowVars(as.matrix(counts))
top_idx <- order(gene_var, decreasing = TRUE)[1:300]
counts <- counts[top_idx, ]
cat(sprintf("Selected top 300 genes\n"))

# Create SpatialExperiment
spe <- SpatialExperiment::SpatialExperiment(
  assays = list(counts = counts),
  spatialCoords = coords
)

# Compute size factors
ls <- colSums(as.matrix(counts))
ls <- ls / mean(ls)
SingleCellExperiment::sizeFactors(spe) <- ls

# Run SpaNorm
cat("Running SpaNorm...\n")
t_start <- proc.time()
spe <- SpaNorm(spe, sample.p = 0.25, df.tps = 3, tol = 1e-2, adj.method = "meanbio", verbose = TRUE)
t_r <- (proc.time() - t_start)["elapsed"]

# Save results
lc <- as.matrix(logcounts(spe))
cat(sprintf("R time: %.3f seconds\n", t_r))
cat(sprintf("Logcounts range: [%.4f, %.4f]\n", min(lc), max(lc)))
cat(sprintf("Logcounts mean: %.4f\n", mean(lc)))

write.csv(as.data.frame(lc), "D:/test/SpaNorm-master/py-spanorm/data/visium_r_logcounts.csv", row.names = FALSE)
writeLines(as.character(t_r), "D:/test/SpaNorm-master/py-spanorm/data/visium_r_time.txt")
cat("Done\n")
