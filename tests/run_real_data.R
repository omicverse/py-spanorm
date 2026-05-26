#!/usr/bin/env Rscript
# Run SpaNorm on real HumanDLPFC data and save results.

suppressMessages({
  library(SpaNorm)
  library(SpatialExperiment)
  library(SingleCellExperiment)
  library(SummarizedExperiment)
})

cat("Loading HumanDLPFC...\n")
data(HumanDLPFC)
cat(sprintf("Full data: %d genes x %d cells\n", nrow(HumanDLPFC), ncol(HumanDLPFC)))

# Filter genes first, then subset
keep <- filterGenes(HumanDLPFC, prop = 0.1)
HumanDLPFC <- HumanDLPFC[keep, ]
cat(sprintf("After filtering: %d genes x %d cells\n", nrow(HumanDLPFC), ncol(HumanDLPFC)))

# Subset
set.seed(42)
gene_idx <- sample(nrow(HumanDLPFC), 50)
spe <- HumanDLPFC[gene_idx, 1:500]
cat(sprintf("Subset: %d genes x %d cells\n", nrow(spe), ncol(spe)))

# Compute size factors manually to avoid scran issues
ls <- colSums(as.matrix(assay(spe, "counts")))
ls <- ls / mean(ls)
SingleCellExperiment::sizeFactors(spe) <- ls
cat("Size factors computed\n")

# Save counts and coords
emat <- as.matrix(assay(spe, "counts"))
coords <- spatialCoords(spe)
write.csv(as.data.frame(emat), "D:/test/SpaNorm-master/py-spanorm/data/real_counts.csv", row.names = FALSE)
write.csv(as.data.frame(coords), "D:/test/SpaNorm-master/py-spanorm/data/real_coords.csv", row.names = FALSE)
cat("Saved counts and coords\n")

# Run SpaNorm
cat("Running SpaNorm...\n")
t_start <- proc.time()
spe <- SpaNorm(spe, sample.p = 0.25, df.tps = 2, tol = 1e-2, adj.method = "meanbio", verbose = TRUE)
t_r <- (proc.time() - t_start)["elapsed"]

# Save results
lc <- as.matrix(logcounts(spe))
write.csv(as.data.frame(lc), "D:/test/SpaNorm-master/py-spanorm/data/real_r_logcounts.csv", row.names = FALSE)
writeLines(as.character(t_r), "D:/test/SpaNorm-master/py-spanorm/data/real_r_time.txt")

# Save model parameters
fit <- S4Vectors::metadata(spe)$SpaNorm
write.csv(data.frame(gmean = fit$gmean), "D:/test/SpaNorm-master/py-spanorm/data/real_r_gmean.csv", row.names = FALSE)
write.csv(as.data.frame(fit$alpha), "D:/test/SpaNorm-master/py-spanorm/data/real_r_alpha.csv", row.names = FALSE)
write.csv(data.frame(psi = fit$psi), "D:/test/SpaNorm-master/py-spanorm/data/real_r_psi.csv", row.names = FALSE)
write.csv(as.data.frame(fit$W), "D:/test/SpaNorm-master/py-spanorm/data/real_r_W.csv", row.names = FALSE)
cat(sprintf("gmean[1:3]: %.6f %.6f %.6f\n", fit$gmean[1], fit$gmean[2], fit$gmean[3]))
cat(sprintf("alpha[1,1:3]: %.6f %.6f %.6f\n", fit$alpha[1,1], fit$alpha[1,2], fit$alpha[1,3]))
cat(sprintf("psi[1:3]: %.6f %.6f %.6f\n", fit$psi[1], fit$psi[2], fit$psi[3]))

cat(sprintf("R time: %.3f seconds\n", t_r))
cat(sprintf("logcounts range: [%.4f, %.4f]\n", min(lc), max(lc)))
cat("Done\n")
