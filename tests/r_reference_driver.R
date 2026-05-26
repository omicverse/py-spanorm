#!/usr/bin/env Rscript
# Reference runner for SpaNorm parity testing.
# Loads a fixture RDS file, runs SpaNorm, writes JSON output.
#
# Usage: Rscript r_reference_driver.R <fixture.rds> <output.json>

suppressMessages({
  library(SpaNorm)
  library(SpatialExperiment)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript r_reference_driver.R <fixture.rds> <output.json>")
}

fixture_path <- args[1]
output_path  <- args[2]

# Load fixture
spe <- readRDS(fixture_path)

# Run SpaNorm with small parameters for speed
set.seed(42)
spe <- SpaNorm(spe, sample.p = 0.25, df.tps = 2, tol = 1e-2, verbose = FALSE)

# Extract logcounts
logcounts_mat <- as.matrix(logcounts(spe))

# Write output as JSON
result <- list(
  logcounts = logcounts_mat,
  n_genes = nrow(logcounts_mat),
  n_cells = ncol(logcounts_mat)
)

jsonlite::write_json(result, output_path, auto_unbox = TRUE, matrix = "rowmajor", digits = NA)

cat("R reference driver completed successfully.\n")
