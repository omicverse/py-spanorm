#!/usr/bin/env Rscript
# Generate a small fixture for parity testing from the HumanDLPFC data.
# Usage: Rscript generate_fixture.R

suppressMessages({
  library(SpaNorm)
  library(SpatialExperiment)
})

# Load built-in data
data(HumanDLPFC)

# Subset to a small number of genes for fast testing
set.seed(42)
gene_idx <- sample(nrow(HumanDLPFC), 50)
spe_small <- HumanDLPFC[gene_idx, ]

# Save RDS fixture
saveRDS(spe_small, "../data/fixture_HumanDLPFC.rds")

cat(sprintf("Saved fixture: %d genes x %d cells\n", nrow(spe_small), ncol(spe_small)))
cat("Run R reference:\n")
cat("  Rscript tests/r_reference_driver.R data/fixture_HumanDLPFC.rds data/r_output.json\n")
