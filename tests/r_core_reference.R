#!/usr/bin/env Rscript
# Core SpaNorm algorithm reference - runs without Bioconductor dependencies.
# Implements bs.tps, fitSpaNormNB, and normalization directly.
# Usage: Rscript r_core_reference.R <output_dir>

args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args) >= 1) args[1] else "data"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

cat("=== R Core Reference for SpaNorm ===\n\n")

# ---- 1. Generate synthetic spatial data ----
set.seed(42)
ngenes <- 50
ncells <- 200

# Spatial coordinates
coords <- cbind(
  runif(ncells, 0, 10),
  runif(ncells, 0, 10)
)

# Generate count data with spatial structure
base_expr <- runif(ngenes, 1, 5)
mu <- matrix(base_expr, ngenes, ncells)

# Add spatial patterns
x_norm <- (coords[,1] - min(coords[,1])) / (max(coords[,1]) - min(coords[,1]))
y_norm <- (coords[,2] - min(coords[,2])) / (max(coords[,2]) - min(coords[,2]))

for (g in 1:min(15, ngenes)) {
  pattern <- sin(x_norm * pi * ((g-1) %% 3 + 1)) * cos(y_norm * pi * ((g-1) %% 2 + 1))
  modulation <- 1 + 2 * (pattern - min(pattern)) / (max(pattern) - min(pattern) + 1e-10)
  mu[g, ] <- mu[g, ] * modulation
}

# Library size variation
ls_factor <- rlnorm(ncells, 0, 0.5)
mu <- mu * matrix(ls_factor, ngenes, ncells, byrow = TRUE)

# Generate NB counts
counts <- matrix(0, ngenes, ncells)
psi_true <- runif(ngenes, 0.1, 0.5)
for (g in 1:ngenes) {
  size_g <- 1 / psi_true[g]
  p_g <- size_g / (size_g + mu[g, ])
  counts[g, ] <- rnbinom(ncells, size = size_g, prob = p_g)
}

cat(sprintf("Data: %d genes x %d cells\n", ngenes, ncells))
cat(sprintf("Count range: [%d, %d]\n", min(counts), max(counts)))

# ---- 2. bs.tps (thin-plate spline basis) ----
bs_tps <- function(x, y, df.tps = 6) {
  stopifnot(df.tps > 0)
  stopifnot(length(x) == length(y))

  xrng <- diff(range(x))
  yrng <- diff(range(y))
  gap <- max(xrng, yrng) / df.tps
  df.tps.x <- ceiling(xrng / gap)
  df.tps.y <- ceiling(yrng / gap)
  df.tps.x <- max(df.tps.x, 1)
  df.tps.y <- max(df.tps.y, 1)

  bs.x <- splines::ns(x, df = df.tps.x)
  bs.y <- splines::ns(y, df = df.tps.y)
  bs.xy <- matrix(0, nrow = length(x), ncol = ncol(bs.x) * ncol(bs.y))
  for (i in seq_len(ncol(bs.x))) {
    for (j in seq_len(ncol(bs.y))) {
      bs.xy[, (i - 1) * ncol(bs.y) + j] <- bs.x[, i] * bs.y[, j]
    }
  }
  bs.xy <- scale(bs.xy, scale = FALSE)
  attr(bs.xy, 'df.tps') <- c(df.tps.x, df.tps.y)
  return(bs.xy)
}

# ---- 3. Scale coordinates ----
coords_scaled <- apply(coords, 2, function(x) {
  (x - min(x)) / (max(x) - min(x)) - 0.5
})

# ---- 4. Build spline bases ----
df_tps <- 2
bs.xy.bio <- bs_tps(coords_scaled[,1], coords_scaled[,2], df.tps = df_tps)
df.tps.bio <- attr(bs.xy.bio, "df.tps")
df.tps.ls <- c(max(1, ceiling(df.tps.bio[1]/2)), max(1, ceiling(df.tps.bio[2]/2)))
bs.xy.ls <- bs_tps(coords_scaled[,1], coords_scaled[,2], df.tps = max(df.tps.ls))

cat(sprintf("Biology spline: %d x %d = %d columns\n", df.tps.bio[1], df.tps.bio[2], ncol(bs.xy.bio)))
cat(sprintf("LS spline: %d x %d = %d columns\n", df.tps.ls[1], df.tps.ls[2], ncol(bs.xy.ls)))

# ---- 5. Compute size factors ----
LS <- colSums(counts)
LS <- LS / mean(LS)
logLS <- log(pmax(1e-8, LS))

# ---- 6. Build covariate matrix W ----
W <- model.matrix(~ logLS + bs.xy.bio + logLS:bs.xy.ls)[, -1, drop = FALSE]
cat(sprintf("W matrix: %d x %d\n", nrow(W), ncol(W)))

# Mark covariate types
n_bio <- prod(df.tps.bio)
n_ls <- prod(df.tps.ls)
wtype <- rep("batch", ncol(W))
wtype[seq(2, n_bio + 1)] <- "biology"
wtype[c(1, seq(n_bio + 2, n_bio + n_ls + 1))] <- "ls"

cat(sprintf("Covariate types: biology=%d, ls=%d, batch=%d\n",
            sum(wtype == "biology"), sum(wtype == "ls"), sum(wtype == "batch")))

# ---- 7. Sample cells for fitting ----
sample_p <- 0.25
nsub <- round(sample_p * ncells)
set.seed(42)
idx <- rep(FALSE, ncells)
idx[sample.int(ncells, size = nsub)] <- TRUE

# ---- 8. Simple NB fitting (simplified IRLS) ----
# This is a simplified version of the SpaNorm NB fitter
# that works without edgeR

cat("\n--- Fitting NB model (simplified) ---\n")

Y_sub <- counts[, idx, drop = FALSE]
W_sub <- W[idx, , drop = FALSE]
nW <- ncol(W_sub)

# Initialize
gmean <- rowMeans(log(Y_sub + 1))
alpha <- matrix(0, ngenes, nW)
alpha[, 1] <- 1

# Simple dispersion estimation (method of moments)
psi <- numeric(ngenes)
for (g in 1:ngenes) {
  y_g <- Y_sub[g, ]
  var_g <- var(y_g)
  mean_g <- mean(y_g)
  if (mean_g > 0) {
    psi[g] <- max((var_g - mean_g) / mean_g^2, 0.01)
  } else {
    psi[g] <- 0.1
  }
}

# IRLS iterations (simplified)
maxit <- 10
tol <- 1e-3
loglik_prev <- -Inf

for (iter in 1:maxit) {
  # Working response
  lmu_hat <- gmean + alpha %*% t(W_sub)
  lmu_hat <- pmin(pmax(lmu_hat, -50), 50)
  mu_hat <- exp(lmu_hat)

  # Log-likelihood
  loglik <- 0
  for (g in 1:ngenes) {
    loglik <- loglik + sum(dnbinom(Y_sub[g, ], mu = mu_hat[g, ], size = 1/psi[g], log = TRUE))
  }

  cat(sprintf("iter %2d: loglik = %.4f\n", iter, loglik))

  # Check convergence
  if (iter > 1 && abs(loglik - loglik_prev) / abs(loglik_prev) < tol) {
    cat("Converged!\n")
    break
  }
  loglik_prev <- loglik

  # Working vector Z
  Z <- lmu_hat + ((Y_sub + 0.01) / (mu_hat + 0.01) - 1)

  # Update alpha (weighted least squares)
  sig_inv <- 1 / (psi * exp(-lmu_hat))
  sig_inv <- pmin(sig_inv, 1e10)
  wt_cell <- colMeans(sig_inv)
  wt_cell <- pmin(wt_cell, quantile(wt_cell, 0.98))

  Z_centered <- Z - gmean
  b <- (Z_centered * wt_cell) %*% W_sub
  WtW <- t(W_sub) %*% (wt_cell * W_sub)

  # Regularization
  lambda_a <- rep(0, nW)
  lambda_a[wtype[-1] == "biology"] <- 0.0001 * ncells
  lambda_a[wtype[-1] == "ls"] <- 0.0001 * ncells
  reg <- diag(lambda_a)

  alpha <- b %*% solve(WtW + reg)

  # First column shared (SpaNorm constraint)
  alpha[, 1] <- mean(alpha[, 1])

  # Update gmean
  Z_res <- Z - alpha %*% t(W_sub)
  gmean <- rowSums(Z_res * sig_inv) / rowSums(sig_inv)
}

# ---- 9. Normalization (logPAC method) ----
cat("\n--- Normalizing data ---\n")

# Full mu
lmu_full <- gmean + alpha %*% t(W)
lmu_full <- pmin(pmax(lmu_full, -50), 50)
mu_full <- exp(lmu_full)

# Biology-only mu
is_bio <- wtype == "biology"
lmu_bio <- gmean + alpha[, is_bio, drop = FALSE] %*% t(W[, is_bio, drop = FALSE])
lmu_bio <- pmin(pmax(lmu_bio, -50), 50)
mu_bio <- exp(lmu_bio)

# Winsorize psi
psi_win <- psi
psi_max <- exp(median(log(psi)) + 3 * mad(log(psi)))
psi_win <- pmin(psi_win, psi_max)

# logPAC normalization
cat("Computing logPAC...\n")
lb <- pnbinom(counts - 1, mu = mu_full, size = 1/psi_win)
ub <- dnbinom(counts, mu = mu_full, size = 1/psi_win) + lb
p <- (lb + ub) / 2
p <- pmax(pmin(p, 0.999), 0.001)

logcounts_r <- log2(qnbinom(p, mu = mu_bio, size = 1/psi_win) + 1)

cat(sprintf("logcounts range: [%.4f, %.4f]\n", min(logcounts_r), max(logcounts_r)))
cat(sprintf("logcounts mean: %.4f\n", mean(logcounts_r)))

# ---- 10. Save results ----
cat("\n--- Saving results ---\n")

# Save R output
write.csv(logcounts_r, file.path(output_dir, "r_logcounts.csv"), row.names = FALSE)
write.csv(counts, file.path(output_dir, "r_counts.csv"), row.names = FALSE)
write.csv(coords, file.path(output_dir, "r_coords.csv"), row.names = FALSE)

# Save parameters
params <- list(
  gmean = gmean,
  alpha = alpha,
  psi = psi,
  psi_win = psi_win,
  W = W,
  wtype = wtype
)
saveRDS(params, file.path(output_dir, "r_params.rds"))

cat(sprintf("\nResults saved to %s/\n", output_dir))
cat("=== R Reference Complete ===\n")
