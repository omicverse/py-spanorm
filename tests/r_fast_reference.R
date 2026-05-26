#!/usr/bin/env Rscript
# Fast R reference for core SpaNorm comparison.
# Uses mean_bio normalization (fast) instead of logPAC (slow).
# Usage: Rscript r_fast_reference.R <output_dir>

args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args) >= 1) args[1] else "data"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

cat("=== R Fast Reference for SpaNorm ===\n\n")

# ---- 1. Generate synthetic spatial data ----
set.seed(42)
ngenes <- 30
ncells <- 100

coords <- cbind(runif(ncells, 0, 10), runif(ncells, 0, 10))

# Generate counts
base_expr <- runif(ngenes, 2, 8)
mu <- matrix(base_expr, ngenes, ncells)

x_norm <- (coords[,1] - min(coords[,1])) / (max(coords[,1]) - min(coords[,1]))
y_norm <- (coords[,2] - min(coords[,2])) / (max(coords[,2]) - min(coords[,2]))

for (g in 1:min(10, ngenes)) {
  pattern <- sin(x_norm * pi * ((g-1) %% 3 + 1)) * cos(y_norm * pi * ((g-1) %% 2 + 1))
  modulation <- 1 + 2 * (pattern - min(pattern)) / (max(pattern) - min(pattern) + 1e-10)
  mu[g, ] <- mu[g, ] * modulation
}

ls_factor <- rlnorm(ncells, 0, 0.3)
mu <- mu * matrix(ls_factor, ngenes, ncells, byrow = TRUE)

counts <- matrix(0L, ngenes, ncells)
psi_true <- runif(ngenes, 0.1, 0.4)
for (g in 1:ngenes) {
  size_g <- 1 / psi_true[g]
  counts[g, ] <- rnbinom(ncells, size = size_g, mu = mu[g, ])
}

cat(sprintf("Data: %d genes x %d cells\n", ngenes, ncells))

# ---- 2. Scale coordinates ----
coords_scaled <- apply(coords, 2, function(x) (x - min(x)) / (max(x) - min(x)) - 0.5)

# ---- 3. Build spline bases (df_tps = 2) ----
df_tps <- 2L
xrng <- diff(range(coords_scaled[,1]))
yrng <- diff(range(coords_scaled[,2]))
gap <- max(xrng, yrng) / df_tps
df.tps.x <- max(ceiling(xrng / gap), 1L)
df.tps.y <- max(ceiling(yrng / gap), 1L)

bs.x <- splines::ns(coords_scaled[,1], df = df.tps.x)
bs.y <- splines::ns(coords_scaled[,2], df = df.tps.y)

bs.xy.bio <- matrix(0, ncells, ncol(bs.x) * ncol(bs.y))
for (i in seq_len(ncol(bs.x))) {
  for (j in seq_len(ncol(bs.y))) {
    bs.xy.bio[, (i-1)*ncol(bs.y) + j] <- bs.x[,i] * bs.y[,j]
  }
}
bs.xy.bio <- scale(bs.xy.bio, scale = FALSE)

# LS spline (half df)
df.ls.x <- max(ceiling(df.tps.x / 2), 1L)
df.ls.y <- max(ceiling(df.tps.y / 2), 1L)
bs.x.ls <- splines::ns(coords_scaled[,1], df = df.ls.x)
bs.y.ls <- splines::ns(coords_scaled[,2], df = df.ls.y)
bs.xy.ls <- matrix(0, ncells, ncol(bs.x.ls) * ncol(bs.y.ls))
for (i in seq_len(ncol(bs.x.ls))) {
  for (j in seq_len(ncol(bs.y.ls))) {
    bs.xy.ls[, (i-1)*ncol(bs.y.ls) + j] <- bs.x.ls[,i] * bs.y.ls[,j]
  }
}
bs.xy.ls <- scale(bs.xy.ls, scale = FALSE)

cat(sprintf("Biology spline: %d cols, LS spline: %d cols\n", ncol(bs.xy.bio), ncol(bs.xy.ls)))

# ---- 4. Size factors ----
LS <- colSums(counts) / mean(colSums(counts))
logLS <- log(pmax(1e-8, LS))

# ---- 5. Build W ----
W <- model.matrix(~ logLS + bs.xy.bio + logLS:bs.xy.ls)[, -1, drop = FALSE]
n_bio <- ncol(bs.xy.bio)
n_ls <- ncol(bs.xy.ls)

wtype <- rep("ls", ncol(W))
wtype[1] <- "ls"  # logLS
if (n_bio > 0) wtype[2:(1+n_bio)] <- "biology"
if (n_ls > 0) wtype[(2+n_bio):(1+n_bio+n_ls)] <- "ls"

cat(sprintf("W: %d x %d (bio=%d, ls=%d)\n", nrow(W), ncol(W), sum(wtype=="biology"), sum(wtype=="ls")))

# ---- 6. Fit model (simplified IRLS) ----
sample_p <- 0.5
nsub <- round(sample_p * ncells)
set.seed(42)
idx <- sample.int(ncells, size = nsub)

Y_sub <- counts[, idx, drop = FALSE]
W_sub <- W[idx, , drop = FALSE]
nW <- ncol(W_sub)

gmean <- rowMeans(log(Y_sub + 1))
alpha <- matrix(0, ngenes, nW)
alpha[, 1] <- 1

# Method-of-moments dispersion
psi <- numeric(ngenes)
for (g in 1:ngenes) {
  v <- var(as.numeric(Y_sub[g, ]))
  m <- mean(as.numeric(Y_sub[g, ]))
  psi[g] <- max((v - m) / max(m^2, 1e-10), 0.01)
}

# IRLS
for (iter in 1:15) {
  lmu <- pmin(pmax(gmean + alpha %*% t(W_sub), -50), 50)
  mu_hat <- exp(lmu)

  # Log-likelihood
  ll <- sum(dnbinom(Y_sub, mu = mu_hat, size = outer(1/psi, rep(1, nsub)), log = TRUE))

  # Working response
  Z <- lmu + ((Y_sub + 0.01) / (mu_hat + 0.01) - 1)

  # Update alpha
  sig_inv <- 1 / (outer(psi, rep(1, nsub)) * exp(-lmu))
  sig_inv <- pmin(sig_inv, 1e10)
  wt <- pmin(colMeans(sig_inv), quantile(colMeans(sig_inv), 0.98))

  b <- ((Z - gmean) * wt) %*% W_sub
  WtW <- t(W_sub) %*% (wt * W_sub)
  lam <- rep(0, nW)
  lam[wtype == "biology"] <- 0.0001 * ncells
  lam[wtype == "ls"] <- 0.0001 * ncells

  alpha_new <- b %*% solve(WtW + diag(lam))
  alpha_new[, 1] <- mean(alpha_new[, 1])

  # Check convergence
  lmu_new <- pmin(pmax(gmean + alpha_new %*% t(W_sub), -50), 50)
  ll_new <- sum(dnbinom(Y_sub, mu = exp(lmu_new), size = outer(1/psi, rep(1, nsub)), log = TRUE))

  cat(sprintf("iter %2d: loglik = %.2f\n", iter, ll))

  if (iter > 1 && abs(ll_new - ll) / max(abs(ll), 1) < 1e-3) {
    alpha <- alpha_new
    cat("Converged\n")
    break
  }
  alpha <- alpha_new

  # Update gmean
  Z_res <- Z - alpha %*% t(W_sub)
  gmean <- rowSums(Z_res * sig_inv) / rowSums(sig_inv)
}

# ---- 7. Normalization (mean_bio - fast) ----
cat("\n--- Normalizing (mean_bio) ---\n")
is_bio <- wtype == "biology"
lmu_bio <- gmean + alpha[, is_bio, drop = FALSE] %*% t(W[, is_bio, drop = FALSE])
lmu_bio <- pmin(pmax(lmu_bio, -50), 50)
logcounts_r <- log2(exp(lmu_bio))

cat(sprintf("logcounts range: [%.4f, %.4f]\n", min(logcounts_r), max(logcounts_r)))

# ---- 8. Also compute full mu for reference ----
lmu_full <- pmin(pmax(gmean + alpha %*% t(W), -50), 50)
mu_full <- exp(lmu_full)

# ---- 9. Save ----
write.csv(as.data.frame(logcounts_r), file.path(output_dir, "r_logcounts.csv"), row.names = FALSE)
write.csv(as.data.frame(counts), file.path(output_dir, "r_counts.csv"), row.names = FALSE)
write.csv(as.data.frame(coords), file.path(output_dir, "r_coords.csv"), row.names = FALSE)
write.csv(as.data.frame(W), file.path(output_dir, "r_W.csv"), row.names = FALSE)

# Save model parameters
cat(sprintf("\nModel: gmean[1:3] = [%.6f, %.6f, %.6f]\n", gmean[1], gmean[2], gmean[3]))
cat(sprintf("alpha[1,1:3] = [%.6f, %.6f, %.6f]\n", alpha[1,1], alpha[1,2], alpha[1,3]))
cat(sprintf("psi[1:3] = [%.6f, %.6f, %.6f]\n", psi[1], psi[2], psi[3]))

# Save params as RDS
params <- list(gmean = gmean, alpha = alpha, psi = psi, W = W, wtype = wtype, logcounts = logcounts_r)
saveRDS(params, file.path(output_dir, "r_params.rds"))

cat(sprintf("\nResults saved to %s/\n", output_dir))
cat("=== R Reference Complete ===\n")
