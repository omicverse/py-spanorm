#!/usr/bin/env Rscript
# Generate larger dataset for better convergence comparison.
args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args) >= 1) args[1] else "data"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

cat("=== R Large Dataset Reference ===\n\n")

set.seed(42)
ngenes <- 100
ncells <- 500

coords <- cbind(runif(ncells, 0, 10), runif(ncells, 0, 10))
base_expr <- runif(ngenes, 2, 8)
mu <- matrix(base_expr, ngenes, ncells)

x_norm <- (coords[,1] - min(coords[,1])) / (max(coords[,1]) - min(coords[,1]))
y_norm <- (coords[,2] - min(coords[,2])) / (max(coords[,2]) - min(coords[,2]))

for (g in 1:min(30, ngenes)) {
  pattern <- sin(x_norm * pi * ((g-1) %% 3 + 1)) * cos(y_norm * pi * ((g-1) %% 2 + 1))
  modulation <- 1 + 2 * (pattern - min(pattern)) / (max(pattern) - min(pattern) + 1e-10)
  mu[g, ] <- mu[g, ] * modulation
}

ls_factor <- rlnorm(ncells, 0, 0.3)
mu <- mu * matrix(ls_factor, ngenes, ncells, byrow = TRUE)

counts <- matrix(0L, ngenes, ncells)
psi_true <- runif(ngenes, 0.1, 0.4)
for (g in 1:ngenes) {
  counts[g, ] <- rnbinom(ncells, size = 1/psi_true[g], mu = mu[g, ])
}

cat(sprintf("Data: %d genes x %d cells\n", ngenes, ncells))

# Save shared data
write.csv(as.data.frame(counts), file.path(output_dir, "large_counts.csv"), row.names = FALSE)
write.csv(as.data.frame(coords), file.path(output_dir, "large_coords.csv"), row.names = FALSE)

# Build model
coords_scaled <- apply(coords, 2, function(x) (x - min(x)) / (max(x) - min(x)) - 0.5)

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

LS <- colSums(counts) / mean(colSums(counts))
logLS <- log(pmax(1e-8, LS))
W <- model.matrix(~ logLS + bs.xy.bio + logLS:bs.xy.ls)[, -1, drop = FALSE]
n_bio <- ncol(bs.xy.bio)
n_ls <- ncol(bs.xy.ls)
wtype <- rep("ls", ncol(W))
wtype[1] <- "ls"
if (n_bio > 0) wtype[2:(1+n_bio)] <- "biology"
if (n_ls > 0) wtype[(2+n_bio):(1+n_bio+n_ls)] <- "ls"

write.csv(as.data.frame(bs.xy.bio), file.path(output_dir, "large_bs_xy_bio.csv"), row.names = FALSE)
write.csv(as.data.frame(bs.xy.ls), file.path(output_dir, "large_bs_xy_ls.csv"), row.names = FALSE)
write.csv(as.data.frame(W), file.path(output_dir, "large_W.csv"), row.names = FALSE)

# Fit
sample_p <- 0.25
nsub <- round(sample_p * ncells)
set.seed(42)
idx <- sample.int(ncells, size = nsub)
write.csv(data.frame(idx = idx), file.path(output_dir, "large_idx.csv"), row.names = FALSE)

Y_sub <- counts[, idx, drop = FALSE]
W_sub <- W[idx, , drop = FALSE]
nW <- ncol(W_sub)

t_start <- proc.time()

gmean <- rowMeans(log(Y_sub + 1))
alpha <- matrix(0, ngenes, nW)
alpha[, 1] <- 1

psi <- numeric(ngenes)
for (g in 1:ngenes) {
  v <- var(as.numeric(Y_sub[g, ]))
  m <- mean(as.numeric(Y_sub[g, ]))
  psi[g] <- max((v - m) / max(m^2, 1e-10), 0.01)
}

for (iter in 1:30) {
  lmu <- pmin(pmax(gmean + alpha %*% t(W_sub), -50), 50)
  mu_hat <- exp(lmu)
  ll <- sum(dnbinom(Y_sub, mu = mu_hat, size = outer(1/psi, rep(1, nsub)), log = TRUE))

  Z <- lmu + ((Y_sub + 0.01) / (mu_hat + 0.01) - 1)
  sig_inv <- 1 / (outer(psi, rep(1, nsub)) * exp(-lmu))
  sig_inv <- pmin(sig_inv, 1e10)
  wt <- pmin(colMeans(sig_inv), quantile(colMeans(sig_inv), 0.98))

  b <- ((Z - gmean) * wt) %*% W_sub
  WtW <- t(W_sub) %*% (wt * W_sub)
  lam <- rep(0, nW - 1)
  lam[wtype[-1] == "biology"] <- 0.0001 * ncells
  lam[wtype[-1] == "ls"] <- 0.0001 * ncells
  reg_mat <- matrix(0, nW, nW)
  diag(reg_mat)[-1] <- lam

  alpha_new <- b %*% solve(WtW + reg_mat)
  alpha_new[, 1] <- mean(alpha_new[, 1])

  lmu_new <- pmin(pmax(gmean + alpha_new %*% t(W_sub), -50), 50)
  ll_new <- sum(dnbinom(Y_sub, mu = exp(lmu_new), size = outer(1/psi, rep(1, nsub)), log = TRUE))

  cat(sprintf("R iter %2d: loglik = %.2f\n", iter, ll))

  if (iter > 1 && abs(ll_new - ll) / max(abs(ll), 1) < 1e-4) {
    alpha <- alpha_new
    cat("R converged\n")
    break
  }
  alpha <- alpha_new

  Z_res <- Z - alpha %*% t(W_sub)
  gmean <- rowSums(Z_res * sig_inv) / rowSums(sig_inv)
}

t_r <- (proc.time() - t_start)["elapsed"]
cat(sprintf("R time: %.3f seconds\n", t_r))

# Normalize
is_bio <- wtype == "biology"
lmu_bio <- pmin(pmax(gmean + alpha[, is_bio, drop = FALSE] %*% t(W[, is_bio, drop = FALSE]), -50), 50)
logcounts_r <- log2(exp(lmu_bio))

# Save
write.csv(as.data.frame(logcounts_r), file.path(output_dir, "large_r_logcounts.csv"), row.names = FALSE)
write.csv(data.frame(gmean = gmean), file.path(output_dir, "large_r_gmean.csv"), row.names = FALSE)
write.csv(as.data.frame(alpha), file.path(output_dir, "large_r_alpha.csv"), row.names = FALSE)
write.csv(data.frame(psi = psi), file.path(output_dir, "large_r_psi.csv"), row.names = FALSE)
writeLines(as.character(t_r), file.path(output_dir, "large_r_time.txt"))

cat("Done\n")
