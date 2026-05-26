# Discovery — SpaNorm

## Phase 0.5 Audit

### 目标 R 包
- **名称**: SpaNorm
- **版本**: 1.5.2
- **来源**: Bioconductor (GitHub: bhuvad/SpaNorm)
- **License**: GPL (>= 3)

### 已有 port 检查
- `gh auth` 未配置，无法远程检查
- 本地确认：无已知 py-SpaNorm 存在

### R 依赖审计

| R 依赖 | 是否有 py-镜像 | 备注 |
|---|---|---|
| edgeR | ❌ 无 | NB 分布估计，已自行实现 |
| scran | ❌ 无 | size factors，已用简化实现替代 |
| BiocSingular | ❌ 无 | PCA，已用 sklearn 替代 |
| Matrix | ✅ scipy.sparse | 稀疏矩阵 |
| matrixStats | ✅ numpy | 行列统计 |
| SingleCellExperiment | ✅ anndata | 数据容器 |
| SpatialExperiment | ✅ anndata + obsm['spatial'] | 空间坐标 |
| SeuratObject | ✅ anndata | 数据容器 |
| ggplot2 | ⬜ 跳过 | 可视化，不在 port 范围 |

### 结论
核心数值依赖（edgeR, scran, BiocSingular）无 py-镜像，已在 port 内自行实现。
