if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("rhdf5")
BiocManager::install("hdf5r")


library(rhdf5)
library(hdf5r)
library(Seurat)
data <- Read10X_h5("./data/scRNA_Seq/sc5p_v2_hs_melanoma_10k_targeted_pan_cancer_filtered_feature_bc_matrix.h5")
seurat_obj <- CreateSeuratObject(counts = data)
seurat_obj


expr_matrix <- GetAssayData(seurat_obj, slot = "counts") # raw counts
expr_matrx2 <- as.matrix(expr_matrix)
View(expr_matrx2)
dim(expr_matrx2)

meta_data <- seurat_obj@meta.data
meta_data
View(meta_data)

meta_data$cell_type

# Checking if the class contains
colnames(seurat_obj@meta.data)


# Meta datas
df2 <- read.csv("./data/scRNA_Seq/sc5p_v2_hs_melanoma_10k_targeted_pan_cancer_tc_barcode_summary.csv", header = T)
df3 <- read.csv("./data/scRNA_Seq/sc5p_v2_hs_melanoma_10k_targeted_pan_cancer_tc_feature_summary.csv", header = T)
View(df2)
View(df3)
