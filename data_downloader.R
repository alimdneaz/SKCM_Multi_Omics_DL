# Corrupted File
# untar("./gdc_download_20250129_115746.697634.tar.gz", exdir = "gdc_data")

# Importing Libraries
library(TCGAbiolinks) 
library(dplyr)
library(maftools)
library(pheatmap)
library(SummarizedExperiment)
library(sesameData)
library(sesame)

query <- GDCquery(project = "TCGA-SKCM")
results <- getResults(query)
View()


# 1. Mutation and CNV data Download
# query_mutation <- GDCquery(project = 'TCGA-SKCM',
#                            data.category = 'Simple Nucleotide Variation',
#                            data.type = "Masked Somatic Mutation",
#                            access = 'open',
#                            workflow.type = "Aliquot Ensemble Somatic Variant Merging and Masking",
#                            barcode = cases)

query_mutation <- GDCquery(project = 'TCGA-SKCM',
                           data.category = 'Simple Nucleotide Variation',
                           access = 'open',
                           workflow.type = "Aliquot Ensemble Somatic Variant Merging and Masking")

output_mutation <- getResults(query_mutation)
View(output_mutation)



output_mutation %>% select(data_category) %>% table()
output_mutation %>% select(type) %>% table()
output_mutation %>% select(data_type) %>% table()
output_mutation %>% select(analysis_workflow_type) %>% table()

cases <- output_mutation$cases
## Download The data
GDCdownload(query_mutation)

maf <- GDCprepare(query_mutation, summarizedExperiment = TRUE)

View(maf)
class(maf)
if(!dir.exists("data")){
  dir.create("data", showWarnings = F, recursive = T)
}

write.csv(maf, "mutation_data.csv")
write.csv(output_mutation, "data/mutation_meta_data.csv")
# maftools utils to read and create dashboard
maftools.input <- read.maf(maf)

plotmafSummary(maf = maftools.input,
               addStat = 'median',
               rmOutlier = TRUE,
               dashboard = TRUE)
dim(maf)


# Transcriptomics Data
query_transcriptom <- GDCquery(project = "TCGA-SKCM",
                       data.category  = "Transcriptome Profiling",
                       data.type = "Gene Expression Quantification",
                       experimental.strategy = 'RNA-Seq',
                       workflow.type = 'STAR - Counts',
                       access = "open",
                       barcode = cases_transcriptom)
output_transcriptom <- getResults(query_transcriptom)
View(output_transcriptom)
cases_transcriptom <- output_transcriptom$cases
#! Face Error While Downloading
GDCdownload(query_transcriptom)


skcm_data <- GDCprepare(query_transcriptom, summarizedExperiment = TRUE)
brca_matrix <- assay(tcga_brca_data, 'fpkm_unstrand')


# Proteomics Data 
query_proteomics <- GDCquery(project = "TCGA-SKCM",
                             data.category  = "Proteome Profiling",
                             data.type = "Protein Expression Quantification",
                             experimental.strategy = 'Reverse Phase Protein Array',
                             access = "open")
output_proteomics <- getResults(query_proteomics)
View(output_proteomics)

GDCdownload(query_proteomics)
proteomics.data <- GDCprepare(query_proteomics, summarizedExperiment = TRUE)

proteomics.data.asay <- assay(proteomics.data)
proteomics.data
View(proteomics.data)

metadata.proteomics
write.csv(output_proteomics, "./data/proteomics_data_meta_data.csv")
write.csv(proteomics.data, "./data/proteomics_data.csv")

# Checking The Meta Data Ids
meta_data_rna <- read.csv("./data/RNA_Seq_transcriptom_meta_data.csv", header= T)
meta_data_mutation <- read.csv("./data/mutation_meta_data.csv", header = T)
meta_data_transcriptom <- read.csv("./data/proteomics_data_meta_data.csv", header = T)


# Checking the ids if they match
ranaids <- meta_data_rna$id
mutationids <- meta_data_mutation$id
transcriptids <- meta_data_transcriptom$id

total_mathc = 0
for(ids in ranaids){
  if(ids %in% mutationids){
    total_mathc = total_mathc+ 1
  }
}
total_mathc

for(ids in ranaids){
  if(ids %in% transcriptids){
    total_mathc = total_mathc+ 1
  }
}
total_mathc
# Clinical Data Download
query_clinical <- GDCquery(project = "TCGA-SKCM",
                            data.category  = "Clinical",
                            data.type = "Clinical Supplement",
                           data.format = "xml",
                            access = "open")
  
output_clinical <- getResults(query_clinical)
View(output_clinical)

GDCdownload(query_clinical)

clinical.data <- GDCprepare_clinic(query_clinical, clinical.info = "patient")

# 2nd Clinical Data Download
clinical <- GDCquery_clinic(project = "TCGA-SKCM",
                            type = "clinical")
clinical %>% 
  head() %>% 
  DT::datatable(filter = "top",
                options = list(scrollX = T, keys = T, pageLength = 5),
                rownames = F)

class(clinical)
View(clinical)
write.csv(clinical, "data/clinical_data.csv",row.names = F)

GDCdownload(clinical)


# XML clinical data: Not working
query_clinical_xml <- GDCquery(project = "TCGA-SKCM",
                               data.category = 'Clinical',
                               data.type = "xml")
?GDCquery

# Tissue slid image (svs format)
query_clinical_image <- GDCquery(project = "TCGA-SKCM",
                                 data.category = "Clinical",
                                 data.type = "Tissue slide image")
query <- GDCquery(project = "TCGA-COAD", 
                  data.category = "Clinical", 
                  data.type = "Clinical Supplement",
                  barcode = c("TCGA-RU-A8FL","TCGA-AA-3972")) 
df <- getResults(query)
View(df)

installed.packages()["TCGAbiolinks", "Version"]
