# A novel graph neural network algorithm for interpreting breast cancer mutation and prognostic risk using hierarchical map of protein communities

## Introduction
This repository provides the source code and raw datasets associated with the paper A novel graph neural network algorithm for interpreting breast cancer mutation and prognostic risk using hierarchical map of protein communities.

Breast cancer (BC) is the most diagnosed cancer and is one of the leading causes of cancer death for women worldwide. Clinical biomarkers of BC are crucial assets for further research as drug targets and early indicators of disease. These biomarkers can be identified using proteomics data, in conjunctions with multi-omics data, to identify protein communities under both mutation burden and survival burden.

We proposed the utilization of a graph neural network (GNN) to identify clusters of protein communities based on multi-omics data. The resulting clusters will be constructed into a hierarchical map based on relevant gene ontology terms. The resulting map will show clinical biomarkers of BC at various levels, ranging from gene-level to system-level. The overall project workflow is depicted below.
![Overall project workflow](https://github.com/maomao853/BC-Multi-Omics-GNN/assets/70410309/3962b0fe-9609-4951-9aec-80f731708c71)

## Graph Neural Network

### Architecture
![GNN architecture](architecture.png)

### Data
The GNN is trained on protein data and multi-omics data from various sources.
* Protein amino acid sequence from [UniProt database](https://doi.org/10.1093/nar/gkaa1100).
* Protein-protein interaction data from [Zheng et al.'s paper](https://doi.org/10.1126/science.abf3067).
* Human gene ontology (GO) terms from the [GO Database](https://doi.org/10.1038/75556).

The data combined with gene-level meta-survival effects and multi-omics data from The Cancer Genome Atlas Program (TCGA) to construct Cox regression models for each gene. Four z-scores were used, corresponding to four different omics data: Copy Number Variation (CNA), gene expression, DNA methylation, and mutation. These were combined to create a graph of nodes/genes/proteins and edges/protein pairs.

An Evolutionary Scale Modeling (ESM)-1b Transformer from [Rives et al.](https://doi.org/10.1073/pnas.2016239118) was used to extract features from the protein dataset.

### Training and Testing
To train and test the GNN model, make sure the dependencies are satisfied, then run
```
bash HGNN_main.sh
```

Dependencies
* [HiLander Model](https://github.com/dmlc/dgl/tree/master/examples/pytorch/hilander)
* [R](https://www.r-project.org/)

###

## Clinical Hotspots Identification

### Mutation/Survival
To identify clinical hotspots based on mutation and survival pressures using the [cBioPortal OncoPrint](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4160307/) database, run the following code.
```
bash group_lasso_mutation_survival.sh
```

### DepMap/CMap
To identify clinical hotspots based on dependencies and connectivity mapping using the [DepMap](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5667678/) and [CMap](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5952941/) databases, run the following code.
```
sh group_lasso_DepMap_cMap.sh
```

## Hierarchical Map Construction
WIP
