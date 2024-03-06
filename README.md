# A novel graph neural network algorithm for interpreting breast cancer mutation and prognostic risk using hierarchical map of protein communities

## Introduction
This repository provides the source code and raw datasets associated with the paper A novel graph neural network algorithm for interpreting breast cancer mutation and prognostic risk using hierarchical map of protein communities.

Breast cancer (BC) is the most diagnosed cancer and is one of the leading causes of cancer death for women worldwide. Clinical biomarkers of BC are crucial assets for further research as drug targets and early indicators of disease. These biomarkers can be identified using proteomics data, in conjunctions with multi-omics data, to identify protein communities under both mutation burden and survival burden.

We proposed the utilization of a graph neural network (GNN) to identify clusters of protein communities based on multi-omics data. The resulting clusters will be constructed into a hierarchical map based on relevant gene ontology terms. The resulting map will show clinical biomarkers of BC at various levels, ranging from gene-level to system-level. The overall project workflow is depicted below.
![Overall project workflow](https://github.com/maomao853/BC-Multi-Omics-GNN/assets/70410309/3962b0fe-9609-4951-9aec-80f731708c71)

## Graph Neural Network
### Data
The GNN is trained on protein data and multi-omics data from various sources.
* Protein amino acid sequence from [UniProt database](https://doi.org/10.1093/nar/gkaa1100).
* Protein-protein interaction data from [Zheng et al.'s paper](https://doi.org/10.1126/science.abf3067).
* Human gene ontology (GO) terms from the [GO Database](https://doi.org/10.1038/75556).

The data combined with gene-level meta-survival effects and multi-omics data from The Cancer Genome Atlas Program (TCGA) to construct Cox regression models for each gene. Four z-scores were used, corresponding to four different omics data: Copy Number Variation (CNA), gene expression, DNA methylation, and mutation. These were combined to create a graph of nodes/genes/proteins and edges/protein pairs.

An Evolutionary Scale Modeling (ESM)-1b Transformer from [Rives et al.](https://doi.org/10.1073/pnas.2016239118) was used to extract features from the protein dataset.

### Training
To train the GNN model, make sure all dependencies are satisfied and place any required data sets in the `/data` directory, then run
```
bash HGNN_train.sh
```

This will train the GNN on the aforementioned data with the following parameters:

- `model_filename`: path to the saved model file (checkpoint/deepglint_sampler.pth)
- `knn_k`: number of neighbours (5)
- `levels`: # of levels in hierarchical map (5)
- `hidden`: # of hidden layers (128)
- `epochs`: 200
- `lr`: learning rate (0.01)
- `batch_size`: 64
- `num_conv`: # of convolutions (3)
- `balance`: true
- `use_cluster_feat`: true

The model will be output in the file indicated using `--model_filename` in the bash script. Additionally, the hierarchical map will be output in `hier.csv` in the root directory.

### Testing
To test the GNN model, place any required data sets in the `/data` directory, then run
```
bash HGNN_test.sh
```

This will test the selected GNN model on the aforementioned data with the following parameters:

- `model_filename`: path to the saved model file (checkpoint/deepglint_sampler.pth)
- `knn_k`: number of neighbours (5)
- `tau`: 0.8
- `level`: # of levels in hierarchical map (5)
- `threshold`: prob
- `faiss_gpu`: true
- `hidden`: # of hidden layers (128)
- `num_conv`: # of convolutions (3)
- `batch_size`: 64
- `early_stop`: true
- `use_cluster_feat`: true

## Clinical Hotspots Identification

### Preparation
Clone the HiSig repository from [Zheng et al.](https://doi.org/10.1126/science.abf3067) into the root directory and install [R](https://www.r-project.org/). Any required data sets should be placed in the `/data` directory.

### Mutation/Survival
To identify clinical hotspots based on mutation data from [Zheng et al.](https://doi.org/10.1126/science.abf3067) and survival data from [Smith and Sheltzer](https://doi.org/10.1016%2Fj.celrep.2022.110569), run the following code.
```
bash group_lasso_mutation_survival.sh
```

### DepMap/CMap
To identify clinical hotspots based on dependencies and connectivity mapping using the [DepMap](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5667678/) and [CMap](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5952941/) databases, run the following code.
```
sh group_lasso_DepMap_cMap.sh
```

## Hierarchical Map Construction
A hierarchical map is constructed using [Cytoscape](https://doi.org/10.1186/s13059-019-1758-4). Use the IPython Notebooks in `/annotation` to prepare the data for creation of a Cytoscape network.

The `prepare_network.ipynb` file will produce a .csv input file for Cytoscape. Import a network with the following settings:
- `parent`: source node, number
- `child`: target node, number
- `genes`: target node attribute, list of strings
- `num_nodes`: target node attribute, number

The significant nodes identified using group lasso may be processed using `prepare_hotspot.ipynb`. The resulting .csv file may be used as an additional target node attribute in the existing network. Then, pruning is conducted manually using a bottom-up breadth first search (BFS) method of traversal, where any insignificant non-leaf nodes were removed from the tree (insignificant nodes are defined as any node that is not a clinical hotspot).

### Annotations
Enrichment analysis is performed using [Enrichr](https://maayanlab.cloud/Enrichr/) and the [KEGG 2021 Human](https://www.kegg.jp/) gene-set library to identify functions of protein communities. Enriched terms are grouped based on biological function and the resulting groups (protein systems) are named using biological knowledge and literature analysis.