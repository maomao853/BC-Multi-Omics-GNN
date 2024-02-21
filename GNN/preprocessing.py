import numpy as np
import pandas as pd
import os
import dgl
from dgl.data import DGLDataset
import torch
from sklearn import preprocessing

import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import NNConv, Set2Set


node = np.load(os.path.join(os.getcwd(), 'data', 'feature.npy'), allow_pickle = True)
node = pd.DataFrame.from_dict(node.item()).T
zscore = pd.read_csv(os.path.join(os.getcwd(), 'data', 'stouffer_z_brca.txt'), sep=" ", index_col=0)
mutation= pd.read_csv(os.path.join(os.getcwd(), 'data', 'stouffer_z_brca.txt'), sep=" ", index_col=0)
node_fz = pd.merge(zscore, node, how='inner',left_index=True, right_index=True)
node_fz.insert(0, "Protein code", preprocessing.LabelEncoder().fit_transform(node_fz.index.to_numpy()), True)

PPI = pd.read_csv(os.path.join(os.getcwd(), 'data', 'PPI.tsv'), sep='\t')
PPI = PPI.loc[PPI['Protein 1'].isin(node_fz.index) & PPI['Protein 2'].isin(node_fz.index)]
PPI = pd.merge(pd.merge(PPI,node_fz['Protein code'],how='left',right_index=True, left_on='Protein 1'),
	node_fz['Protein code'],how='left',right_index=True, left_on='Protein 2')


class PPIDataset(DGLDataset):
    def __init__(self,PPI,node_fz):
        super().__init__(name='PPI')
        self.PPI = PPI
        self.node_fz = node_fz

    def process(self):
        nodes_data = node_fz
        edges_data = PPI
        node_features = torch.from_numpy(nodes_data.iloc[:, 2:].to_numpy())
        node_labels = torch.from_numpy(nodes_data['Protein code'].to_numpy())
        edge_features = torch.from_numpy(edges_data['Integrated score'].to_numpy())
        edges_src = torch.from_numpy(edges_data['Protein code_x'].to_numpy())
        edges_dst = torch.from_numpy(edges_data['Protein code_y'].to_numpy())
        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=nodes_data.shape[0])
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels
        self.graph.edata['weight'] = edge_features

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1



dataset = PPIDataset(PPI,node_fz)
graph = dataset[0]
print(graph)
print(graph.ndata['label'])
print(graph.ndata['feat'])
print(graph.edata['weight'])

