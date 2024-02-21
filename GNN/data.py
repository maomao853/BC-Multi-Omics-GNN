import numpy as np
import pickle
import csv

import dgl
import torch
from torch.nn.functional import normalize

from utils import (build_knns, fast_knns2spmat, row_normalize, knns2ordered_nbrs,
                   density_estimation, sparse_mx_to_indices_values, l2norm,
                   decode, build_next_level)

class LanderDataset(object):
    def __init__(self, features, edges, labels, cluster_features=None, k=10, levels=1, faiss_gpu=False):
    #def __init__(self, features, labels, cluster_features=None, k=10, levels=1, faiss_gpu=False): 
        self.k = int(k)
        self.gs = []
        self.nbrs = []
        self.dists = []
        self.levels = levels

        # Initialize features and labels
        features = features.astype('float32')
        features = l2norm(features)
        features = features.copy(order='C')
        print("features.flags:",features.flags)
        global_features = features.copy()
        if cluster_features is None:
            cluster_features = features
        global_num_nodes = features.shape[0]
        global_edges = ([], [])
        edges = edges
        global_peaks = np.array([], dtype=np.long)
        ids = np.arange(global_num_nodes)

        # Recursive graph construction
        global_labels = []
        for lvl in range(self.levels):
            print("features.shape[0]",type(features.shape[0]))
            print("self.k",type(self.k))
            if features.shape[0] <= int(self.k):
                self.levels = lvl
                break
            if faiss_gpu:
                knns = build_knns(features, self.k, 'faiss_gpu')
            else:
                knns = build_knns(features, self.k, 'faiss')
            
            dists, nbrs = knns2ordered_nbrs(knns)
            self.nbrs.append(nbrs)
            self.dists.append(dists)
            density = density_estimation(dists, nbrs, labels)

            if lvl == 0:
              edges = edges
              g = self._build_i_graph(features, edges, cluster_features, labels, density, knns)
            else:
              edges = []
              g = self._build_graph(features, edges, cluster_features, labels, density, knns)

            
            self.gs.append(g)

            if lvl >= self.levels - 1:
                break

            # Decode peak nodes
            new_pred_labels, peaks,\
                global_edges, global_pred_labels, global_peaks = decode(g, 0, 'sim', True,
                                                                        ids, global_edges, global_num_nodes,
                                                                        global_peaks)
            ids = ids[peaks]
            features, labels, cluster_features = build_next_level(features, labels, peaks,
                                                                  global_features, global_pred_labels, global_peaks)
            print("lvl:",lvl)
            print("labels:",labels)
            print("labels.shape:",labels.shape)
            print("global_pred_labels:",global_pred_labels)
            print("global_pred_labels.shape:",global_pred_labels.shape)

            global_labels.append(global_pred_labels)
            with open('global_pred_labels.csv', 'w', newline='') as file:
                mywriter = csv.writer(file, delimiter=',')
                mywriter.writerows(global_labels)


    def _build_i_graph(self, features, egs, cluster_features, labels, density, knns):
        
        g = dgl.graph((egs[:,5].astype(int), egs[:,6].astype(int)), num_nodes=features.shape[0])
        g.ndata['features'] = torch.FloatTensor(features)
        g.ndata['cluster_features'] = torch.FloatTensor(features)
        g.ndata['labels'] = torch.LongTensor(labels)
        g.ndata['density'] = torch.FloatTensor(np.zeros(features.shape[0]))
        g.edata['efeat'] = torch.FloatTensor(egs[:,0:5])
        g.edata['affine'] = torch.sum(g.edata['efeat'], dim = 1)

        # A Bipartite from DGL sampler will not store global eid, so we explicitly save it here
        g.edata['global_eid'] = g.edges(form='eid')
        g.edata['raw_affine'] = g.edata['affine']
        #g.edata['labels_conn'] = g.edata['raw_affine'].long()
        g.apply_edges(lambda edges: {'labels_conn': (edges.src['labels'] == edges.dst['labels']).long()})
        g.edata['mask_conn'] = g.edata['labels_conn'].bool()

        return g



    def _build_graph(self, features, egs, cluster_features, labels, density, knns):

        adj = fast_knns2spmat(knns, self.k)
        adj, adj_row_sum = row_normalize(adj)
        indices, values, shape = sparse_mx_to_indices_values(adj)

        g = dgl.graph((indices[1], indices[0]))
        g.ndata['features'] = torch.FloatTensor(features)
        g.ndata['cluster_features'] = torch.FloatTensor(cluster_features)
        g.ndata['labels'] = torch.LongTensor(labels)
        g.ndata['density'] = torch.FloatTensor(density)
        g.edata['affine'] = torch.FloatTensor(values)
        g.edata['efeat'] = torch.FloatTensor(np.tile(values,(5,1)).T)

        # A Bipartite from DGL sampler will not store global eid, so we explicitly save it here
        g.edata['global_eid'] = g.edges(form='eid')
        g.ndata['norm'] = torch.FloatTensor(adj_row_sum)
        g.apply_edges(lambda edges: {'raw_affine': edges.data['affine'] / edges.dst['norm']})
        
        #g.edata['labels_conn'] = g.edata['raw_affine']
        #g.edata['labels_conn'] -=  torch.min(g.edata['labels_conn'])
        #g.edata['labels_conn'] /=  torch.max(g.edata['labels_conn'])
        #g.edata['labels_conn'] *= 1.5
        #g.edata['labels_conn'] = g.edata['labels_conn'].long()

        g.apply_edges(lambda edges: {'labels_conn': (edges.src['labels'] == edges.dst['labels']).long()})
        g.apply_edges(lambda edges: {'mask_conn': (edges.src['density'] > edges.dst['density']).bool()})


        return g

    def __getitem__(self, index):
        assert index < len(self.gs)
        return self.gs[index]

    def __len__(self):
        return len(self.gs)
