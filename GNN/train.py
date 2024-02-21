import argparse, time, os, pickle
import numpy as np
import pandas as pd
import dgl
import torch
import torch.optim as optim
from sklearn import preprocessing

from models import LANDER
from dataset import LanderDataset
from sklearn.model_selection import train_test_split


###########
# ArgParser
parser = argparse.ArgumentParser()

# Dataset
#parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--levels', type=str, default='1')
parser.add_argument('--faiss_gpu', action='store_true')
parser.add_argument('--model_filename', type=str, default='lander.pth')

# KNN
parser.add_argument('--knn_k', type=str, default='10')
parser.add_argument('--num_workers', type=int, default=0)

# Model
parser.add_argument('--hidden', type=int, default=512)
parser.add_argument('--num_conv', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.)
parser.add_argument('--gat', action='store_true')
parser.add_argument('--gat_k', type=int, default=1)
parser.add_argument('--balance', action='store_true')
parser.add_argument('--use_cluster_feat', action='store_true')
parser.add_argument('--use_focal_loss', action='store_true')


# Training
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=1e-5)

args = parser.parse_args()
print(args)

###########################
# Environment Configuration
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

##################
# Data Preparation
#with open(args.data_path, 'rb') as f:
#    features, labels = pickle.load(f)
#print("features:", features, features.shape)
#print("labels:", labels, labels.shape, len(np.unique(labels)),np.unique(labels))

def data_path(filename):
    return os.path.join(os.getcwd(), 'data', filename)

node = np.load(data_path('qian_feature.npy'), allow_pickle = True)
node = pd.DataFrame.from_dict(node.item()).T


node_train, node_test_val = train_test_split(node, test_size=0.3, random_state = 67)
node_val, node_test = train_test_split(node_test_val, test_size=2/3, random_state = 67)

node = node_train
#node = node.iloc[0:50]

GO_term = pd.read_csv(data_path('GO_term.csv'), sep=',', index_col=0)

node = pd.merge(node,GO_term,how='inner',left_index=True, right_index=True)

node = node[~node.index.duplicated(keep='first')]

node.insert(0, "label", preprocessing.LabelEncoder().fit_transform(node["term"].to_numpy()), True)


PPI = pd.read_csv(data_path('PPI.tsv'), sep='\t')
PPI = PPI.loc[PPI['Protein 1'].isin(node.index) & PPI['Protein 2'].isin(node.index)]

PPI = pd.merge(pd.merge(PPI,node["label"],how='left',right_index=True, left_on='Protein 1'),
	node["label"],how='left',right_index=True, left_on='Protein 2')



features = node.iloc[:, 1:1281].to_numpy()
edges = PPI.iloc[:, 3:].to_numpy()
labels = node['label'].to_numpy()



k_list = [int(k) for k in args.knn_k.split(',')]
lvl_list = [int(l) for l in args.levels.split(',')]
gs = []
nbrs = []
ks = []
for k, l in zip(k_list, lvl_list):
    dataset = LanderDataset(features=features, edges= edges, labels=labels, k=k,
                            levels=l, faiss_gpu=args.faiss_gpu)
    gs += [g for g in dataset.gs]
    ks += [k for g in dataset.gs]
    nbrs += [nbr for nbr in dataset.nbrs]

gpl = pd.read_csv('global_pred_labels.csv').T
gpl.index = node.index
print('gpl:',gpl)
print('node.index:',node.index)
gpl.to_csv('hier.csv',header=False)
#global_pred_labels.csv


print('Dataset Prepared.')

def set_train_sampler_loader(g, k):
    fanouts = [k-1 for i in range(args.num_conv + 1)]
    sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)
    # fix the number of edges
    train_dataloader = dgl.dataloading.NodeDataLoader(
        g, torch.arange(g.number_of_nodes()), sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers
    )
    return train_dataloader

train_loaders = []
for gidx, g in enumerate(gs):
    train_dataloader = set_train_sampler_loader(gs[gidx], ks[gidx])
    train_loaders.append(train_dataloader)

##################
# Model Definition
feature_dim = gs[0].ndata['features'].shape[1]
model = LANDER(feature_dim=feature_dim, nhid=args.hidden,
               num_conv=args.num_conv, dropout=args.dropout,
               use_GAT=args.gat, K=args.gat_k,
               balance=args.balance,
               use_cluster_feat=args.use_cluster_feat,
               use_focal_loss=args.use_focal_loss)
model = model.to(device)
model.train()

#################
# Hyperparameters
opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                weight_decay=args.weight_decay)

# keep num_batch_per_loader the same for every sub_dataloader
num_batch_per_loader = len(train_loaders[0])
train_loaders = [iter(train_loader) for train_loader in train_loaders]
num_loaders = len(train_loaders)
scheduler = optim.lr_scheduler.CosineAnnealingLR(opt,
                                                 T_max=args.epochs * num_batch_per_loader * num_loaders,
                                                 eta_min=1e-5)

print('Start Training.')

###############
# Training Loop
for epoch in range(args.epochs):
    loss_den_val_total = []
    loss_conn_val_total = []
    loss_val_total = []
    for batch in range(num_batch_per_loader):
        for loader_id in range(num_loaders):
            try:
                minibatch = next(train_loaders[loader_id])
            except:
                train_loaders[loader_id] = iter(set_train_sampler_loader(gs[loader_id], ks[loader_id]))
                minibatch = next(train_loaders[loader_id])
            input_nodes, sub_g, bipartites = minibatch
            sub_g = sub_g.to(device)
            bipartites = [b.to(device) for b in bipartites]
            # get the feature for the input_nodes
            opt.zero_grad()
            output_bipartite = model(bipartites)
            loss, loss_den_val, loss_conn_val = model.compute_loss(output_bipartite)
            loss_den_val_total.append(loss_den_val)
            loss_conn_val_total.append(loss_conn_val)
            loss_val_total.append(loss.item())
            loss.backward()
            opt.step()
            if (batch + 1) % 10 == 0:
                print('epoch: %d, batch: %d / %d, loader_id : %d / %d, loss: %.6f, loss_den: %.6f, loss_conn: %.6f'%
                      (epoch, batch, num_batch_per_loader, loader_id, num_loaders,
                       loss.item(), loss_den_val, loss_conn_val))
            scheduler.step()
    print('epoch: %d, loss: %.6f, loss_den: %.6f, loss_conn: %.6f'%
          (epoch, np.array(loss_val_total).mean(),
           np.array(loss_den_val_total).mean(), np.array(loss_conn_val_total).mean()))
    torch.save(model.state_dict(), args.model_filename)

torch.save(model.state_dict(), args.model_filename)


print('Start validation.')


node = node_val
node = pd.merge(node,GO_term,how='inner',left_index=True, right_index=True)
node = node[~node.index.duplicated(keep='first')]
node.insert(0, "label", preprocessing.LabelEncoder().fit_transform(node["term"].to_numpy()), True)

PPI = pd.read_csv(data_path('PPI.tsv'), sep='\t')
PPI = PPI.loc[PPI['Protein 1'].isin(node.index) & PPI['Protein 2'].isin(node.index)]
PPI = pd.merge(pd.merge(PPI,node["label"],how='left',right_index=True, left_on='Protein 1'),
    node["label"],how='left',right_index=True, left_on='Protein 2')

features = node.iloc[:, 1:1281].to_numpy()
edges = PPI.iloc[:, 3:].to_numpy()
labels = node['label'].to_numpy()

global_features = features.copy()
dataset = LanderDataset(features=features, edges= edges, labels=labels, k=args.knn_k,
                        levels=1, faiss_gpu=args.faiss_gpu)
g = dataset.gs[0]
g.ndata['pred_den'] = torch.zeros((g.number_of_nodes()))
g.edata['prob_conn'] = torch.zeros((g.number_of_edges(), 2))
global_labels = labels.copy()
ids = np.arange(g.number_of_nodes())
global_edges = ([], [])
global_peaks = np.array([], dtype=np.long)
global_edges_len = len(global_edges[0])
global_num_nodes = g.number_of_nodes()


fanouts = [int(args.knn_k)-1 for i in range(args.num_conv + 1)]
sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)
# fix the number of edges
test_loader = dgl.dataloading.NodeDataLoader(
    g, torch.arange(g.number_of_nodes()), sampler,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=args.num_workers
)


feature_dim = g.ndata['features'].shape[1]
model = LANDER(feature_dim=feature_dim, nhid=args.hidden,
                num_conv=args.num_conv, dropout=args.dropout,
                use_GAT=args.gat, K=args.gat_k,
                balance=args.balance,
                use_cluster_feat=args.use_cluster_feat,
                use_focal_loss=args.use_focal_loss)
model.load_state_dict(torch.load(args.model_filename))
model = model.to(device)
model.eval()


num_edges_add_last_level = np.Inf
##################################
# Predict connectivity and density
for level in range(int(args.levels)):
    print(args.levels)
    print('level:', level)
    total_batches = len(test_loader)
    for batch, minibatch in enumerate(test_loader):
        input_nodes, sub_g, bipartites = minibatch
        sub_g = sub_g.to(device)
        bipartites = [b.to(device) for b in bipartites]
        print('bipartites:', bipartites)
        with torch.no_grad():
            output_bipartite = model(bipartites)
        global_nid = output_bipartite.dstdata[dgl.NID]
        global_eid = output_bipartite.edata['global_eid']
        g.ndata['pred_den'][global_nid] = output_bipartite.dstdata['pred_den'].to('cpu')
        g.edata['prob_conn'][global_eid] = output_bipartite.edata['prob_conn'].to('cpu')
        torch.cuda.empty_cache()
        if (batch + 1) % 10 == 0:
            print('Batch %d / %d for inference' % (batch, total_batches))
                
    
    new_pred_labels, peaks,\
        global_edges, global_pred_labels, global_peaks = decode(g, args.tau, args.threshold, args.use_gt,
                                                                ids, global_edges, global_num_nodes,
                                                                global_peaks)
                                                                
    
    ids = ids[peaks]
    new_global_edges_len = len(global_edges[0])
    num_edges_add_this_level = new_global_edges_len - global_edges_len

    global_edges_len = new_global_edges_len
    num_edges_add_last_level = num_edges_add_this_level

    # build new dataset
    features, labels, cluster_features = build_next_level(features, labels, peaks,
                                                          global_features, global_pred_labels, global_peaks)

    knns = build_knns(features, args.knn_k, 'faiss')
            
    dists, nbrs = knns2ordered_nbrs(knns)
    density = density_estimation(dists, nbrs, labels)

    edges = []
    
    adj = fast_knns2spmat(knns, args.knn_k)
    adj, adj_row_sum = row_normalize(adj)
    indices, values, shape = sparse_mx_to_indices_values(adj)
    print('indices.shape:', indices.shape)
    print('indices:', indices)

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
    
    g.apply_edges(lambda edges: {'labels_conn': (edges.src['labels'] == edges.dst['labels']).long()})

    g.apply_edges(lambda edges: {'mask_conn': (edges.src['density'] > edges.dst['density']).bool()})
    
    g.ndata['pred_den'] = torch.zeros((g.number_of_nodes()))
    g.edata['prob_conn'] = torch.zeros((g.number_of_edges(), 2))
    test_loader = dgl.dataloading.NodeDataLoader(
        g, torch.arange(g.number_of_nodes()), sampler,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers
    )

evaluation(global_pred_labels, global_labels, args.metrics)
