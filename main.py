import os

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import json
import pickle 
import sys
import random

import design_matrix
#from hdmf_zarr import NWBZarrIO
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
import networkx as nx
import glob

#from nwbwidgets import nwb2widget

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.data import Data

from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GCN
from torch_geometric.nn import GraphSAGE
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GAT

import seaborn
from datetime import datetime

import MouseNetwork

from load_data import load_data_spike_trains_cells_speed, load_graph, load_filters_waveforms_isis, load_graph_DataLoader, data_loader
from utils import indices_train_test, weighing, save_solution
from model import define_model, define_model_no_graph
from train import train, test, train_no_graph, test_no_graph

from args import parse_args

args = parse_args()
seed = args.seed
print("seed", seed)

np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False

#load datasets to be used as features
spike_trains_, cell_types, running_speeds_, spike_trains_permuted, loaded_average_spikes, experiments, loaded_dict, all_nwb_paths = load_data_spike_trains_cells_speed()
Graph_all, Directed_Graph_all, edge_weights, edge_weights2, ind_0, ind_1, indices_for_new_session, mice = load_graph(all_nwb_paths, loaded_dict, cell_types)
clustering_data, filt_waveforms, filt_isis, filt_firing_rates, spike_filters, running_filters, cell_types__, ISI_features = load_filters_waveforms_isis()
x = np.concatenate((filt_isis, np.array(running_filters)), axis = 1)

#ind_train, ind_test
labeled_ind = np.where(cell_types != -10)[0]
unlabeled_ind = np.where(cell_types == -10)[0]
cell_types_labeled = cell_types[labeled_ind]
indices_to_look_for = [1228, 1231, 1284, 1290, 1298, 1299] + [1280, 1283, 1300, 1306, 1349]
indices_to_look_for2 = [[1228, 1231, 1284, 1290, 1298, 1299], [1280, 1283, 1300, 1306, 1349]]

ind_train, ind_train_with_session, ind_test, ind_test_with_session, ind_0_sum_train, ind_1_sum_train, train_mask, test_mask, test_mask_labeled = indices_train_test(ind_0, ind_1, cell_types, spike_trains_, indices_to_look_for, indices_to_look_for2, labeled_ind, unlabeled_ind, seed)

#for now, don't use GNN's
edge_weights2 = edge_weights2
full_data = Data(x=x, edge_index=Directed_Graph_all, train_mask = train_mask, test_mask = test_mask_labeled, y = cell_types)#, edge_attr = edge_weights2/edge_weights2.max())

#graph data loader
train_loader, train_loader2, test_loader, batch_size = load_graph_DataLoader(cell_types, full_data, seed)
#non-graph data loader
#train_loader, train_loader2, test_loader, batch_size = data_loader(x, cell_types, train_mask, test_mask_labeled, seed)

#hyperparameters
hp = {"batch_size": 64, "num_neighbors": 10, "num_hops": 3, "hidden_dim": 10, "num_layers": 4, "dropout": 0.1, "lr": 0.01, "l2_penalty": 5e-4}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model
model, optimizer, scheduler = define_model(hp)
#model, optimizer, scheduler = define_model_no_graph(hp)

with_weighing, weights_per_class, output_dim, weights_per_class_INS, weights_per_class_ISNS, weights_per_class_ENS = weighing(cell_types, train_mask)

model, score_all, test_acc, test_train_acc_to_see, ignore = train(model, train_loader, train_loader2, test_loader, with_weighing, optimizer, batch_size, output_dim, seed)
train_loss, train_acc, test_score, test_acc, ignore, outputs = test(model, train_loader, train_loader2, test_loader, with_weighing, optimizer, output_dim, seed)
#model, score_all, test_acc, test_train_acc_to_see, batch_size = train_no_graph(model, train_loader, train_loader2, test_loader, with_weighing, optimizer, batch_size, output_dim, seed)
#train_loss, train_acc, test_score, test_acc, batchsize, outputs = test_no_graph(model, train_loader, train_loader2, test_loader, with_weighing, optimizer, output_dim, seed)


out_test, pred, batch_y = outputs
experiment = 29
save_solution(experiment, score_all, train_loss, train_acc, test_score, test_acc, ind_train, ind_test, ind_train_with_session, ind_test_with_session, out_test, pred, batch_y, batch_size, hp, with_weighing, seed)