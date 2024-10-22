import numpy as np
import torch
from datetime import datetime
import os
import pickle
import random

def indices_train_test(ind_0, ind_1, cell_types, spike_trains_, indices_to_look_for, indices_to_look_for2, labeled_ind, unlabeled_ind, seed):

            np.random.seed(seed)
            random.seed(seed)

            ind_test = []
            ind_test_with_session = []
            class_choice = 1
            ind_ = [0, 0]

            ind_0_permuted = [ind_0[j] for j in np.random.permutation(range(len(ind_0)))]
            ind_1_permuted = [ind_1[j] for j in np.random.permutation(range(len(ind_1)))]
            
            dead_cells = []
            for j in range(len(cell_types)):
                if spike_trains_[j,:].sum() == 0:
                    dead_cells.append(j)
            
            dead_and_labeled = []
            for dead in dead_cells:
                if dead in labeled_ind:
                    dead_and_labeled.append(dead)
            
            while len(ind_test)<=45:
                if class_choice == 0:
                    ind_test += ind_0_permuted[ind_[0]]
                    ind_test_with_session.append(ind_0_permuted[ind_[0]])
                    ind_[0] += 1
                    class_choice = 1
                else:
                    ind_test += ind_1_permuted[ind_[1]]
                    ind_test_with_session.append(ind_1_permuted[ind_[1]])
                    ind_[1] += 1
                    class_choice = 0
            
            ind_train = [ind for ind in labeled_ind if (ind not in ind_test)] #and (ind not in ind_valid)]
            ind_train_with_session = [ind_coeff for ind_coeff in ind_0 + ind_1 if ind_coeff not in ind_test_with_session]
            #ind_train = ind_train + ind_test    
            
            for ind_no in dead_and_labeled:
                if ind_no in ind_train:
                    ind_train.remove(ind_no)
                if ind_no in ind_test:
                    ind_test.remove(ind_no)
            
                for l in ind_train_with_session:
                    if ind_no in l:
                        l.remove(ind_no)
            
                for l in ind_test_with_session:
                    if ind_no in l:
                        l.remove(ind_no)

            #folder = "/Users/dorisvoina/Desktop/work_stuff/P3_GNNs/code_ocean"
            #ind_train = np.load(folder + "/results/experiment23/17_10_2024_23_29/ind_train.npy", allow_pickle=True)
            #ind_train_with_session = np.load(folder + "/results/experiment23/17_10_2024_23_29/ind_train_with_session.npy", allow_pickle=True)
            #ind_test = np.load(folder + "/results/experiment23/17_10_2024_23_29/ind_test.npy", allow_pickle=True)
            #ind_test_with_session = np.load(folder + "/results/experiment23/17_10_2024_23_29/ind_test_with_session.npy", allow_pickle=True)
            
            ind_0_sum_train = 0
            ind_1_sum_train = 0
            for ind in indices_to_look_for:
                if ind in ind_train:
                    if ind in indices_to_look_for2[0]:
                        ind_0_sum_train += 1
                    elif ind in indices_to_look_for2[1]:
                        ind_1_sum_train += 1
            
            print(ind_0_sum_train, ind_1_sum_train)
            len(ind_train), len(ind_test)
            
            
            num_train = len(labeled_ind)
            num_total = len(cell_types)
            
            train_idx = ind_train;
            test_idx = ind_test
            #ind_valid #indices[:split], indices[split:] #
            test_idx_labeled = test_idx.copy()
            test_idx = list(test_idx_labeled) + list(unlabeled_ind)
            
            train_mask = torch.tensor([False]*num_total)
            train_mask[train_idx] = True
            test_mask = torch.tensor([False]*num_total)
            test_mask[test_idx] = True
            test_mask_labeled = torch.tensor([False]*num_total)
            test_mask_labeled[test_idx_labeled] = True

            return ind_train, ind_train_with_session, ind_test, ind_test_with_session, ind_0_sum_train, ind_1_sum_train, train_mask, test_mask, test_mask_labeled



def weighing(cell_types, train_mask):

    with_weighing = False
    w = 100
    
    beta = 0.99
    output_dim = 2
    
    per_class = torch.tensor([(cell_types[train_mask]==jj).sum().item() for jj in range(2)])
    #per_class = torch.tensor([(cell_types==jj).sum().item() for jj in range(output_dim)])
    weights_per_class_INS = 1/per_class.double()
    weights_per_class_ISNS = 1/np.sqrt(per_class).double()
    weights_per_class_ENS = (1-beta)/(1-beta**per_class).double()
    
    weights_per_class = weights_per_class_ISNS

    return with_weighing, weights_per_class, output_dim, weights_per_class_INS, weights_per_class_ISNS, weights_per_class_ENS



def save_solution(n_experiment, score_all, train_loss, train_acc, test_score, test_acc, ind_train, ind_test, ind_train_with_session, ind_test_with_session, out_test, pred, batch_y, batch_size, hp, with_weighing, seed):

            now = datetime.now()
            dt_string = now.strftime("%d_%m_%Y_%H_%M")
            print("date and time =", dt_string)
            
            folder = "/Users/dorisvoina/Desktop/work_stuff/P3_GNNs/code_ocean/results/experiment" + str(n_experiment) + "/" + dt_string
            
            if not os.path.exists(folder):
                        os.makedirs(folder)
                
            np.save(folder + "/score_all.npy", np.array(score_all))
            np.save(folder + "/train_loss.npy", np.array([train_loss]))
            np.save(folder + "/train_acc.npy", np.array([train_acc]))
            np.save(folder + "/test_score.npy", np.array([test_score]))
            np.save(folder + "/test_acc.npy", np.array([test_acc]))
            np.save(folder + "/ind_train.npy", ind_train)
            np.save(folder + "/ind_test.npy", ind_test)
            np.save(folder + "/out_test.npy", out_test)
            np.save(folder + "/pred_test.npy", pred)
            np.save(folder + "/real_test.npy", batch_y)
            with open(folder + "/ind_train_with_session.npy", "wb") as fp: 
                    pickle.dump(ind_train_with_session, fp)
            with open(folder + "/ind_test_with_session.npy", "wb") as fp: 
                    pickle.dump(ind_test_with_session, fp)
            
            details = {}
            details["seed"] = seed
            details["features"] = "isis + running filts"
            details["n_features"] = 100
            details["use_graph"] = True
            details["use_directed_graph"] = True
            details["use_edge_weight"] = False
            details["normalize_edge_weight"] = False
            #details["comment"] = "so far we have normalized edge weight"
            
            details["use_edge_ccg"] = False
            details["batchsize"] = batch_size
            details["num_neighbors"] = "[30]*2"
            details["num_neighbors_details"] = "for all"
            details["data loader"] = "NeighborLoader"
            details["model"] = "GraphSAGE"
            details["hidden_dims"] = hp["hidden_dim"]
            details["num_layers"] = hp["num_layers"]
            details["dropout"] = [True, 0.1]
            details["activation"] = "ELU"
            details["batch_norm?"] = [True, "BatchNorm"]
            details["model2?"] = False
            details["optimizer"] = "Adam"
            details["other optimizer_details"] = []
            details["scheduler?"] = [False, []]
            details["with_weighing"] = with_weighing
            details["w"] = None
            details["weights_per_class_"] = None
                
            file = folder + '/meta_dict.pkl'
            with open(file, 'wb') as f:
                pickle.dump(details, f)

            return details
