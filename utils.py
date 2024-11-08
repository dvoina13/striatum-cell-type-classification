import numpy as np
import torch
from datetime import datetime
import os
import pickle
import random
from utils_sparseLayer import BinaryMask, BinaryGates, ConcreteSelector

def indices_train_test(ind_0, ind_1, cell_types, spike_trains_, indices_to_look_for, indices_to_look_for2, labeled_ind, unlabeled_ind, include_valid, seed):

            np.random.seed(seed)
            random.seed(seed)

            ind_test = []
            ind_test_with_session = []
            ind_valid = []
            ind_valid_with_session = []
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
            
            
            if include_valid == False:
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
            
            else:
                
                while len(ind_valid)<=25:
                    if class_choice == 0:
                        ind_valid += ind_0_permuted[ind_[0]]
                        ind_valid_with_session.append(ind_0_permuted[ind_[0]])
                        ind_[0] += 1
                        class_choice = 1
                    else:
                        ind_valid += ind_1_permuted[ind_[1]]
                        ind_valid_with_session.append(ind_1_permuted[ind_[1]])
                        ind_[1] += 1
                        class_choice = 0
                        
                while len(ind_test)<=35:
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
                        
                        
                        
            ind_train = [ind for ind in labeled_ind if ((ind not in ind_test) and (ind not in ind_valid))]
            ind_train_with_session = [ind_coeff for ind_coeff in ind_0 + ind_1 if (ind_coeff not in ind_test_with_session) and (ind_coeff not in ind_valid_with_session)]
            
            for ind_no in dead_and_labeled:
                if ind_no in ind_train:
                    ind_train.remove(ind_no)
                if ind_no in ind_test:
                    ind_test.remove(ind_no)
                if ind_no in ind_valid:
                    ind_valid.remove(ind_no)
                    
                for l in ind_train_with_session:
                    if ind_no in l:
                        l.remove(ind_no)
            
                for l in ind_test_with_session:
                    if ind_no in l:
                        l.remove(ind_no)
                        
                for l in ind_valid_with_session:
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
            test_idx = ind_test;
            valid_idx = ind_valid;
            #ind_valid #indices[:split], indices[split:] #
            test_idx_labeled = test_idx.copy()
            test_idx = list(test_idx_labeled) + list(unlabeled_ind)
            
            train_mask = torch.tensor([False]*num_total)
            train_mask[train_idx] = True
            valid_mask = torch.tensor([False]*num_total)
            valid_mask[valid_idx] = True
            test_mask = torch.tensor([False]*num_total)
            test_mask[test_idx] = True
            test_mask_labeled = torch.tensor([False]*num_total)
            test_mask_labeled[test_idx_labeled] = True

            return ind_train, ind_train_with_session, ind_valid, ind_valid_with_session, ind_test, ind_test_with_session, ind_0_sum_train, ind_1_sum_train, train_mask, valid_mask, test_mask, test_mask_labeled



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




###functions from persist
def modified_secant_method(x0, y0, y1, x, y):
    '''
    A modified version of secant method, used here to determine the correct lam
    value. Note that we use x = lam and y = 1 / (1 + num_remaining) rather than
    y = num_remaining, because this gives better results.

    The standard secant method uses the two previous points to calculate a
    finite difference rather than an exact derivative (as in Newton's method).
    Here, we used a robustified derivative estimator: we find the curve,
    which passes through the most recent point (x0, y0), that minimizes a
    weighted least squares loss for all previous points (x, y). This improves
    robustness to nearby guesses (small |x - x'|) and noisy evaluations.

    Args:
      x0: most recent x.
      y0: most recent y.
      y1: target y value.
      x: all previous xs.
      y: all previous ys.
    '''
    # Get robust slope estimate.
    weights = 1 / np.abs(x - x0)
    slope = (
        np.sum(weights * (x - x0) * (y - y0)) /
        np.sum(weights * (x - x0) ** 2))

    # Clip slope to minimum value.
    slope = np.clip(slope, a_min=1e-6, a_max=None)

    # Guess x1.
    x1 = x0 + (y1 - y0) / slope
    return x1


def input_layer_penalty(input_layer, m):
    if isinstance(input_layer, BinaryGates):
        return torch.mean(torch.sum(m, dim=1))
    else:
        raise ValueError('only BinaryGates layer has penalty')
        
        
def input_layer_fix(input_layer):
    '''Fix collisions in the input layer.'''
    required_fix = False

    if isinstance(input_layer, (BinaryMask, ConcreteSelector)):
        # Extract logits.
        logits = input_layer._logits
        argmax = torch.argmax(logits, dim=1).cpu().data.numpy()

        # Locate collisions and reinitialize.
        for i in range(len(argmax) - 1):
            if argmax[i] in argmax[i+1:]:
                required_fix = True
                logits.data[i] = torch.randn(
                    logits[i].shape, dtype=logits.dtype, device=logits.device)
        return required_fix

    return required_fix

  
def input_layer_summary(input_layer, n_samples=256):
    '''Generate summary string for input layer's convergence.'''
    with torch.no_grad():
        if isinstance(input_layer, BinaryMask):
            m = input_layer.sample(n_samples)
            mean = torch.mean(m, dim=0)
            sorted_mean = torch.sort(mean, descending=True).values
            relevant = sorted_mean[:input_layer.num_selections]
            return 'Max = {:.2f}, Mean = {:.2f}, Min = {:.2f}'.format(
                relevant[0].item(), torch.mean(relevant).item(),
                relevant[-1].item())

        elif isinstance(input_layer, ConcreteSelector):
            M = input_layer.sample(n_samples)
            mean = torch.mean(M, dim=0)
            relevant = torch.max(mean, dim=1).values
            return 'Max = {:.2f}, Mean = {:.2f}, Min = {:.2f}'.format(
                torch.max(relevant).item(), torch.mean(relevant).item(),
                torch.min(relevant).item())

        elif isinstance(input_layer, BinaryGates):
            m = input_layer.sample(n_samples)
            mean = torch.mean(m, dim=0)
            dist = torch.min(mean, 1 - mean)
            return 'Mean dist = {:.2f}, Max dist = {:.2f}, Num sel = {}'.format(
                torch.mean(dist).item(),
                torch.max(dist).item(),
                int(torch.sum((mean > 0.5).float()).item()))
        

def input_layer_converged(input_layer, tol=1e-2, n_samples=256):
    '''Determine whether the input layer has converged.'''
    with torch.no_grad():
        if isinstance(input_layer, BinaryMask):
            m = input_layer.sample(n_samples)
            mean = torch.mean(m, dim=0)
            return (
                torch.sort(mean).values[-input_layer.num_selections].item()
                > 1 - tol)

        elif isinstance(input_layer, BinaryGates):
            m = input_layer.sample(n_samples)
            mean = torch.mean(m, dim=0)
            return torch.max(torch.min(mean, 1 - mean)).item() < tol

        elif isinstance(input_layer, ConcreteSelector):
            M = input_layer.sample(n_samples)
            mean = torch.mean(M, dim=0)
            return torch.min(torch.max(mean, dim=1).values).item() > 1 - tol
        
        
def save_solution(n_experiment, score_all, train_loss, train_acc, test_score, test_acc, ind_train, ind_test, ind_train_with_session, ind_test_with_session, out_test, pred, batch_y, batch_size, hp, with_weighing, seed, true_inds=None):

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
            details["features"] = "isi histogram + sparsityLayer"
            details["n_features"] = 100
            details["use_graph"] = True
            details["use_directed_graph"] = True
            details["use_edge_weight"] = False
            details["normalize_edge_weight"] = False
            #details["comment"] = "so far we have normalized edge weight"
            
            details["use_edge_ccg"] = False
            details["batchsize"] = batch_size
            details["num_neighbors"] = "[30]*2 (all neighbors)"
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
            details["true_inds"] = true_inds
            
            file = folder + '/meta_dict.pkl'
            with open(file, 'wb') as f:
                pickle.dump(details, f)

            return details
