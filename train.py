import torch
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from sklearn.metrics import roc_auc_score
from utils import modified_secant_method, input_layer_penalty, input_layer_fix, input_layer_summary, input_layer_converged
from utils_sparseLayer import BinaryMask, BinaryGates, ConcreteSelector
from model import Model_with_GraphSAGE_and_SparsityLayer

def train(model, train_loader, train_loader2, test_loader, with_weighing, optimizer, batch_size, output_dim, seed):
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
    
            logs = {}

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            real_class_j = np.zeros(output_dim)
            true_positive_j = np.zeros(output_dim)
            true_negative_j = np.zeros(output_dim)
            false_positive_j = np.zeros(output_dim)
            false_negative_j = np.zeros(output_dim)
            score_all = []; test_acc = []
            test_train_acc_to_see = []
            score_all_confident = []
            test_acc_confident = []
            
            for epoch in range(500):
                # Train
                model.train()
            
                print("EPOCH", epoch)
                
                # Iterate through the loader to get a stream of subgraphs instead of the whole graph
                for bid, batch in enumerate(train_loader):
                    batchsize = batch.x.shape[0]
                    print("batch_size", batchsize)
                    batch.to(device)

                    if batchsize==1:
                        continue
                    # Forward pass
                    print(batch.x.shape)
                    out = model(batch.x.float(), batch.edge_index).float()
                    #out = model2(out)
                    #print("out shape", out.shape, out)
            
                    pred = out.argmax(dim=1)
            
                    print("pred", pred[batch.train_mask].shape, pred[batch.train_mask])
                    print("real", batch.y[batch.train_mask].shape, batch.y[batch.train_mask])
                    #print("out", out)
            
                    # Calculate loss
                    if with_weighing == True:
                        loss = F.cross_entropy(out[batch.train_mask], batch.y[batch.train_mask], weight = w*weights_per_class_ISNS.float())
                    else:
                        loss = F.cross_entropy(out[batch.train_mask], batch.y[batch.train_mask])
            
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
                    print("how many in train?", (batch.train_mask == 1).sum())
                    print("loss", loss)
                    
                    # Predict on training data
                    with  torch.no_grad():
                        pred = out.argmax(dim=1)
                        epoch_train_acc = (pred[batch.train_mask] == batch.y[batch.train_mask]).sum()/batch.train_mask.sum()
            
                    test_train_acc_to_see.append(epoch_train_acc)
            
                    # Log training status after each batch
                    logs["loss"] = loss.item()
                    logs["acc"] = epoch_train_acc
                    print(
                        "Epoch {}, Train Batch {}, Loss {:.4f}, Accuracy {:.4f}".format(
                            epoch, bid, logs["loss"], logs["acc"]
                        )
                    )
                  
                # Evaluate
                model.eval()
            
                #for batch in train_loader2:
                for batch in test_loader:
                    batchsize = batch.x.shape[0]
                    batch.to(device)

                    with torch.no_grad():
                        # Forward pass
                        out = model(batch.x.float(), batch.edge_index)
                        
                        # Calculate loss 
                        if with_weighing == True:
                            valid_loss = F.cross_entropy(out[batch.test_mask], batch.y[batch.test_mask], weight = w*weights_per_class_ISNS.float())
                        else:
                            valid_loss = F.cross_entropy(out[batch.test_mask], batch.y[batch.test_mask])
            
                        # Prediction
                        pred = out.argmax(dim=1)
                        
                        out_max = out.max(dim=1)[0]
                        ind_confident = np.where((out_max[batch.test_mask]>=2))[0]
                        pred_confident = pred[batch.test_mask][ind_confident]
                        real_class_confident = batch.y[batch.test_mask][ind_confident]

                        print("pred", pred[batch.test_mask])
                        print("real", batch.y[batch.test_mask])
                        print("pred-real", np.abs(pred[batch.test_mask] - batch.y[batch.test_mask]))

                        print("ind_confident", ind_confident)
                        print("pred_confident", pred_confident)
                        print("real_class_confident", real_class_confident)

                        print("len pred versus len pred_confident", len(pred[batch.test_mask]), len(pred_confident))

                        epoch_val_acc = (pred[batch.test_mask]==batch.y[batch.test_mask]).sum()/(batch.test_mask==True).sum()
                        epoch_val_acc_confident = (pred_confident==real_class_confident).sum()/len(ind_confident)
                        print("epoch_val_acc_confident", epoch_val_acc_confident)
                        score = roc_auc_score(batch.y[batch.test_mask], pred[batch.test_mask])
                        try:
                            score_confident = roc_auc_score(real_class_confident, pred_confident)
                        except:
                            score_confident = None
                        print("score_confident", score_confident)
                        
                        if score>0.7:
                            torch.save(model.state_dict(), "GraphSAGE_61_features.pt")
                            
                        print("score", score)
                        for j in range(output_dim):
                            real_class_j[j] += (batch.y[batch.test_mask] == j).sum()
            
                            ind_true_j = np.where(batch.y[batch.test_mask] == j)[0]
                            ind_false_j = np.where(batch.y[batch.test_mask] != j)[0]
            
                            true_positive_j[j] += (pred[batch.test_mask][ind_true_j] == j).sum()
                            false_negative_j[j] += (pred[batch.test_mask][ind_true_j] != j).sum()
            
                            true_negative_j[j] += (pred[batch.test_mask][ind_false_j] != j).sum()
                            false_positive_j[j] += (pred[batch.test_mask][ind_false_j] == j).sum()
            
                score_all.append(score);
                test_acc.append(epoch_val_acc)

                score_all_confident.append(score_confident)
                test_acc_confident.append(epoch_val_acc_confident)
                
                # Log testing result after each epoch
                logs["val_loss"] = valid_loss.item()
                logs["val_acc"] = epoch_val_acc
                print(
                    "Epoch {}, Valid Loss {:.4f}, Valid Accuracy {:.4f}".format(
                        epoch, logs["val_loss"], logs["val_acc"]
                    )
                )
                        
                for batch in train_loader2:
                    batchsize = batch.x.shape[0]
                    batch.to(device)
                    with torch.no_grad():
                        # Forward pass
                        out = model(batch.x.float(), batch.edge_index)
                        #out = model2(out)
                        # Calculate loss
                        
                        if with_weighing == True:
                            valid_loss = F.cross_entropy(out[batch.train_mask], batch.y[batch.train_mask], weight = w*weights_per_class_ISNS.float())
                        else:
                            valid_loss = F.cross_entropy(out[batch.train_mask], batch.y[batch.train_mask])
            
                        # Prediction
                        pred = out.argmax(dim=1)
                        print("pred", pred[batch.train_mask])
                        print("real", batch.y[batch.train_mask])
                        print("pred-real", np.abs(pred[batch.train_mask] - batch.y[batch.train_mask]))
                       
                        epoch_val_acc = (pred[batch.train_mask]==batch.y[batch.train_mask]).sum()/(batch.train_mask==True).sum()
                        score = roc_auc_score(batch.y[batch.train_mask], pred[batch.train_mask])
                        #epoch_val_acc = (pred[batch.test_mask]==batch.y[batch.test_mask]).sum()/(batch.test_mask==True).sum()
                        #score = roc_auc_score(batch.y[batch.test_mask], pred[batch.test_mask])
            
                        print("score", score)
                        for j in range(output_dim):
                            real_class_j[j] += (batch.y[batch.test_mask] == j).sum()
            
                            ind_true_j = np.where(batch.y[batch.test_mask] == j)[0]
                            ind_false_j = np.where(batch.y[batch.test_mask] != j)[0]
            
                            true_positive_j[j] += (pred[batch.test_mask][ind_true_j] == j).sum()
                            false_negative_j[j] += (pred[batch.test_mask][ind_true_j] != j).sum()
            
                            true_negative_j[j] += (pred[batch.test_mask][ind_false_j] != j).sum()
                            false_positive_j[j] += (pred[batch.test_mask][ind_false_j] == j).sum()
                            
                    # Log testing result after each epoch
                    logs["val_loss"] = valid_loss.item()
                    logs["val_acc"] = epoch_val_acc
                    print(
                    "TRAIN Epoch {}, Valid Loss {:.4f}, Valid Accuracy {:.4f}".format(
                        epoch, logs["val_loss"], logs["val_acc"]
                    )
                )
            
                model.train()
                #scheduler.step()
            
            return model, score_all, test_acc, test_train_acc_to_see, score_all_confident, test_acc_confident, batch_size



def test(model, train_loader, train_loader2, test_loader, with_weighing, optimizer, output_dim, seed):
            model.eval()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logs = {}

            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)

            real_class_j = np.zeros(output_dim)
            true_positive_j = np.zeros(output_dim)
            true_negative_j = np.zeros(output_dim)
            false_positive_j = np.zeros(output_dim)
            false_negative_j = np.zeros(output_dim)
    
            for batch in train_loader2:
                    batchsize = batch.x.shape[0]
                    batch.to(device)
                    with torch.no_grad():
                        # Forward pass
                        out = model(batch.x.float(), batch.edge_index)
                        
                        # Calculate loss                        
                        if with_weighing == True:
                            valid_loss = F.cross_entropy(out[batch.train_mask], batch.y[batch.train_mask], weight = w*weights_per_class_ISNS.float())
                        else:
                            valid_loss = F.cross_entropy(out[batch.train_mask], batch.y[batch.train_mask])
                            #valid_loss = F.cross_entropy(out[batch.test_mask], batch.y[batch.test_mask])
            
                        # Prediction
                        pred = out.argmax(dim=1)
                        print("pred", pred[batch.train_mask])
                        print("real", batch.y[batch.train_mask])
                        print("pred-real", np.abs(pred[batch.train_mask] - batch.y[batch.train_mask]))
                        #print("pred", pred[batch.test_mask])
                        #print("real", batch.y[batch.test_mask])
                        #print("pred-real", np.abs(pred[batch.test_mask] - batch.y[batch.test_mask]))

                        epoch_val_acc = (pred[batch.train_mask]==batch.y[batch.train_mask]).sum()/(batch.train_mask==True).sum()
                        score = roc_auc_score(batch.y[batch.train_mask], pred[batch.train_mask])

                        #epoch_val_acc = (pred[batch.test_mask]==batch.y[batch.test_mask]).sum()/(batch.test_mask==True).sum()
                        #score = roc_auc_score(batch.y[batch.test_mask], pred[batch.test_mask])
            
                        print("score", score)
                        for j in range(output_dim):
                            real_class_j[j] += (batch.y[batch.test_mask] == j).sum()
            
                            ind_true_j = np.where(batch.y[batch.test_mask] == j)[0]
                            ind_false_j = np.where(batch.y[batch.test_mask] != j)[0]
            
                            true_positive_j[j] += (pred[batch.test_mask][ind_true_j] == j).sum()
                            false_negative_j[j] += (pred[batch.test_mask][ind_true_j] != j).sum()
            
                            true_negative_j[j] += (pred[batch.test_mask][ind_false_j] != j).sum()
                            false_positive_j[j] += (pred[batch.test_mask][ind_false_j] == j).sum()
                            
                    # Log testing result after each epoch
                    logs["val_loss"] = valid_loss.item()
                    logs["val_acc"] = epoch_val_acc
                    print(
                    "TRAIN overall Valid Loss {:.4f}, Valid Accuracy {:.4f}".format(
                           logs["val_loss"], logs["val_acc"]
                    )
                )
            
            train_loss = logs["val_loss"]
            train_acc = logs["val_acc"]
            
            for batch in test_loader:
                    batchsize = batch.x.shape[0]
                    batch.to(device)
                    with torch.no_grad():
                        # Forward pass
                        out_test = model(batch.x.float(), batch.edge_index)
                        
                    pred = out_test.argmax(dim=1)
            
                    print("pred", pred[batch.test_mask])
                    print("real", batch.y[batch.test_mask])
                    print("pred-real", np.abs(pred[batch.test_mask] - batch.y[batch.test_mask]))

                    out_max = out_test.max(dim=1)[0]
                    ind_confident = np.where((out_max[batch.test_mask]>=0.75))[0]
                    pred_confident = pred[batch.test_mask][ind_confident]
                    real_class_confident = batch.y[batch.test_mask][ind_confident]

                    print("pred_confident", pred_confident)
                    print("real_confident", real_class_confident)
                    print("pred_confident-real_confident", pred_confident - real_class_confident)

                    print("len pred versus len pred_confident", len(pred[batch.test_mask]), len(pred_confident))
                    epoch_val_acc = (pred[batch.test_mask]==batch.y[batch.test_mask]).sum()/(batch.test_mask==True).sum()
                    epoch_val_acc_confident = (pred_confident==real_class_confident).sum()/len(ind_confident)
                    
                    score = roc_auc_score(batch.y[batch.test_mask], pred[batch.test_mask])
                    try:
                        score_confident = roc_auc_score(real_class_confident, pred_confident)
                    except:
                        score_confident = None
                        
                    print("test acc", "score", epoch_val_acc.item(), score)
            
            
            test_acc = epoch_val_acc.item()
            test_score = score
            outputs = [out_test, pred[batch.test_mask],  batch.y[batch.test_mask]]
        
            return train_loss, train_acc, test_score, test_acc, epoch_val_acc_confident, score_confident, batchsize, outputs

        
def train_with_sparsityLayer_eliminate(args, hp, model, train_loader, train_loader2, valid_loader, test_loader, with_weighing, optimizer, batch_size, seed, target, output_dim=2, lam_init=0.0001, mbsize=64, tol=0.2, start_temperature=10.0, end_temperature=0.01, lookback=10, max_trials=10):
              
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
            
    preselected = model.preselected_inds
    input_size = model.input_size
    output_size = model.output_size
    
    candidates = np.array([i for i in range(input_size) if i not in preselected])
    included = np.sort(np.concatenate([candidates, preselected]))
    preselected_relative = np.array([np.where(included == ind)[0][0] for ind in preselected])
    
    set_features(model, None)
    
    lam_list = [0]
    num_remaining = input_size
    num_remaining_list = [num_remaining]
    lam = lam_init
    trials = 0
    score_ALL = []
        
   
    # Iterate until num_remaining is near the target value.
    while np.abs(num_remaining - target) > target * tol:
        # Ensure not done.
        if trials == max_trials:
            #raise ValueError(
            #        'reached maximum number of trials without selecting the '
            #        'desired number of features! The results may have large '
            #        'variance due to small dataset size, or the initial lam '
            #        'value may be bad')
            break
            
        trials += 1
    
        model, train_loss_arr, valid_loss_arr, train_acc_arr, valid_acc_arr, score_all, outputs = fit(model, optimizer, with_weighing, train_loader, train_loader2, valid_loader, test_loader, start_temperature, end_temperature, lam, lookback, batch_size, nn.CrossEntropyLoss())
        
        score_ALL += score_all
        
        inds = model.model_input_layer.get_inds(threshold=0.5)
        print("inds", len(inds))

        num_remaining = len(inds)
        print(f'lam = {lam:.6f} yielded {num_remaining} features')
            
        if np.abs(num_remaining - target) <= target * tol:
                print(f'done, lam = {lam:.6f} yielded {num_remaining} features')

        else:
                # Guess next lam value.
                next_lam = modified_secant_method(
                    lam, 1 / (1 + num_remaining), 1 / (1 + target),
                    np.array(lam_list), 1 / (1 + np.array(num_remaining_list)))

                # Clip lam value for stability.
                next_lam = np.clip(next_lam, a_min=0.1 * lam, a_max=10 * lam)

                # Possibly reinitialize model.
                if num_remaining < target * (1 - tol):
                    # BinaryGates layer is not great at allowing features
                    # back in after inducing too much sparsity.
                    print('Reinitializing model for next iteration')
                    
                    model = Model_with_GraphSAGE_and_SparsityLayer(args, hp, input_size, output_size, preselected_inds = 
                                                                   preselected_relative)
                    model_graph, model_input_layer, optimizer, scheduler = model.return_models()
                else:
                    print('Warm starting model for next iteration')

                # Prepare for next iteration.
                lam_list.append(lam)
                num_remaining_list.append(num_remaining)
                lam = next_lam
                print(f'next attempt is lam = {lam:.6f}')
    
    # Set eligible features.
    true_inds = candidates[inds]
    set_features(model, true_inds)
   
    return true_inds, model, score_ALL, train_loss_arr, train_acc_arr, score_ALL, valid_acc_arr, outputs

def set_features(model, candidates = None):
    
    if candidates is None:
            # All features but pre-selected ones.
            candidates = np.array(
                [i for i in range(model.max_input_size)
                 if i not in model.preselected_inds])
        
    else:
            # Ensure that candidates do not overlap with pre-selected features.
            assert len(np.intersect1d(candidates, model.preselected_inds)) == 0
        

    # Set features in datasets.
    included = np.sort(np.concatenate([candidates, model.preselected_inds]))
    model.included = included
    #self.train.set_inds(included)
    #self.val.set_inds(included)

    # Set relative indices for pre-selected features.
    model.preselected_relative = np.array(
            [np.where(included == ind)[0][0] for ind in model.preselected_inds])


def train_with_sparsityLayer_select(num_select, args, hp, model, train_loader, train_loader2, valid_loader, test_loader, with_weighing, optimizer, batch_size, seed, start_temperature=10.0, end_temperature=0.01, lookback=10):
                                    
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    set_features(model, None)
    input_size = model.input_size
    output_size = model.output_size
    #model.included = np.arange(1, input_size)
    included_inds = np.sort(model.included)
    
    args.input_layer = "binary_mask"    
    print("num_selections", num_select)
    new_model = Model_with_GraphSAGE_and_SparsityLayer(args, hp, input_size, output_size, num_selections=num_select, preselected_inds = model.preselected_relative)
    model_graph, model_input_layer, optimizer, scheduler = new_model.return_models()

    set_features(new_model, included_inds)

    new_model, train_loss_arr, valid_loss_arr, train_acc_arr, valid_acc_arr, new_score, outputs = fit(new_model, optimizer, with_weighing, train_loader, train_loader2, valid_loader, test_loader, start_temperature, end_temperature, 0, lookback, batch_size, nn.CrossEntropyLoss())
        
    true_inds = new_model.model_input_layer.get_inds()
    #print(f'done, selected {len(true_inds)} genes')
    return true_inds, new_model, new_score


def fit(model, optimizer, with_weighing, train_loader, train_loader2, valid_loader, test_loader, start_temperature, end_temperature, lam, lookback, mbsize, loss_fn = nn.CrossEntropyLoss(), eta=0, max_nepochs = 500, verbose=True):
              
        
        if lam != 0:
            if not isinstance(model.model_input_layer, BinaryGates):
                raise ValueError('lam should only be specified when using '
                                 'BinaryGates layer')
        else:
            if isinstance(model.model_input_layer, BinaryGates):
                raise ValueError('lam must be specified when using '
                                 'BinaryGates layer')
                
        if eta > 0:
            if isinstance(model.model_input_layer, BinaryGates):
                raise ValueError('lam cannot be specified when using '
                                  'BinaryGates layer')

        if end_temperature > start_temperature:
            raise ValueError('temperature should be annealed downwards, must '
                             'have end_temperature <= start_temperature')
        elif end_temperature == start_temperature:
            loss_early_stopping = True
        else:
            loss_early_stopping = False

        # Set up optimizer.
        device = next(model.parameters()).device
        
        model.model_input_layer.temperature = start_temperature
        r = np.power(end_temperature / start_temperature,
                     1 / ((len(train_loader) // mbsize) * max_nepochs))

        # For tracking loss.
        train_loss_arr = []
        valid_loss_arr = []
        train_acc_arr = []
        valid_acc_arr = []
        score_all = []
        
        best_loss = np.inf
        best_epoch = -1
        
    
        # Begin training.
        for epoch in range(max_nepochs):
            # For tracking mean train loss.
            model.train()
            
            for n, p in model.model_input_layer.named_parameters():
                p.requires_grad = False
            for n, p in model.model_graph.named_parameters():
                p.requires_grad = True
                
            print("epoch", epoch)
            train_loss = 0
            N = 0
            
            if epoch % 5 != 0:
                
                for bid, batch in enumerate(train_loader):
                    batchsize = batch.x.shape[0]
                    batch.to(device)
                
                    if batchsize==1:
                        continue
                        
                    x = batch.x.float().to(device)
                    y = batch.y.to(device)
                    edge_index = batch.edge_index.to(device)
                    
                    #ind_zero
                    ind_zero = np.array([True] * x.shape[1])
                    ind_zero[model.included.astype(int)] = False
                    x[:, ind_zero] = 0
                
                    # Calculate loss.
                    out, x, m = model(x, edge_index)
                    
                    #print("out", torch.abs(out).max())
                    #for name, param in model.model_input_layer.named_parameters():
                    #    print("grad", name, param.requires_grad, param.grad)
                    
                    #for name, param in model.model_graph.named_parameters():
                    #    print("grad", name, param.requires_grad, param.grad)
                    
                    #print("are weights really changing?", model.model_graph.convs[0].lin_l.weight[0,0:5])
                    if with_weighing == True:
                        loss = loss_fn(out[batch.train_mask], batch.y[batch.train_mask], weight = w*weights_per_class_ISNS.float())
                    else:
                        loss = loss_fn(out[batch.train_mask], batch.y[batch.train_mask])
                
                    # Calculate penalty if necessary.
                    #if lam > 0:
                    #    penalty = input_layer_penalty(model.model_input_layer, m)
                    #    loss = loss + lam * penalty
                    
                    # Add expression penalty if necessary -- not sure if this is needed
                    #if eta > 0:
                    #    expressed = torch.mean(torch.sum(x, dim=1))
                    #    loss = loss + eta * expressed

                    # Update mean train loss.
                    train_loss = (
                    (N * train_loss + batchsize * loss.item()) / (N + batchsize))
                    N += batchsize
                
                    #print("temperature", r, model.model_input_layer.temperature)
                    #print("r", r, end_temperature, start_temperature, len(train_loader), mbsize, max_nepochs)

                    # Gradient step.
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    #print("logits after", model.model_input_layer.probs)

                    with  torch.no_grad():
                        pred = out.argmax(dim=1)
                        train_acc = (pred[batch.train_mask] == y[batch.train_mask]).sum()/batch.train_mask.sum()
                        train_acc_arr.append(train_acc)
                    
                    # For tracking mean train loss.
                if isinstance(model.model_input_layer, BinaryGates):
                    inds = model.model_input_layer.get_inds(threshold=0.5)
                else:
                    inds = model.model_input_layer.get_inds()
                print("len inds", len(inds))
                print("len included", len(model.included))
                
                
                # Check progress.
                with torch.no_grad():
                    # Calculate loss.
                    model.eval()
                    train_or_valid = True
                    outputs, val_loss, epoch_val_acc, score = validate(
                    model, test_loader, train_loader2, train_or_valid, with_weighing, loss_fn, lam, eta)
                    val_loss = val_loss.item()
                    model.train()

                    # Record loss.
                    train_loss_arr.append(train_loss)
                    valid_loss_arr.append(val_loss)
                    valid_acc_arr.append(epoch_val_acc)
                    score_all.append(score)
                
                    print("SCORE", score)
                    if verbose:
                        print(f'{"-" * 8}Epoch = {epoch + 1}{"-" * 8}')
                        print(f'Train loss = {train_loss:.4f}')
                        print(f'Val loss = {val_loss:.4f}')
                        #if eta > 0:
                        #    print(f'Mean expressed features = {val_expressed:.4f}')
                        print(input_layer_summary(model.model_input_layer))
                    
                   
            else:    
                    
                model.train()
            
                for n, p in model.model_input_layer.named_parameters():
                    p.requires_grad = True
                for n, p in model.model_graph.named_parameters():
                    p.requires_grad = False
            
                for bid, batch in enumerate(valid_loader):
                    batchsize = batch.x.shape[0]
                    batch.to(device)
                
                    if batchsize==1:
                        continue
                        
                    x = batch.x.float().to(device)
                    y = batch.y.to(device)
                    edge_index = batch.edge_index.to(device)
                
                    ind_zero = np.array([True] * x.shape[1])
                    ind_zero[model.included.astype(int)] = False
                    x[:, ind_zero] = 0
                    
                    # Calculate loss.
                    out, x, m = model(x, edge_index)
                
                    #print("out", torch.abs(out).max())
                    #for name, param in model.model_input_layer.named_parameters():
                    #    print("grad", name, param.requires_grad, param.grad)
                    
                    #for name, param in model.model_graph.named_parameters():
                    #    print("grad", name, param.requires_grad, param.grad)
                
                    #print("are weights really changing?", model.model_graph.convs[0].lin_l.weight[0,0:5])
                    if with_weighing == True:
                        loss = loss_fn(out[batch.val_mask], batch.y[batch.val_mask], weight = w*weights_per_class_ISNS.float())
                    else:
                        loss = loss_fn(out[batch.val_mask], batch.y[batch.val_mask])
                
                    # Calculate penalty if necessary.
                    if lam > 0:
                        penalty = input_layer_penalty(model.model_input_layer, m)
                        loss = loss + lam * penalty
                    
                    # Add expression penalty if necessary -- not sure if this is needed
                    if eta > 0:
                        expressed = torch.mean(torch.sum(x, dim=1))
                        loss = loss + eta * expressed

                    # Update mean train loss.
                    train_loss = ((N * train_loss + batchsize * loss.item()) / (N + batchsize))
                    N += batchsize
                
                    #print("temperature", r, model.model_input_layer.temperature)
                    #print("r", r, end_temperature, start_temperature, len(train_loader), mbsize, max_nepochs)

                    # Gradient step.
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    #print("logits after", model.model_input_layer.probs)
        
                    with  torch.no_grad():
                         pred = out.argmax(dim=1)
                         train_acc = (pred[batch.val_mask] == y[batch.val_mask]).sum()/batch.val_mask.sum()
                         train_acc_arr.append(train_acc)
                    
                    #print("train_acc", train_acc)
                    # Adjust temperature.
               
                    #print("pred", pred[batch.train_mask])
                    #print("y", y[batch.train_mask])
                    #print("train_acc", train_acc)
                    model.model_input_layer.temperature *= r
                
                if isinstance(model.model_input_layer, BinaryGates):
                    inds = model.model_input_layer.get_inds(threshold=0.5)
                    print("helllloooo!!", len(inds))
                else:
                    inds = model.model_input_layer.get_inds()
                print("len inds", len(inds))
                
                # Check progress.
                with torch.no_grad():
                    # Calculate loss.
                    model.eval()
                    train_or_valid = False
                    outputs, val_loss, epoch_val_acc, score = validate(
                    model, test_loader, valid_loader, train_or_valid, with_weighing, loss_fn, lam, eta)
                    val_loss = val_loss.item()
                    model.train()

                    # Record loss.
                    train_loss_arr.append(train_loss)
                    valid_loss_arr.append(val_loss)
                    valid_acc_arr.append(epoch_val_acc)
                    score_all.append(score)
                
                    print("SCORE", score)
                    if verbose:
                        print(f'{"-" * 8}Epoch = {epoch + 1}{"-" * 8}')
                        print(f'Train loss = {train_loss:.4f}')
                        print(f'Val loss = {val_loss:.4f}')
                        #if eta > 0:
                        #    print(f'Mean expressed features = {val_expressed:.4f}')
                        print(input_layer_summary(model.model_input_layer))
                    
                # Fix input layer if necessary.
                required_fix = input_layer_fix(model.model_input_layer)

                if not required_fix:
                   # Stop early if input layer is converged.
                   if input_layer_converged(model.model_input_layer, n_samples=mbsize):
                        if verbose:
                            print('Stopping early: input layer converged')
                        break

                   # Stop early if loss converged.
                   if loss_early_stopping:
                        if val_loss < best_loss:
                            best_loss = val_loss
                            best_epoch = epoch
                        elif (epoch - best_epoch) == lookback:            
                            if verbose:
                                print('Stopping early: loss converged')
                            break
            
        return model, train_loss_arr, valid_loss_arr, train_acc_arr, valid_acc_arr, score_all, outputs


def validate(model, valid_loader, train_loader2, train_or_valid, with_weighing, loss_fn, lam, eta):
        '''Calculate average loss.'''
        device = next(model.parameters()).device
        mean_valid_loss = 0
        mean_expressed = 0
        N = 0
        
        print("TRAIN ACC")
        train_acc_total = []
        for bid, batch in enumerate(train_loader2):
            batchsize = batch.x.shape[0]
            #print("batch_size", batchsize)
            batch.to(device)

            if batchsize==1:
                continue
                        
            x = batch.x.float().to(device)
            y = batch.y.to(device)
            edge_index = batch.edge_index.to(device)
            
            ind_zero = np.array([True] * x.shape[1])
            ind_zero[model.included.astype(int)] = False
            x[:, ind_zero] = 0
                    
            out, x, m = model(x, edge_index)
            pred = out.argmax(dim=1)
            if train_or_valid:
                epoch_train_acc = (pred[batch.train_mask]==y[batch.train_mask]).sum()/(batch.train_mask==True).sum()
            else:
                epoch_train_acc = (pred[batch.val_mask]==y[batch.val_mask]).sum()/(batch.val_mask==True).sum()
            train_acc_total.append(epoch_train_acc)        
            
        print("epoch_train_acc", np.array(train_acc_total).mean())
            
        print("VALIDATE")
        for bid, batch in enumerate(valid_loader):
                
                batchsize = batch.x.shape[0]
                #print("batch_size", batchsize)
                batch.to(device)
                
                if batchsize==1:
                        continue
                        
                x = batch.x.float().to(device)
                y = batch.y.to(device)
                #x[:, model.included] = 0
                edge_index = batch.edge_index.to(device)
                n = len(x)
                
                ind_zero = np.array([True] * x.shape[1])
                ind_zero[model.included.astype(int)] = False
                x[:, ind_zero] = 0
            
                print("valid isnan?!", torch.isnan(x).sum(), torch.isnan(y).sum(), torch.isnan(edge_index).sum())

                # Calculate loss.
                out, x, m = model(x, edge_index)
                
                print("out valid", out.max())
                if with_weighing == True:
                     valid_loss = loss_fn(out[batch.test_mask], y[batch.test_mask], weight = w*weights_per_class_ISNS.float())
                else:
                     valid_loss = loss_fn(out[batch.test_mask], y[batch.test_mask])
                
                print("valid_loss", valid_loss)
                # Add penalty term.
                if lam > 0:
                    penalty = input_layer_penalty(model.model_input_layer, m)
                    valid_loss = valid_loss + lam * penalty
                
                print("valid loss with penalty", valid_loss)
                # Add expression penalty term.
                #expressed = torch.mean(torch.sum(x, dim=1))
                #if eta > 0:
                #    loss = loss + eta * expressed
                
                pred = out.argmax(dim=1)
                epoch_val_acc = (pred[batch.test_mask]==y[batch.test_mask]).sum()/(batch.test_mask==True).sum()
                score = roc_auc_score(y[batch.test_mask], pred[batch.test_mask])
                    
                print("acc + score", epoch_val_acc, score)
                mean_valid_loss = (N * mean_valid_loss + n * valid_loss) / (N + n)
                #mean_expressed = (N * mean_expressed + n * expressed) / (N + n)
                N += n
                
                print(f"Accuracy {epoch_val_acc} and score {score}")
                
        outputs = [out, pred[batch.test_mask],  y[batch.test_mask]]
        return outputs, mean_valid_loss, epoch_val_acc, score


def train_no_graph(model, train_loader, train_loader2, test_loader, with_weighing, optimizer, batch_size, output_dim, seed):
            
            logs = {}
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
    
            real_class_j = np.zeros(output_dim)
            true_positive_j = np.zeros(output_dim)
            true_negative_j = np.zeros(output_dim)
            false_positive_j = np.zeros(output_dim)
            false_negative_j = np.zeros(output_dim)
            score_all = []; test_acc = []
            test_train_acc_to_see = []
            
            for epoch in range(500):
                # Train
                model.train()
                model.float()

                
                print("EPOCH", epoch)
                
                # Iterate through the loader to get a stream of subgraphs instead of the whole graph
                for bid, batch in enumerate(train_loader):
                    
                    x, y = batch
                    x.to(device); y.to(device);
                    batch_size = x.shape[0]

                    # Forward pass
                    out = model(x.float())          
                    pred = out.argmax(dim=1)
            
                    print("pred", pred.shape, pred.shape)
                    print("real", y)
            
                    # Calculate loss
                    if with_weighing == True:
                        loss = F.cross_entropy(out, y, weight = w*weights_per_class_ISNS.float())
                    else:
                        loss = F.cross_entropy(out, y)
            
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
                    print("how many in train?", len(x))
                    print("loss", loss)
                    
                    # Predict on training data
                    with  torch.no_grad():
                        pred = out.argmax(dim=1)
                        epoch_train_acc = ((pred == y).sum())/len(y)
            
                    test_train_acc_to_see.append(epoch_train_acc)
            
                    # Log training status after each batch
                    logs["loss"] = loss.item()
                    logs["acc"] = epoch_train_acc
                    print(
                        "Epoch {}, Train Batch {}, Loss {:.4f}, Accuracy {:.4f}".format(
                            epoch, bid, logs["loss"], logs["acc"]
                        )
                    )
                  
                # Evaluate
                model.eval()
            
                #for batch in train_loader2:
                for batch in test_loader:
                    x, y = batch
                    x.to(device).float(); y.to(device);
                    
                    with torch.no_grad():
                        # Forward pass
                        out = model(x.float()).float()

                        # Calculate loss                        
                        if with_weighing == True:
                            valid_loss = F.cross_entropy(out, y, weight = w*weights_per_class_ISNS.float())
                        else:
                            valid_loss = F.cross_entropy(out, y)
            
                        # Prediction
                        pred = out.argmax(dim=1)
                        print("pred", pred)
                        print("real", y)
                        print("pred-real", np.abs(pred - y))
            
                        epoch_val_acc = (pred == y).sum()/len(y)
                        score = roc_auc_score(y, pred)
            
                        if score>0.7:
                            torch.save(model.state_dict(), "ANN_isis_features.pt")
                            
                        print("score", score)
                        for j in range(output_dim):
                            real_class_j[j] += (y == j).sum()
            
                            ind_true_j = np.where(y == j)[0]
                            ind_false_j = np.where(y != j)[0]
            
                            true_positive_j[j] += (pred[ind_true_j] == j).sum()
                            false_negative_j[j] += (pred[ind_true_j] != j).sum()
            
                            true_negative_j[j] += (pred[ind_false_j] != j).sum()
                            false_positive_j[j] += (pred[ind_false_j] == j).sum()
            
                score_all.append(score);
                test_acc.append(epoch_val_acc)
                # Log testing result after each epoch
                logs["val_loss"] = valid_loss.item()
                logs["val_acc"] = epoch_val_acc
                print(
                    "Epoch {}, Valid Loss {:.4f}, Valid Accuracy {:.4f}".format(
                        epoch, logs["val_loss"], logs["val_acc"]
                    )
                )
                        
                for batch in train_loader2:
                    
                    x, y = batch
                    x.to(device).float(); y.to(device);

                    with torch.no_grad():
                        # Forward pass
                        out = model(x.float())
                        
                        # Calculate loss                        
                        if with_weighing == True:
                            valid_loss = F.cross_entropy(out, y, weight = w*weights_per_class_ISNS.float())
                        else:
                            valid_loss = F.cross_entropy(out, y)
                            #valid_loss = F.cross_entropy(out[batch.test_mask], batch.y[batch.test_mask])
            
                        # Prediction
                        pred = out.argmax(dim=1)
                        print("pred", pred)
                        print("real", y)
                        print("pred-real", np.abs(pred - y))
                       
                        epoch_val_acc = (pred == y).sum()/len(y)
                        score = roc_auc_score(y, pred)
                       
                        print("score", score)
                        for j in range(output_dim):
                            real_class_j[j] += (y == j).sum()
            
                            ind_true_j = np.where(y == j)[0]
                            ind_false_j = np.where(y != j)[0]
            
                            true_positive_j[j] += (pred[ind_true_j] == j).sum()
                            false_negative_j[j] += (pred[ind_true_j] != j).sum()
            
                            true_negative_j[j] += (pred[ind_false_j] != j).sum()
                            false_positive_j[j] += (pred[ind_false_j] == j).sum()
                            
                    # Log testing result after each epoch
                    logs["val_loss"] = valid_loss.item()
                    logs["val_acc"] = epoch_val_acc
                    print(
                    "TRAIN Epoch {}, Valid Loss {:.4f}, Valid Accuracy {:.4f}".format(
                        epoch, logs["val_loss"], logs["val_acc"]
                    )
                )
            
                model.train()
                #scheduler.step()
            
            return model, score_all, test_acc, test_train_acc_to_see, batch_size




def test_no_graph(model, train_loader, train_loader2, test_loader, with_weighing, optimizer, output_dim, seed):
            model.eval()
            model.float()
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logs = {}

            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
    
            real_class_j = np.zeros(output_dim)
            true_positive_j = np.zeros(output_dim)
            true_negative_j = np.zeros(output_dim)
            false_positive_j = np.zeros(output_dim)
            false_negative_j = np.zeros(output_dim)
    
            for batch in train_loader2:
                    x, y = batch
                    x.to(device).float(); y.to(device);
                    batch_size = x.shape[0]
                
                    with torch.no_grad():
                        # Forward pass
                        out = model(x.float())
                        
                        # Calculate loss                        
                        if with_weighing == True:
                            valid_loss = F.cross_entropy(out, y, weight = w*weights_per_class_ISNS.float())
                        else:
                            valid_loss = F.cross_entropy(out, y)
            
                        # Prediction
                        pred = out.argmax(dim=1)
                        print("pred", pred)
                        print("real", y)
                        print("pred-real", np.abs(pred - y))
                  
                        epoch_val_acc = (pred==y).sum()/len(y)
                        score = roc_auc_score(y, pred)
                     
                        print("score", score)
                        for j in range(output_dim):
                            real_class_j[j] += (y == j).sum()
            
                            ind_true_j = np.where(y == j)[0]
                            ind_false_j = np.where(y != j)[0]
            
                            true_positive_j[j] += (pred[ind_true_j] == j).sum()
                            false_negative_j[j] += (pred[ind_true_j] != j).sum()
            
                            true_negative_j[j] += (pred[ind_false_j] != j).sum()
                            false_positive_j[j] += (pred[ind_false_j] == j).sum()
                            
                    # Log testing result after each epoch
                    logs["val_loss"] = valid_loss.item()
                    logs["val_acc"] = epoch_val_acc
                    print(
                    "TRAIN overall Valid Loss {:.4f}, Valid Accuracy {:.4f}".format(
                           logs["val_loss"], logs["val_acc"]
                    )
                )
            
            train_loss = logs["val_loss"]
            train_acc = logs["val_acc"]
            
            for batch in test_loader:
                    x, y = batch
                    x.to(device).float(); y.to(device);
                
                    with torch.no_grad():
                        out_test = model(x.float())
                        
                    pred = out_test.argmax(dim=1)
            
                    print("pred", pred)
                    print("real", y)
                    print("pred-real", np.abs(pred - y))
            
                    epoch_val_acc = (pred == y).sum()/len(y)
                    score = roc_auc_score(y, pred)
            
                    print("test acc", "score", epoch_val_acc.item(), score)
            
            
            test_acc = epoch_val_acc.item()
            test_score = score
            outputs = [out_test, pred,  y]
        
            return train_loss, train_acc, test_score, test_acc, batch_size, outputs
