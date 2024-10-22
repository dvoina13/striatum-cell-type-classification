import torch
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from sklearn.metrics import roc_auc_score

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
                      
                        print("pred", pred[batch.test_mask])
                        print("real", batch.y[batch.test_mask])
                        print("pred-real", np.abs(pred[batch.test_mask] - batch.y[batch.test_mask]))
            
                        epoch_val_acc = (pred[batch.test_mask]==batch.y[batch.test_mask]).sum()/(batch.test_mask==True).sum()
                        score = roc_auc_score(batch.y[batch.test_mask], pred[batch.test_mask])
            
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
            
            return model, score_all, test_acc, test_train_acc_to_see, batch_size



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
            
                    epoch_val_acc = (pred[batch.test_mask]==batch.y[batch.test_mask]).sum()/(batch.test_mask==True).sum()
                    score = roc_auc_score(batch.y[batch.test_mask], pred[batch.test_mask])
            
                    print("test acc", "score", epoch_val_acc.item(), score)
            
            
            test_acc = epoch_val_acc.item()
            test_score = score
            outputs = [out_test, pred[batch.test_mask],  batch.y[batch.test_mask]]
        
            return train_loss, train_acc, test_score, test_acc, batchsize, outputs





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
