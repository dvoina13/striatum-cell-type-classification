from torch_geometric.nn import GraphSAGE
import torch
import torch.nn as nn
import numpy as np
from utils_sparseLayer import clamp_probs, concrete_sample, bernoulli_concrete_sample, BinaryMask, BinaryGates, ConcreteSelector 

def define_model(hp):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = GraphSAGE(
        in_channels=61, #920, #100, #61, #901, #61, #6115, 
        hidden_channels=hp["hidden_dim"],
        num_layers=hp["num_layers"],
        out_channels=2,
        dropout=hp["dropout"],
        act="ELU",
        norm = "BatchNorm",
    ).to(device)

    """
    model = GAT(in_channels= 61, 
            hidden_channels= hp["hidden_dim"], 
            num_layers= hp["num_layers"], 
            out_channels= 2, 
            dropout= hp["dropout"], 
            act= "ELU", 
            norm= "BatchNorm",
           ).to(device)
    #model2 = nn.Sequential(nn.ELU(), nn.Linear(100,10), nn.ELU(), nn.Linear(10,2))
    """

    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    return model, optimizer, scheduler




class ANN_linear(nn.Module):
    
    def __init__(self, input_dim=100, output_dim=2, hd = [50, 10, 10], no_layers=3):
        super(ANN_linear, self).__init__()
        
        self.hd = hd
        self.no_layers = no_layers
        
        if type(hd) == int:
            HD = [hd]*no_layers
        else:
            HD = hd
        
        HD = [input_dim] + HD + [output_dim]
        
        layers = []
        
        for i in range(no_layers+1):
            if i<no_layers:
                layers += [nn.Linear(HD[i], HD[i+1]), nn.ELU(), nn.Dropout(0.1)]
            else:
                layers += [nn.Linear(HD[i], HD[i+1])]#, nn.Softmax(dim=1)]
        self.ffwd = nn.Sequential(*layers)
        
    def forward(self,x):
        
        y = self.ffwd(x)
        return y
        

def define_model_no_graph(hp):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ANN_linear()
    model = model.to(device).float()

    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    return model, optimizer, scheduler



#model with sparsifying layer
class Model_with_GraphSAGE_and_SparsityLayer(nn.Module):
        
        def __init__(self, args, hp, input_size, output_size, num_selections = None, preselected_inds = [], **kwargs):
                super(Model_with_GraphSAGE_and_SparsityLayer, self).__init__()
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                #num_selections = args.num_selections
                input_layer = args.input_layer
                hidden = hp["hidden_dim"]
        
                self.input_size = input_size
                self.output_size = output_size
                self.max_input_size = input_size
                
                if input_layer not in ["binary_mask", "binary_gates", "concrete_selector"]:
                    return 

                #for default binary_gates, num_selections = None!
                if num_selections is None:
                    if input_layer in ('binary_mask', 'concrete_selector'):
                        raise ValueError(
                            f'must specify num_selections for {input_layer} layer')
                else:
                    if input_layer in ('binary_gates'):
                        raise ValueError('num_selections cannot be specified for '
                                         f'{input_layer} layer')


                #if there are pre-selected_inds (list is empty)
                self.preselected_inds = np.sort(preselected_inds)
                assert len(self.preselected_inds) < input_size
                self.preselected = np.array(
                    [i in self.preselected_inds for i in range(input_size)])
                preselected_size = len(self.preselected_inds)
                self.has_preselected = preselected_size > 0
                
                print("self.preselected_inds", self.preselected_inds)
                
                # Set up input layer.
                if input_layer == 'binary_mask':
                    model_input_size = input_size
                    self.model_input_layer = BinaryMask(
                        input_size - preselected_size, num_selections, **kwargs)
                elif input_layer == 'binary_gates':
                    model_input_size = input_size
                    self.model_input_layer = BinaryGates(input_size - preselected_size, **kwargs)
                elif input_layer == 'concrete_selector':
                    model_input_size = num_selections + preselected_size
                    self.model_input_layer = ConcreteSelector(
                        input_size - preselected_size, num_selections, **kwargs)
                else:
                    raise ValueError('unsupported input layer: {}'.format(input_layer))

                self.model_graph = GraphSAGE(
                                in_channels=100, #920, #100, #61, #901, #61, #6115, 
                                hidden_channels=hp["hidden_dim"],
                                num_layers=hp["num_layers"],
                                out_channels=2,
                                dropout=hp["dropout"],
                                act="ELU",
                                norm = "BatchNorm",
                            ).to(device)


        def forward(self, x, edge_index):
            
            if self.has_preselected:
                pre = x[:, self.preselected]
                x, m = self.model_input_layer(x[:, ~self.preselected])
                x = torch.cat([pre, x], dim=1)
            else:
                x, m = self.model_input_layer(x)
            
            pred = self.model_graph(x, edge_index)
            
            return pred, x, m
        
        
        def return_models(self):
            
            optimizer = torch.optim.Adam(list(self.model_graph.parameters()) + list(self.model_input_layer.parameters()))
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
            
            return self.model_graph, self.model_input_layer, optimizer, scheduler