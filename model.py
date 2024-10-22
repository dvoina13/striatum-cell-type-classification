from torch_geometric.nn import GraphSAGE
import torch
import torch.nn as nn

def define_model(hp):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = GraphSAGE(
        in_channels=120, #920, #100, #61, #901, #61, #6115, 
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