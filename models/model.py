import torch
import torch.nn as nn
import torch.nn.functional as func

# Pytorch Iris Model
class Model(nn.Module):
    # Initial function
    def __init__(self):
        super(Model, self).__init__()
        self.inputs = nn.Linear(4, 8)
        self.outputs = nn.Linear(8, 3)
    
    # Compile Function
    def forward(self, x):
        x = func.relu(self.inputs(x))
        x = func.softmax(self.outputs(x), dim=1)
        return x
    
    # Load weight
    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
