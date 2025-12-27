import torch.nn as nn

class IdentityHead(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()          

    
    def forward(self, x, *args, **kwargs):
        return x                    

    
    def loss(self, pred, target=None, mask=None):
        return {}                   