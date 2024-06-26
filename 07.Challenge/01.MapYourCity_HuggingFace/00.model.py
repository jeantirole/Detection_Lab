
import torch
import torch.nn as nn 

class EnsembleModel(nn.Module):

    '''
    - two models will be ensembled

    '''


    def __init__(self, modelA, modelB):
        super().__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.classifier = nn.Linear(7 * 2, 7)
        
    def forward(self, x):
        x1 = self.modelA(x)
        x2 = self.modelB(x)
        x = torch.cat( (x1, x2), dim=1)
        out = self.classifier(x)
        return out
    
