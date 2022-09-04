import torch
import torch.nn as nn



class CenterLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        dim,
        alpha=1.0,
    ):
        super(CenterLoss, self).__init__()
        self.center = torch.zeros((num_classes, dim))
        self.mse = nn.MSELoss()
        self.alpha = alpha
        self.num_classes = num_classes
        self.dim = dim
        
    
    def computeCneter(
        self,
        inputs,
        labels,
    ):
        inputs = inputs.detach()
        device = inputs.device
        self.center = self.center.to(device)
        sum_inputs = torch.zeros((self.num_classes, self.dim)).to(device)
        counts = torch.ones((self.num_classes, 1)).to(device)
        for (data, label) in zip(inputs, labels):
            sum_inputs[label] += data
            counts[label] += 1
        
        self.center *= 1 + counts - self.alpha*counts
        self.center /= 1 + counts
        sum_inputs /= 1 + counts
        sum_inputs *= self.alpha
        self.center += sum_inputs
            
            
    def forward(
        self,
        inputs,
        labels
    ):
        device = inputs.device
        self.center = self.center.to(device)
        loss = self.mse(inputs, self.center[labels])
        return loss
    
    
    def setAplha(self, alpha):
        self.alpha = alpha
        
        
    def resetCenter(self, center=None):
        if center is None:
            self.center *= 0
        else:
            self.center = center