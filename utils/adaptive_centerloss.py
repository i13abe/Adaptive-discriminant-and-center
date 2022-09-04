import torch
import torch.nn as nn
        
    
    
class AdaptiveCenterLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        dim,
        alpha=0.9,
    ):
        super(AdaptiveCenterLoss, self).__init__()
        self.center = torch.zeros((num_classes, dim))
        self.mse = nn.MSELoss()
        self.alpha = alpha
        
    
    def computeCneter(
        self,
        inputs,
        labels,
    ):
        inputs = inputs.detach()
        device = inputs.device
        self.center = self.center.to(device)
        for (data, label) in zip(inputs, labels):
            self.center[label] *= self.alpha
            self.center[label] += (1-self.alpha)*data
            
            
    def forward(
        self,
        inputs,
        labels,
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