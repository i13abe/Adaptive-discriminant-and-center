import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class DiscriminantLoss(nn.Module):
    def __init__(self, num_classes):
        super(DiscriminantLoss, self).__init__()
        self.num_classes = num_classes
        
        
    def forward(self, inputs, labels):
        one_hot = F.one_hot(labels, num_classes=self.num_classes)
        other_one_hot = (one_hot==0).to(torch.int64)
        
        Nk = torch.sum(one_hot, dim=0).clamp(min=1.) #"min=1" to avoid 0 divide
        mu_k = torch.sum(inputs*one_hot, dim=0)
        mu_k /= Nk
        
        hat_Nk = torch.sum(other_one_hot, dim=0).clamp(min=1.)
        hat_mu_k = torch.sum(inputs*other_one_hot, dim=0)
        hat_mu_k /= hat_Nk
        
        mu_T_k = torch.mean(inputs, dim=0)
        
        sigma_W = one_hot*(inputs - mu_k)**2 + other_one_hot*(inputs-hat_mu_k)**2
        sigma_W = torch.mean(sigma_W, dim=0)
        
        sigma_T = (inputs - mu_T_k)**2
        sigma_T = torch.mean(sigma_T, dim=0).clamp(min=1e-8)
        
        G = torch.sum(sigma_W/sigma_T)
        return G