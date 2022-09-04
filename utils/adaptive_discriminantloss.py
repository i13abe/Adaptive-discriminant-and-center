import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AdaptiveDiscriminantLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.9):
        super(AdaptiveDiscriminantLoss, self).__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        
        self.mu_k = torch.zeros(num_classes)
        self.hat_mu_k = torch.zeros(num_classes)
        self.mu_T_k = torch.zeros(num_classes)
        
        self.sigma_W = torch.zeros(num_classes)
        self.sigma_T = torch.zeros(num_classes)
        
        
    def forward(self, inputs, labels):
        N = len(inputs)
        
        device = inputs.device
        self.mu_k = self.mu_k.to(device)
        self.hat_mu_k = self.hat_mu_k.to(device)
        self.mu_T_k = self.mu_T_k.to(device)
        self.sigma_W = self.sigma_W.to(device)
        self.sigma_T = self.sigma_T.to(device)
        
        one_hot = F.one_hot(labels, num_classes=self.num_classes)
        other_one_hot = (one_hot==0).to(torch.int64)
        
        """hint
        mu_(n) = alpha*mu_(n-1) + (1-alpha)*data_(n)
               = alpha*alpha*mu_(n-2) +  alpha*(1-alpha)*data_(n-1) + (1-alpha)*data_(n)
        """
        alphas = self.alpha**torch.arange(N-1, -1, -1)
        alphas = alphas.view(-1, 1).to(device)
        
        mu_k = torch.sum(inputs*one_hot*alphas, dim=0)
        mu_k = self.mu_k*self.alpha**N + (1-self.alpha)*mu_k
        
        hat_mu_k = torch.sum(inputs*other_one_hot*alphas, dim=0)
        hat_mu_k = self.hat_mu_k*self.alpha**N + (1-self.alpha)*hat_mu_k
        
        mu_T_k = torch.sum(inputs*alphas, dim=0)
        mu_T_k = self.mu_T_k*self.alpha**N + (1-self.alpha)*mu_T_k
        
        sigma_W = one_hot*(inputs - mu_k)**2 + other_one_hot*(inputs - hat_mu_k)**2
        sigma_W = torch.sum(sigma_W*alphas, dim=0)
        sigma_W = self.sigma_W*self.alpha**N + (1-self.alpha)*sigma_W
        
        sigma_T = (inputs - mu_T_k)**2
        sigma_T = torch.sum(sigma_T*alphas, dim=0)
        sigma_T = self.sigma_T*self.alpha**N + (1-self.alpha)*sigma_T
        sigma_T = sigma_T.clamp(min=1e-8)
        
        if self.training:
            self.mu_k = mu_k.clone().detach()
            self.hat_mu_k = hat_mu_k.clone().detach()
            self.mu_T_k = mu_T_k.clone().detach()
            self.sigma_W = sigma_W.clone().detach()
            self.sigma_T = sigma_T.clone().detach()
            
        G = torch.sum(sigma_W/sigma_T)   
        return G