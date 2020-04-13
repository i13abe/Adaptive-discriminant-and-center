import torch
import torch.nn as nn
import numpy as np

class AdaptiveDiscriminantLoss(nn.Module):
    def __init__(self, alpha, num_classes, device = torch.device("cpu")):
        super(AdaptiveDiscriminantLoss, self).__init__()
        self.alpha = alpha
        self.num = num_classes
        self.device = device
        
        self.mean_target = torch.zeros(self.num).to(device)
        self.mean_other = torch.zeros(self.num).to(device)
        self.mean_total = torch.zeros(self.num).to(device)
        
        self.mean_target2 = torch.zeros(self.num).to(device)
        self.mean_other2 = torch.zeros(self.num).to(device)
        self.mean_total2 = torch.zeros(self.num).to(device)
        
        
    def forward(self, data, label, train = False):
        batch_size = len(label)
        G = 0
        
        one_hot = torch.zeros((len(label),self.num)).to(self.device)
        for i in range(batch_size):
            one_hot[i][label[i]] += 1.
        
        other_hot = (one_hot.clone().detach() == 0).type(torch.float32).to(self.device)
        
        alphas = (self.alpha**torch.tensor(range(batch_size-1, -1, -1), dtype=torch.float32)).to(self.device)
        alphas = alphas.view(-1, 1)
        
        
        mean_target = self.mean_target.clone().detach()
        mean_other = self.mean_other.clone().detach()
        mean_total = self.mean_total.clone().detach()
        
        mean_target2 = self.mean_target2.clone().detach()
        mean_other2 = self.mean_other2.clone().detach()
        mean_total2 = self.mean_total2.clone().detach()
        
        
        target_data = data*one_hot
        other_data = data*other_hot
        
        target_data2 = (data*one_hot)**2
        other_data2 = (data*other_hot)**2
        
        
        target_data = (1 - self.alpha)*target_data*alphas
        other_data = (1 - self.alpha)*other_data*alphas
        total_data = (1 - self.alpha)*data*alphas
        
        target_data2 = (1 - self.alpha)*target_data2*alphas
        other_data2 = (1 - self.alpha)*other_data2*alphas
        total_data2 = (1 - self.alpha)*(data**2)*alphas

        
        mean_target = mean_target*(self.alpha**batch_size)
        mean_other = mean_other*(self.alpha**batch_size)
        mean_total = mean_total*(self.alpha**batch_size)
        
        mean_target2 = mean_target2*(self.alpha**batch_size)
        mean_other2 = mean_other2*(self.alpha**batch_size)
        mean_total2 = mean_total2*(self.alpha**batch_size)
        
        
        mean_target = mean_target + torch.sum(target_data, 0)
        mean_other = mean_other + torch.sum(other_data, 0)
        mean_total = mean_total + torch.sum(total_data, 0)
        
        mean_target2 = mean_target2 + torch.sum(target_data2, 0)
        mean_other2 = mean_other2 + torch.sum(other_data2, 0)
        mean_total2 = mean_total2 + torch.sum(total_data2, 0)
        
        
        
        sigma_w = mean_target2 + mean_other2 - mean_target**2 - mean_other**2
        sigma_t = mean_total2 - mean_total**2
        
        
        if train:
            self.mean_target = mean_target.clone().detach()
            self.mean_other = mean_other.clone().detach()
            self.mean_total = mean_total.clone().detach()
            self.mean_target2 = mean_target2.clone().detach()
            self.mean_other2 = mean_other2.clone().detach()
            self.mean_total2 = mean_total2.clone().detach()
            
        
 
        G = torch.mean(sigma_w / (sigma_t.detach().clone() + 1e-5))
                
        return G