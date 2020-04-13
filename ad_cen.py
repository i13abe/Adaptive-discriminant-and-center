import torch
import torch.nn as nn

class AdaptiveCenterLoss(nn.Module):
    def __init__(self, alpha, num_classes, dim, device = torch.device("cpu")):
        super(AdaptiveCenterLoss, self).__init__()
        self.alpha = alpha
        self.num = num_classes
        self.dim = dim
        self.device = device
        
        self.mse = nn.MSELoss(reduction = 'sum')
        
        self.cen = torch.zeros(num_classes, dim).to(device)
        
    def forward(self, data, labels):
        l = len(labels)
        center = torch.zeros(l,self.dim)
        center = center.to(self.device)
        
        for i in range(self.num):
            center[labels == i] = self.cen[i]
        
        loss = self.mse(data, center)
        loss = loss / l
        return loss
    
    def compute_center(self, data, labels):
        for i in range(self.num):
            i_data = data[labels == int(i)]
            l = len(i_data)
            for j in range(l):
                self.cen[i] = self.alpha*self.cen[i] + (1 - self.alpha)*i_data[j]
        return self.cen
    
    def get_center(self):
        return self.cen