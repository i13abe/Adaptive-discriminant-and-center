import torch
import torch.nn as nn
import numpy as np

class NetworkFit(object):
    def __init__(self, model, optimizer, soft_criterion, disc_criterion, cen_criterion):
        self.model = model
        self.optimizer = optimizer
        
        self.soft_criterion = soft_criterion
        self.disc_criterion = disc_criterion
        self.cen_criterion = cen_criterion
        

    def train(self, inputs, labels, lam1 = 1.0, lam2 = 1.0):
        self.optimizer.zero_grad()
        self.model.train()

        outputs = self.model(inputs)
        em_out = outputs[0]
        out = outputs[1]

        self.cen_criterion.compute_center(em_out.detach().clone(), labels)
        cen_loss = self.cen_criterion(em_out, labels)
        
        disc_loss = self.disc_criterion(out, labels, train = True)
        
        soft_loss = self.soft_criterion(out, labels)

        loss = soft_loss + lam1*disc_loss + lam2*cen_loss

        loss.backward()
        self.optimizer.step()
            
            
    def test(self, inputs, labels, lam1 = 1.0, lam2 = 1.0):
        self.model.eval()
        
        outputs = self.model(inputs)
        em_out = outputs[0]
        out = outputs[1]
        
        cen_loss = self.cen_criterion(em_out, labels)
        
        disc_loss = self.disc_criterion(out, labels)
            
        soft_loss = self.soft_criterion(out, labels)
        
        loss = soft_loss + lam1*disc_loss + lam2*cen_loss
        
        _, predicted = out.max(1)
        correct = (predicted == labels).sum().item()
        
        return [loss.item(), soft_loss.item(), disc_loss.item(), cen_loss.item()], [correct]
    
    def get_data(self, inputs):
        self.model.eval()

        outputs = self.model(inputs)
        
        return outputs
        
        
    
