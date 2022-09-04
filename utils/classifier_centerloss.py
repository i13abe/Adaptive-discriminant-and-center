import torch
from tqdm import tqdm
import numpy as np



class ClassifierCenterLoss(object):
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        centerloss,
        lam=1.0,
    ):
        """This is classifier fitter with center loss.

        Args:
            model: torch model.
            optimzier: torch optimzier.
            criterion: classfier criterion.
            centerloss: center loss function.
            lam: The coefficient of center loss. Defaults to 1.0.
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.centerloss = centerloss
        self.lam = lam
    
    
    def fit(
        self,
        EPOCH,
        trainloader,
        testloader=None,
        validation_mode=True,
        scheduler=None,
        device="cuda:0",
        lam=None,
    ):
        losses = {"train":[], "test":[]}
        cel_losses = {"train":[], "test":[]}
        center_losses = {"train":[], "test":[]}
        accuracies = {"train":[], "test":[]}
        for epoch in range(EPOCH):
            print(f"epoch:{epoch+1}")
            self.train(trainloader, device, lam)
            if validation_mode:
                print("Training data results-----------------------------")
                loss, cel_loss, center_loss, acc = self.test(
                    trainloader,
                    device,
                    lam,
                )
                losses["train"].append(loss)
                cel_losses["train"].append(cel_loss)
                center_losses["train"].append(center_loss)
                accuracies["train"].append(acc)
                if testloader is not None:
                    print("Test data results---------------------------------")
                    loss, cel_loss, center_loss, acc = self.test(
                        testloader,
                        device,
                        lam,
                    )
                    losses["test"].append(loss)
                    cel_losses["test"].append(cel_loss)
                    center_losses["test"].append(center_loss)
                    accuracies["test"].append(acc)
            if scheduler is not None:
                scheduler.step()
        return losses, cel_losses, center_losses, accuracies
            
        
    def train(
        self,
        dataloader,
        device="cuda:0",
        lam=None,
    ):
        device = torch.device(device)
        
        if lam is None:
            lam = self.lam
            
        self.model.train()
        for (inputs, labels) in tqdm(dataloader):
            self.optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)

            latents, outputs = self.model(inputs)
            self.centerloss.computeCneter(latents, labels)

            cel_loss = self.criterion(outputs, labels)
            center_loss = self.centerloss(latents, labels)
            
            loss = cel_loss + lam*center_loss
            
            loss.backward()
            self.optimizer.step()
            
            
    def test(
        self,
        dataloader,
        device="cuda:0",
        lam=None,
    ):
        sum_loss = 0.
        sum_cel_loss = 0.
        sum_center_loss = 0.
        sum_acc = 0.
        
        if lam is None:
            lam = self.lam
        
        self.model.eval()
        for (inputs, labels) in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            latents, outputs = self.model(inputs)

            cel_loss = self.criterion(outputs, labels)
            center_loss = self.centerloss(latents, labels)
            
            loss = cel_loss + lam*center_loss
            
            sum_loss += loss.item()*inputs.shape[0]
            sum_cel_loss += cel_loss.item()*inputs.shape[0]
            sum_center_loss += center_loss.item()*inputs.shape[0]
            _, predicted = (outputs).max(1)
            correct = (predicted == labels).sum().item()
            sum_acc += correct
        
        sum_loss /= len(dataloader.dataset)
        sum_cel_loss /= len(dataloader.dataset)
        sum_center_loss /= len(dataloader.dataset)
        sum_acc /= len(dataloader.dataset)
        
        print(f"mean_loss={sum_loss}, mean_cel_loss={sum_cel_loss}, "\
              f"mean_center_loss={sum_center_loss}, acc={sum_acc}")
        
        return sum_loss, sum_cel_loss, sum_center_loss, sum_acc
    
    
    def getLatents(
        self,
        dataloader,
        based_labels,
        device="cuda:0",
    ):
        data_dict = dict(zip(based_labels, [[] for i in range(len(based_labels))]))
        self.model.eval()
        for (inputs, labels) in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            latents = self.model.getLatents(inputs)
            
            for data, label in zip(latents, labels):
                data_dict[based_labels[label]].append(data.cpu().detach().numpy())
            
        for key in based_labels:
            data_dict[key] = np.vstack(data_dict[key])
        
        data_dict["center"] = self.centerloss.center.cpu().detach().numpy()
        return data_dict
            
    
    def testSummary(
        self,
        dataloader,
        device="cuda:0",
    ):
        sum_loss = 0.
        sum_acc = 0.
        
        num_classes = len(dataloader.dataset.classes)
        num_class = np.zeros((1, num_classes))[0]
        class_acc = np.zeros((1, num_classes))[0]
        
        self.model.eval()
        for (inputs, labels) in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = self.model(inputs)

            loss = self.criterion(outputs, labels)
            
            sum_loss += loss.item()*inputs.shape[0]
            _, predicted = (outputs).max(1)
            correct = (predicted == labels).sum().item()
            sum_acc += correct
            
            for i, label in enumerate(labels):
                num_class[label] += 1
                class_acc[label] += (predicted[i] == label).item()
        
        sum_loss /= len(dataloader.dataset)
        sum_acc /= len(dataloader.dataset)
        class_acc /= num_class
        
        print(f"mean_loss={sum_loss}, acc={sum_acc}")
        
        for i in range(len(class_acc)):
            print(f"class {i} accuracy={class_acc[i]}")
        
        return sum_loss, sum_acc, class_acc
                  
        
    def setModel(self, model):
        self.model = model
    
    
    def setOptimizer(self, optimizer):
        self.optimizer = optimizer
        
        
    def setCriterion(self, criterion):
        self.criterion = criterion
        
        
    def setCenterLoss(self, centerloss):
        self.centerloss = centerloss
        
    