import torch
from tqdm import tqdm
import numpy as np



class ClassifierDiscriminantLoss(object):
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        discriminantloss,
        lam=1.0,
    ):
        """This is classifier fitter with discriminat loss.

        Args:
            model: torch model.
            optimzier: torch optimzier.
            criterion: classfier criterion.
            descriminatloss: discriminat loss function.
            lam: The coefficient of discriminant loss. Defaults to 1.0.
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.discriminantloss = discriminantloss
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
        discriminant_losses = {"train":[], "test":[]}
        accuracies = {"train":[], "test":[]}
        for epoch in range(EPOCH):
            print(f"epoch:{epoch+1}")
            self.train(trainloader, device, lam)
            if validation_mode:
                print("Training data results-----------------------------")
                loss, cel_loss, discriminant_loss, acc = self.test(
                    trainloader,
                    device,
                    lam,
                )
                losses["train"].append(loss)
                cel_losses["train"].append(cel_loss)
                discriminant_losses["train"].append(discriminant_loss)
                accuracies["train"].append(acc)
                if testloader is not None:
                    print("Test data results---------------------------------")
                    loss, cel_loss, discriminant_loss, acc = self.test(
                        testloader,
                        device,
                        lam,
                    )
                    losses["test"].append(loss)
                    cel_losses["test"].append(cel_loss)
                    discriminant_losses["test"].append(discriminant_loss)
                    accuracies["test"].append(acc)
            if scheduler is not None:
                scheduler.step()
        return losses, cel_losses, discriminant_losses, accuracies
            
        
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

            outputs = self.model(inputs)
            
            cel_loss = self.criterion(outputs, labels)
            self.discriminantloss.train()
            discriminant_loss = self.discriminantloss(outputs, labels)
            
            loss = cel_loss + lam*discriminant_loss
            
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
        sum_discriminant_loss = 0.
        sum_acc = 0.
        
        if lam is None:
            lam = self.lam
        
        self.model.eval()
        for (inputs, labels) in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = self.model(inputs)

            cel_loss = self.criterion(outputs, labels)
            self.discriminantloss.eval()
            discriminant_loss = self.discriminantloss(outputs, labels)
            
            loss = cel_loss + lam*discriminant_loss
            
            sum_loss += loss.item()*inputs.shape[0]
            sum_cel_loss += cel_loss.item()*inputs.shape[0]
            sum_discriminant_loss += discriminant_loss.item()*inputs.shape[0]
            _, predicted = (outputs).max(1)
            correct = (predicted == labels).sum().item()
            sum_acc += correct
        
        sum_loss /= len(dataloader.dataset)
        sum_cel_loss /= len(dataloader.dataset)
        sum_discriminant_loss /= len(dataloader.dataset)
        sum_acc /= len(dataloader.dataset)
        
        print(f"mean_loss={sum_loss}, mean_cel_loss={sum_cel_loss}, "\
              f"mean_center_loss={sum_discriminant_loss}, acc={sum_acc}")
        
        return sum_loss, sum_cel_loss, sum_discriminant_loss, sum_acc
    
    
    def getOutputs(
        self,
        dataloader,
        based_labels,
        device="cuda:0",
    ):
        data_dict = dict(zip(based_labels, [[] for i in range(len(based_labels))]))
        self.model.eval()
        for (inputs, labels) in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = self.model(inputs)
            
            for data, label in zip(outputs, labels):
                data_dict[based_labels[label]].append(data.cpu().detach().numpy())
            
        for key in based_labels:
            data_dict[key] = np.vstack(data_dict[key])
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
        
        
    def setDiscriminantLoss(self, discriminantloss):
        self.discriminantloss = discriminantloss
        
    