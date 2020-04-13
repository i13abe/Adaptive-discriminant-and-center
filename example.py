# to load dataset
from datasets import Datasets
# criterions
from ad_cen import AdaptiveCenterLoss
from ad_disc import AdaptiveDiscriminantLoss
# for network and training
from network import Net
from network_fit import NetworkFit
# to calculate the score
import savescore
from score import Score
from score_calc import ScoreCalc
# pytorch
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
# numpy and matplotlib
import numpy as np
import matplotlib.pyplot as plt
# to show the histogram and feature mapping
import histgram
from plot_feature import PlotFeature


# initialize for  each parameters
DATASET = 'MNIST'
BATCH_SIZE = 100
NUM_WORKERS = 2
WEIGHT_DECAY = 0.01
LEARNING_RATE = 0.01
MOMENTUM = 0.9
SCHEDULER_STEPS = 50
SCHEDULER_GAMMA = 0.1
SEED = 1
EPOCH = 100
FEATURE = 100
OUTPUTS = 10
ALPHA1 = 0.999
ALPHA2 = 0.999
LAMBDA1 = 0.001
LAMBDA2 = 1.0


# fixing the seed
torch.cuda.manual_seed_all(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


# check if gpu is available
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("gpu mode")
else:
    device = torch.device("cpu")
    print("cpu mode")


# the name of results files
codename = 'ad_ac_example'
fnnname = codename + "_fnn_model"
total_loss_name = codename + "_total_loss"
acc_name = codename + "_accuracy"
soft_loss_name = codename + "_softmax_loss"
ad_disc_loss_name = codename + "_adaptivediscriminant_loss"
ad_cen_loss_name = codename + "_adaptivecenter_loss"
result_name = codename + "_result"


# load the data set
instance_datasets = Datasets(DATASET, BATCH_SIZE, NUM_WORKERS)
data_sets = instance_datasets.create()

trainloader = data_sets[0]
testloader = data_sets[1]
classes = data_sets[2]
based_labels = data_sets[3]
trainset = data_sets[4]
testset = data_sets[5]


# network and criterions
model = Net(FEATURE, OUTPUTS).to(device)

optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEPS, gamma=SCHEDULER_GAMMA)

soft_criterion = nn.CrossEntropyLoss()
disc_criterion = AdaptiveDiscriminantLoss(ALPHA1, len(classes), device)
cen_criterion = AdaptiveCenterLoss(ALPHA2, len(classes), FEATURE, device)


# fit for training and test
fit = NetworkFit(model, optimizer, soft_criterion, disc_criterion, cen_criterion)


# to manage all scores
loss = Score()
loss_s = Score()
loss_d = Score()
loss_c = Score()
correct = Score()
score_loss = [loss, loss_s, loss_d, loss_c]
score_correct = [correct]
sc = ScoreCalc(score_loss, score_correct, BATCH_SIZE)


# training and test
for epoch in range(EPOCH):
    print('epoch', epoch+1)

    for (inputs, labels) in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        fit.train(inputs, labels, LAMBDA1, LAMBDA2)
    
    for (inputs, labels) in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        losses, corrects = fit.test(inputs, labels, LAMBDA1, LAMBDA2)
        
        sc.calc_sum(losses, corrects)
    
    sc.score_print(len(trainset))
    sc.score_append(len(trainset))
    sc.score_del()
    
    for (inputs, labels) in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        losses, corrects = fit.test(inputs, labels, LAMBDA1, LAMBDA2)
        
        sc.calc_sum(losses, corrects)
    
    sc.score_print(len(testset), train = False)
    sc.score_append(len(testset), train = False)
    sc.score_del()
    
    scheduler.step()


# get the scores
train_losses, train_corrects = sc.get_value()
test_losses, test_corrects = sc.get_value(train = False)


# output the glaphs of the scores
torch.save(model.state_dict(), fnnname + '.pth')
savescore.plot_score(EPOCH, train_losses[0], test_losses[0], y_lim = 2.0, y_label = 'LOSS', legend = ['train loss', 'test loss'], title = 'total loss', filename = total_loss_name)
savescore.plot_score(EPOCH, train_corrects[0], test_corrects[0], y_lim = 1, y_label = 'ACCURACY', legend = ['train acc', 'test acc'], title = 'accuracy', filename = acc_name)
savescore.plot_score(EPOCH, train_losses[1], test_losses[1], y_lim = 2.0, y_label = 'LOSS', legend = ['train loss', 'test loss'], title = 'softmax loss', filename = soft_loss_name)
savescore.plot_score(EPOCH, train_losses[2], test_losses[2], y_lim = 2.0, y_label = 'LOSS', legend = ['train loss', 'test loss'], title = 'discriminant loss', filename = ad_disc_loss_name)
savescore.plot_score(EPOCH, train_losses[3], test_losses[3], y_lim = 2.0, y_label = 'LOSS', legend = ['train loss', 'test loss'], title = 'center loss', filename = ad_cen_loss_name)
savescore.save_data(train_losses[0], test_losses[0], train_corrects[0], test_corrects[0], result_name)


# get the embeddings and neurons of outputs for training data
k = 0
feature1 = np.zeros((len(trainset), FEATURE))
feature2 = np.zeros((len(trainset), OUTPUTS))
total_labels = np.zeros(len(trainset))
cpu_device = torch.device("cpu")

for (inputs, labels) in trainloader:
    inputs = inputs.to(device)

    outputs = fit.get_data(inputs)
    
    embeddings = outputs[0].detach().clone().to(cpu_device)
    neurons = outputs[1].detach().clone().to(cpu_device)
    
    feature1[k: k+len(inputs)] = embeddings
    feature2[k: k+len(inputs)] = neurons
    total_labels[k: k+len(inputs)] = labels.detach().clone().to(cpu_device)
    k += len(inputs)


# make the histogram for training data
c = 6

output_c = feature2[total_labels == c][:, c]
output_other = feature2[total_labels != c][:, c]

histgram.histgram([output_c, output_other], bins = 50, filename = "histogram", ylim = 1000)


# make the embeddings mapping
plot_f = PlotFeature(based_labels)
center = cen_criterion.get_center()
center = center.detach().clone().to(cpu_device)

if FEATURE > 2:
    from sklearn.decomposition import PCA
    pca = PCA(n_components = 2)
    pca.fit(feature1)
    feature1 = pca.transform(feature1)
    center = pca.transform(center)

plot_f.plot_feature(feature1, total_labels, center, filename = "embeddings")


