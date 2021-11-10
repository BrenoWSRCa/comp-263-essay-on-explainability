"""
Here we describe experiments with a simple CNN
CIFAR 10 dataset.

Adapted from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np


from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from os.path import join as path_join, exists
from os import makedirs

# Setting random seed for reproducibility of the results
torch.manual_seed(0)
np.random.seed(0)

CNN_PATH = './cifar_net.pth'

IMG_TO_FILE = True  # Set this oto true if you want to store the outputs in specific files
IMG_PATH = "."

def plot_fig(name=None, **args):
    if IMG_TO_FILE and name is not None:
        makedirs(IMG_PATH, exist_ok=True)
        plt.savefig(path_join(IMG_PATH, name), **args)
    else:
        plt.show(**args)

# Download and process the dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,  batch_size=batch_size,
                                          shuffle=True, num_workers=1)

testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=1)

classes = trainset.classes
#%%
# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # un-normalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
#%%

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

#%%
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc(x)
        return x


net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

if exists(CNN_PATH):
    net.load_state_dict(torch.load(CNN_PATH))
else:

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    torch.save(net.state_dict(), CNN_PATH)

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))



# from nbdt.model import SoftNBDT
# from nbdt.loss import SoftTreeSupLoss
# from nbdt.hierarchy import generate_hierarchy
#
#
# generate_hierarchy(dataset='CIFAR10', arch='vanilla', model=net, method='random')
# crit_nbdt = SoftTreeSupLoss(dataset='CIFAR10', hierarchy='induced-vanilla', criterion=criterion, path_graph="./nbdt/hierarchies/CIFAR10/graph-random.json")
# model = SoftNBDT(dataset='CIFAR10', model=net)