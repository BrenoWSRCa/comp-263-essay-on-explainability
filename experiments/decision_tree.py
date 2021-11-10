"""
Here we describe experiments with both "vanilla" decision trees and state of the art decision tree settings for the
CIFAR 10 dataset.
"""

import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from os.path import join as path_join
from os import makedirs

# Setting random seed for reproducibility of the results
torch.manual_seed(0)
np.random.seed(0)

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

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,  batch_size=len(trainset), # batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), # batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = trainset.classes
#%%
# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # un-normalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
#%%

# # get some random training images
# dataiter = iter(trainloader)
# images, labels = dataiter.next()
#
# # show images
# imshow(torchvision.utils.make_grid(images))
# # print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

#%%

X_train, y_train = next(iter(trainloader))
X_train = X_train.reshape((X_train.shape[0], np.prod(X_train.shape[1:])))
#%%
dt = tree.DecisionTreeClassifier()
dt_fit = dt.fit(X_train, y_train)
print(f"The depth of this tree is {dt_fit.get_depth()} and it has {dt_fit.get_n_leaves()} leaves")
tree.plot_tree(dt_fit)
plt.show()
#%%
X_test, y_test = next(iter(testloader))
X_test = X_test.reshape((X_test.shape[0], np.prod(X_test.shape[1:])))
#%%
print(classification_report(y_test, dt_fit.predict(X_test), target_names=classes))


parameters = {'max_depth': (5, 10, 15, 20, 30),
              #'min_samples_split': (2,3,5,10,15,20),
              'min_samples_leaf': (2,5,10,15,20),
              'max_features': ('sqrt', 'log2', None)}
opt_dt = RandomizedSearchCV(tree.DecisionTreeClassifier(), parameters, cv=3, n_jobs=-1, refit=True, n_iter=20)
opt_dt_fit = opt_dt.fit(X_train, y_train) # Hyper-parameter optimized DT
print(f"The depth of this tree is {opt_dt_fit.best_estimator_.get_depth()} and it has {opt_dt_fit.best_estimator_.get_n_leaves()} leaves")
tree.plot_tree(opt_dt_fit.best_estimator_)
plot_fig("opt_dt_fitted.pdf")
print(classification_report(y_test, opt_dt_fit.predict(X_test), target_names=classes))
