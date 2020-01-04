# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
"""
This is the main entry point for MP6. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils import data
import math

#Note : Part of this code comes from the tutorial of Stanford Universityfrom this link below:
#https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
#-------------------------------------------------------------------------------
class DataSet(data.Dataset):
    def __init__(self, data, labels):
        self.labels = labels
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = self.data[index]
        y = self.labels[index]
        return X, y 
#--------------------------------------------------------------------------------

class NeuralNet(torch.nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initialize the layers of your neural network

        @param lrate: The learning rate for the model.
        @param loss_fn: A loss function defined in the following way:
            @param yhat - an (N,out_size) tensor
            @param y - an (N,) tensor
            @return l(x,y) an () tensor that is the mean loss
        @param in_size: Dimension of input
        @param out_size: Dimension of output

        For Part 1 the network should have the following architecture (in terms of hidden units):

        in_size -> 32 ->  out_size

        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        #----------------------------------------------------------------------
        self.layer1 = torch.nn.Conv2d(3, 6, 3, stride=1, padding=0)
        self.layer2 = torch.nn.MaxPool2d(2, 2)
        self.layer3 = torch.nn.Conv2d(6, 15, 6, stride=1, padding=0)
        self.layer4 = torch.nn.MaxPool2d(2, 2)
        self.layer5 = torch.nn.Linear(375, 128)
        self.layer6 = torch.nn.Linear(128, 64)
        self.layer7 = torch.nn.Linear(64, 2)
        #----------------------------------------------------------------------
        self.lrate = lrate
        self.optimizer = optim.SGD(self.parameters(), lr=lrate, momentum=0.9)

    def set_parameters(self, params):
        """ Set the parameters of your network
        @param params: a list of tensors containing all parameters of the nVariable(self.layer1.type(torch.TensorFloat)etwork
        """
        self.layer1.weight = torch.nn.Parameter(params[0])
        self.layer1.bias = torch.nn.Parameter(params[1])
        self.layer2.weight = torch.nn.Parameter(params[2])
        self.layer2.bias = torch.nn.Parameter(params[3])
        self.layer3.weight = torch.nn.Parameter(params[4])
        self.layer3.bias = torch.nn.Parameter(params[5])
        self.layer4.weight = torch.nn.Parameter(params[6])
        self.layer4.bias = torch.nn.Parameter(params[7])
        self.layer5.weight = torch.nn.Parameter(params[8])
        self.layer5.bias = torch.nn.Parameter(params[9])
        self.layer6.weight = torch.nn.Parameter(params[10])
        self.layer6.bias = torch.nn.Parameter(params[11])
        self.layer7.weight = torch.nn.Parameter(params[12])
        self.layer7.bias = torch.nn.Parameter(params[13])
        pass

    def get_parameters(self):
        """ Get the parameters of your network
        @return params: a list of tensors containing all parameters of the network
        """
        return []

    def forward(self, x):
        """ A forward pass of your neural net (evaluates f(x)).

        @param x: an (N, in_size) torch tensor

        @return y: an (N, out_size) torch tensor of output from the network
        """
        x = torch.Tensor((x.numpy()).reshape((len(x), 3, 32, 32)))
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.layer5(x))
        x = F.relu(self.layer6(x))
        x = self.layer7(x)
        return x

    def step(self, x, y):
        """
        Performs one gradient step through a batch of data x with labels y
        @param x: an (N, in_size) torch tensor
        @param y: an (N,) torch tensor
        @return L: total empirical risk (mean of losses) at this time step as a float
        """
        #lamada = 0.005
        #----clear buffers-----------
        self.optimizer.zero_grad()
        #----------------------------
        yhat = self.forward(x)
        loss = self.loss_fn(yhat, y)
        #----------------------------
        #all_parameters = self.get_parameters()
        #for i in range(int(len(all_parameters)/2)):
        #    loss += lamada*np.linalg.norm(all_parameters[2*i].detach().numpy())
        loss.backward()
        self.optimizer.step()
        return loss


def fit(train_set, train_labels, dev_set, n_iter, batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) torch tensor
    @param train_labels: an (N,) torch tensor
    @param dev_set: an (M,) torch tensor
    @param n_iter: int, the number of epochs of training
    @param batch_size: The size of each batch to train on. (default 100)

    # return all of these:

    @return losses: Array of total loss at the beginning and after each iteration. Ensure len(losses) == n_iter
    @return yhats: an (M,) NumPy array of approximations to labels for dev_set
    @return net: A NeuralNet object

    # NOTE: This must work for arbitrary M and N
    """
    #num_epoch = n_iter
    model = NeuralNet(2e-3, torch.nn.CrossEntropyLoss(), len(train_set[0]), 2) #output size应该是啥?
    final_loss = []
    final_result = []
    #----------------preprocessing-----------------
    #train_set = torch.Tensor((train_set.numpy()).reshape((len(train_set), 3, 32, 32)))
    #dev_set = torch.Tensor((dev_set.numpy()).reshape((len(dev_set), 3, 32, 32)))
    #----------------------------------------------
    for i in range(int(1.5*n_iter)):
        print(i)
        train_set_batch = DataSet(train_set, train_labels)
        train_loader = DataLoader(dataset = train_set_batch, batch_size = batch_size) #generate batch
        for x, y in train_loader:
            model.step(x, y)
        current_value = model.forward(train_set)
        final_loss.append(model.loss_fn(current_value, train_labels).item())
    #-----------classification--------------
    #-----------TODO------------------------
    for i in range(len(dev_set)):
        temp_result = model.forward(torch.Tensor([(dev_set[i].numpy().tolist())]))
        if(list(temp_result)[0][0]) > (list(temp_result)[0][1]):
            final_result.append(0)
        else:
            final_result.append(1)
    print(final_loss)
    print(model.get_parameters())
    return final_loss, final_result, model
