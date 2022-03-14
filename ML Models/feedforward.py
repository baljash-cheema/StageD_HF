from torch import nn, tensor
import torch
from random import randint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import transformation_utils as tu
import validation_utils as vu
import copy


def softmax(x):
    '''
    DEPRECIATED: now using pytorch's built in softmax
    returns softmax of x
    '''
    # x = np.array(x.detach().numpy()) #convert to number
    Ndims = len(x.shape)
    if Ndims == 2:
        return torch.div(torch.exp(x), torch.sum(torch.exp(x), 1).unsqueeze(1))
    elif Ndims == 1:
        return torch.div(torch.exp(x), torch.sum(torch.exp(x)))


class FeedForwardSoftmax(nn.Module):
    def __init__(self, inputN, outputN, hiddenNs=None, bias=True,
                 inputMethod=nn.Sigmoid, hiddenMethods=[]):
        """
        Feed-forward linear neural network, using softmax at the output layer to normalise the outputs.

        :param: inputN = integer. Number of input nodes to our network
        :param: hiddenNs = list of integers. One element for each hidden layer. Each element tells
                       us the size of that layer
        :param: outputN = integer. Nuber of output nodes for our network
        :param: bias = True/False bool. If True pytorch will add a bias term into the calculations
        :param: inputMethod = nn.Module object. transformation to apply to the input layer
        :param: hiddenMethods = list of nn.Module objects. transformation to apply to the hidden layers

        """

        super(FeedForwardSoftmax, self).__init__()

        if hiddenNs is None:
            hiddenNs = []
        if len(hiddenNs) != 0 and len(hiddenMethods) == 0:
            hiddenMethods = [nn.Sigmoid] * len(hiddenNs)  # default to Sigmoid in hidden layers if not specified

        layer_sizes = [inputN] + hiddenNs + [outputN]  # form list of all layer sizes in the network
        self._layer_sizes = layer_sizes

        self._layers = nn.ModuleList()  # initialise container for the layer objects

        # initialise container for the methods list (transformation for each layer, eg: sigmoid)
        self._methods = [inputMethod()] \
                        + [x() for x in hiddenMethods] \
                        + [nn.Softmax(dim=1)]

        for i in range(len(layer_sizes) - 1):  # build the architecture
            self._layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias=bias))

    def forward(self, x, outMethod=True):
        """
        Performs a forward pass of the network by for-looping through the layers in sequence, passing a vector
        x from one layer to the next until the output.

        :param: x = input vector to the network
        :param: outMethod: True/False, whether to use a predetermined transformation on the output (eg: Softmax)
        returns network output
        """
        # assert our vector x is a tensor
        x = tensor(np.array(x)).to(torch.float32)

        for i in range(len(self._layers)):  # iterate through the layers, passing x from one layer to the next
            x = self._layers[i](x)
            if i != len(self._layers) - 1:
                x = self._methods[i](x)
            elif i == len(self._layers) - 1 and outMethod:
                x = self._methods[i](x)
        return x

    def get_weights(self):
        '''
        quick function to return the weights in the NN
        '''
        weights = []
        for layer in self._layers:
            weights.append(layer.weight)
        return weights


def trainNN(dataset, model, loss_func, optimizer, max_epoch=10000,
            loss_target=0.1, method="batch", plot=True, verbosity=True,
            _lambdaL1=0, _lambdaL2=0, minibatch_size=100,
            outMethod=True):
    '''
    takes a dataset, and a model (such as FeedForwardSoftmax), a loss function, and a
    pytorch optimizer and trains the model using a batch method

    :param: dataset = list holding data (in Demeter's format)
    :param: model = torch.nn.Module object -> our neural network object
    :param: loss_func: function to calculate the loss
    :param: optimizer: torch.optim object -> our optimizing function
    :param: max_epoch: maximum number of iterations we'll allow before force-stopping
    :param: loss_target: the target loss we're aiming for -> if this is reached the training will stop
    :param: method: either "batch" or "stochastic" or "minibatch".
    :poram: plot: True/False. If true, matplotlib called to plot the loss vs epoch
    :param: _lambdaL2: the regularisation constant for L2
    :param: _lambdaL1: the regularisation constant for L1
    :param: minibatch_size: the size of the minibatch to be used if method = minibatch
    :param: outMethod: True/False depending if we want to run the NN output through a softmax or not when calculating losses
    '''

    model.train()  # tell the model we're training
    train_loss = []

    Noutputs = model._layer_sizes[-1]

    # set up the data
    X = dataset[0]
    y = dataset[1].squeeze().to(torch.long)

    # ybin = tu.labels_to_binary(y, Noutputs)

    full_loss = 1e8  # initialise to a value somewhere above the threshold
    epoch = 0  # counter for which training epoch we are in
    if verbosity:
        print('Training {} using method: {}'.format(type(model).__name__, method))

    while full_loss > loss_target and epoch < max_epoch:
        if method.lower() == "batch":  # if batch -> use all the training data in each iteration
            _X = X
            _y = y
        elif method.lower() == 'stochastic':  # if stochastic -> use one randomly selected example for each epoch
            randomi = randint(0, len(X) - 1)
            _X = X[randomi]
            _y = y[randomi]
        elif method.lower() == "minibatch":
            minibatch_indices = []
            for i in range(minibatch_size):
                minibatch_indices.append(randint(0, len(X) - 1))
            minibatch_indices = torch.tensor(minibatch_indices).unique()
            _X = torch.index_select(X, 0, minibatch_indices)
            _y = torch.index_select(y, 0, minibatch_indices)
        else:
            raise (NameError("Kwarg 'method' must be either 'batch' or 'stochastic'"))

        # make predictions
        pred = model.forward(_X, outMethod=outMethod)

        # calculate the loss
        loss = loss_func(pred, _y)

        # deal with regularization
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters()).item()
        l1_norm = sum(p.abs().sum() for p in model.parameters()).item()
        loss = loss + (_lambdaL2 * l2_norm) + (_lambdaL1 * l1_norm)

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        full_loss = loss_func(model.forward(X, outMethod=outMethod), y).item()
        print(full_loss)
        train_loss.append(full_loss)
        epoch += 1

        if epoch % 50 == 0 and verbosity:
            print(
                'epoch: {} | loss: {} | target: {}'.format(
                    epoch, round(full_loss, 4), loss_target), end="\r"
            )

    if full_loss <= loss_target:
        reason = "loss small enough!"
    elif pd.isna(full_loss):
        reason = "loss function breaking"
    else:
        reason = "max epoch reached ({})".format(max_epoch)

    if verbosity:
        print("\nTraining complete! : {}".format(reason))  # print we're complete and reason the training stopped
        print(
            'Final epoch: {} | Final loss: {} | target: {}'.format(
                epoch, round(full_loss, 4), loss_target)
        )

    if plot:
        f = plt.figure()
        plt.plot(train_loss)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.suptitle('Training Loss vs Epoch')
        plt.show()
    return train_loss


def generate_learning_curve(train, valid, model, loss_func, optimizer, max_epoch,
                            method="batch", plot=True,
                            _lambdaL1=0, _lambdaL2=0, minibatch_size=100,
                            outMethod=True):
    '''
    for a model, with train data and valid data + other params, plot a training curve using loss as
    the metric. Plots training epoch vs loss separately for training data and validation data.
    '''

    epochs = range(1, max_epoch, 1000)
    D_epoch = epochs[1] - epochs[0]

    Noutputs = model._layer_sizes[-1]

    # set up the data
    Xtrain = train[0]
    ytrain = train[1].squeeze()
    #ytrain_bin = tu.labels_to_binary(ytrain, Noutputs)
    Xvalid = valid[0]
    yvalid = valid[1].squeeze()
    #yvalid_bin = tu.labels_to_binary(yvalid, Noutputs)

    train_losses = []  # containers to plot learning curves
    valid_losses = []
    print('Generating Learning Curves for {}'.format(type(model).__name__))
    for each in epochs:
        # train our model for another D_epochs using training data.
        # Set loss_target negative so its guaranteed to train for D_epochs
        losses = trainNN(copy.deepcopy(train), model, loss_func, optimizer,
                         max_epoch=D_epoch,
                         loss_target=-1,
                         method=method,
                         minibatch_size=minibatch_size,
                         plot=False,
                         verbosity=False,
                         _lambdaL1=_lambdaL1,
                         _lambdaL2=_lambdaL2,
                         outMethod=outMethod)

        # calculate training loss
        train_preds = model.forward(Xtrain, outMethod=outMethod)
        train_loss = loss_func(train_preds, ytrain.type(torch.long)).item()

        # calculate validation loss
        valid_preds = model.forward(Xvalid, outMethod=outMethod)
        valid_loss = loss_func(valid_preds, yvalid.type(torch.long)).item()

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print('Epoch: {} | training loss: {} | validation loss: {}'.format(
            each, round(train_loss, 4), round(valid_loss, 4)), end='\r')
    print('\n')

    if plot:
        f = plt.figure()
        plt.plot(epochs, train_losses, label='training loss')
        plt.plot(epochs, valid_losses, label='validation loss')
        plt.xlabel('Training Epoch')
        plt.ylabel('Loss')
        plt.suptitle("Learning Curve")
        plt.legend()
        plt.show()

    return train_losses, valid_losses
