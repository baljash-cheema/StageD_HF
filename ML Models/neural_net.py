import datareader as dr
import transformation_utils as tu
import validation_utils as vu
from torch import nn
import torch
import matplotlib.pyplot as plt

class net(nn.Module):
    '''
    Fully connected feedforward neural network.
    Added dropout technique.
    Create neural net object of your own, subclass torch.Module or renamed as nn.Module.
    '''

    # reference parent class
    # establish structure of network
    def __init__(self,size_in,size_out):
        super().__init__()
        self.architecture = nn.Sequential(
            nn.Linear(size_in, 100),
            nn.Dropout(p=0.75),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.Dropout(p=0.75),
            nn.LeakyReLU(),
            nn.Linear(100,size_out),
        )

    # establish forward pass through network
    def forward(self, x):
        x = self.architecture(x)
        return x

def training_loop(train_data, valid_data, model, loss_function, optimizer):
    '''
    This function takes training data in the form of list with [data,label], validation data in same format,
    neural net model, loss function, and optimizer, and trains a net while providing training and validation error
    as outputs as a tuple (train error, val error).
    '''

    # forward pass
    model.train()

    train_predict = model(train_data[0]) # do not use .forward method

    # calculate loss
    train_loss = loss_function(train_predict, train_data[1])
    train_loss_ = train_loss.item() # for tracking loss over time in list

    # regularization
    l2_norm = sum(p.pow(2.0).sum() for p in model.parameters()).item()
    l1_norm = sum(p.abs().sum() for p in model.parameters()).item()
    train_loss = train_loss + l2_norm + l1_norm  # elastic net

    # backprop
    optimizer.zero_grad() # clear out old gradients
    train_loss.backward() # calculate gradients going backwards
    optimizer.step() # adjust weights, according to learning rate

    model.eval()  # calculate validation error without adjusting gradients

    with torch.no_grad(): # make sure gradients aren't tracked
        valid_predict = model(valid_data[0])

    valid_loss = loss_function(valid_predict, valid_data[1])

    return (train_loss_, valid_loss)

def main():
    # read data
    catdata = dr.read_cat()

    # split data and scale
    train, valid, test = dr.generate_sets(catdata, splits=[70, 15, 15])
    train, valid, test = tu.scale01(train, [train, valid, test])

    # split data and label
    train_x = train[0]
    train_y = train[1].squeeze().to(torch.long)

    valid_x = valid[0]
    valid_y = valid[1].squeeze().to(torch.long)

    # establish net, loss function, lr, and optimizer
    neural_net = net(len(train[0][0]), 3)
    loss_function = nn.CrossEntropyLoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(neural_net.parameters(), lr=learning_rate)

    # empty lists for training/validation error
    train_loss, valid_loss = [],[]

    # track epochs
    epoch = 0
    num_epoch = 1

    # training loop
    for each in range(num_epoch):
        train_loss_, valid_loss_ = training_loop(train_data=[train_x,train_y],
                                                valid_data=[valid_x,valid_y],
                                                model=neural_net,
                                                loss_function=loss_function,
                                                optimizer=optimizer)
        train_loss.append(train_loss_)
        valid_loss.append(valid_loss_)

        if epoch % 100 == 0:
            print(f'Epoch: {epoch} --- Training Loss: {train_loss_} --- Validation Loss: {valid_loss_} ')

        epoch += 1

        if valid_loss_ < 0.44:
            break

    # printing total number of parameters in the model
    param_list = [p.numel()
                  for p in neural_net.parameters()
                  if p.requires_grad == True]
    print(sum(param_list), param_list)

    # plot train/valid loss
    f = plt.figure()
    plt.plot(train_loss, label = 'Train loss')
    plt.plot(valid_loss, label = 'Test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.suptitle('Training Loss vs Epoch')
    plt.show()

    # confusion matrix per HF class
    predicts = tu.binary_to_labels(neural_net(valid[0]))
    conf_matrix = vu.confusion_matrix(predicts, valid[1].squeeze().to(torch.long))
    classes = {0: 'Not HF', 1: 'Stage C', 2: 'Stage D'}

    for each in classes:
        precision, recall, f1 = vu.precision_recall_F1(conf_matrix, each)
        print(f'{classes[each]}: Precision: {round(precision, 2)}, Recall: {round(recall, 2)}, F1: {round(f1, 2)}')

if __name__ == "__main__":

    main()
    # catdata = dr.read_cat()
    #
    # # split data and scale
    # train, valid, test = dr.generate_sets(catdata, splits=[70, 15, 15])
    # train, valid, test = tu.scale01(train, [train, valid, test])
    #
    # # split data and label
    # train_x = train[0]
    # train_y = train[1].squeeze().to(torch.long)
    #
    # valid_x = valid[0]
    # valid_y = valid[1].squeeze().to(torch.long)





