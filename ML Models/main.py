import feedforward as ff
import datareader as dr
import transformation_utils as tu
import validation_utils as vu
from torch import nn
import torch
import torch.optim.lr_scheduler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def compound_neural_net():
    '''
    2 neural nets -> first separates HD from no-HD. Second separates Stage C from stage D
    '''

    catdata = dr.read_cat()

    train, valid, test = dr.generate_sets(catdata, splits=[70, 15, 15])
    train, valid, test = tu.scale01(train, [train, valid, test])
    trainHD, validHD = tu.joinclass(train, 1, 2), tu.joinclass(valid, 1, 2)
    trainCD, validCD = tu.filter(train, [1, 2]), tu.filter(valid, [1, 2])
    trainHD = tu.equalize_portions(trainHD) # make sure we have equal numbers of classes in the training set
    trainCD = tu.equalize_portions(trainCD)

    netHD = ff.FeedForwardSoftmax(len(trainHD[0][0]), 2, hiddenNs=[20])
    netCD = ff.FeedForwardSoftmax(len(trainCD[0][0]), 2, hiddenNs=[20])

    loss = nn.CrossEntropyLoss()
    optimizerHD = torch.optim.SGD(netHD.parameters(), lr=1e-1, momentum=0.9)
    optimizerCD = torch.optim.SGD(netCD.parameters(), lr=1e-1, momentum=0.9)
    schedulerHD = torch.optim.lr_scheduler.ConstantLR(optimizerHD, factor=0.2222222, total_iters=5, last_epoch=- 1, verbose=False)
    schedulerCD = torch.optim.lr_scheduler.ConstantLR(optimizerHD, factor=0.2222222, total_iters=5, last_epoch=- 1, verbose=False)
    optimizerHD.step()
    schedulerHD.step()
    optimizerCD.step()
    optimizerHD.step()
    
    #ff.generate_learning_curve(trainHD, validHD, netHD, loss, optimizerHD,
    #                           max_epoch=50000,
    #                           method='minibatch',
    #                           minibatch_size=300,
    #                           outMethod=False,
    #                           _lambdaL1=0,
    #                           _lambdaL2=0)

    #ff.generate_learning_curve(trainCD, validCD, netCD, loss, optimizerCD,
    #                           max_epoch=50000,
    #                           method='minibatch',
    #                           minibatch_size=300,
    #                           outMethod=False,
    #                           _lambdaL1=0,
    #                           _lambdaL2=0)


    lossesHD = ff.trainNN(trainHD, netHD, loss, optimizerHD,
                        max_epoch=100000,
                        loss_target=0.25,
                        method='minibatch',  # pick "batch" or "stochastic" or "minibatch"
                        minibatch_size=300,
                        plot=True,
                        verbosity=True,

                        # L1 and L2 regularisation strengths. NB: you can use a combination of both - this is called an
                        # elastic net
                        _lambdaL1=0.,
                        _lambdaL2=0.,
                        outMethod = False)

    lossesCD = ff.trainNN(trainCD, netCD, loss, optimizerCD,
                        max_epoch=100000,
                        loss_target=0.25,
                        method='minibatch',  # pick "batch" or "stochastic" or "minibatch"
                        minibatch_size=300,
                        plot=True,
                        verbosity=True,

                        # L1 and L2 regularisation strengths. NB: you can use a combination of both - this is called an
                        # elastic net
                        _lambdaL1=0.,
                        _lambdaL2=0.,
                        outMethod = False)


    HD_predictions = tu.binary_to_labels(netHD.forward(valid[0], outMethod=True)) #predictions of HD
    CD_predictions = tu.binary_to_labels(netCD.forward(valid[0], outMethod=True)) #predictions of Stage C or D
    test_predictions = CD_predictions + 1
    test_predictions[HD_predictions==0] = 0

    test_true = valid[1].squeeze()

    cm = vu.confusion_matrix(test_true,test_predictions,classes = [0,1,2])
    F10 = vu.precision_recall_F1(cm,0)[2]
    F11 = vu.precision_recall_F1(cm,1)[2]
    F12 = vu.precision_recall_F1(cm,2)[2]

    print('--- confusion matrix ---')
    print(cm)

    print('F-measures \n no HF: {} \n Stage C: {} \n Stage D: {}'.format(F10,F11,F12))

    return None


def neural_net():
    '''
    NOTES
    -> because we have a lot of features, some that may not be relevant, i want to try L1 regularization
        L1 Functionality is now added into the ff.trainNN() function. I have put in L1 and L2 regularisation terms.
        Note that if we have too much time on our hands, we can use both of these in tandem/combination with eachother.
        This is a common method called an "elastic net", which gives a bit of the benefit of both L1 and L2 methods.

    -> For L1 regularisation in particular, we should progress by cross-validating the dataset into multiple parts
    to work out which parameters are useless with a higher degree of certainty. Some modules normally have this functionality
    built in, but I'm not sure about pytorch.
    '''

    catdata = dr.read_cat()

    train, valid, test = dr.generate_sets(catdata, splits=[70, 15, 15])
    train, valid, test = tu.scale01(train, [train, valid, test])


    # lets make a simple feed forward NN with one hidden layer, softmax output
    net = ff.FeedForwardSoftmax(len(train[0][0]), 3, hiddenNs=[20])
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

    # ff.generate_learning_curve(train, valid, net, loss, optimizer,
    #                            max_epoch=100000,
    #                            method='minibatch',
    #                            minibatch_size=300,
    #                            outMethod=False,
    #                            _lambdaL1=0,
    #                            _lambdaL2=0)

    losses = ff.trainNN(train, net, loss, optimizer,
                            max_epoch=100000,
                            loss_target=0.4,
                            method='minibatch',  # pick "batch" or "stochastic" or "minibatch"
                            minibatch_size=300,
                            plot=True,
                            verbosity=True,

                            # L1 and L2 regularisation strengths. NB: you can use a combination of both - this is called an
                            # elastic net
                            _lambdaL1=1.,
                            _lambdaL2=0,
                            outMethod = False)

    test_predictions = tu.binary_to_labels(net.forward(valid[0], outMethod=True))
    test_true = valid[1].squeeze()

    cm = vu.confusion_matrix(test_true, test_predictions, classes=[0, 1, 2])
    F10 = vu.precision_recall_F1(cm, 0)[2]
    F11 = vu.precision_recall_F1(cm, 1)[2]
    F12 = vu.precision_recall_F1(cm, 2)[2]

    print('--- confusion matrix ---')
    print(cm)

    print('F-measures \n no HF: {} \n Stage C: {} \n Stage D: {}'.format(F10, F11, F12))

    return None


def vote_score():
    '''
    Beginning is all data prep.
    After this, data stored as (train_df, test_df) and labels as (train_labels, test_labels)
    FYI - I tried PCA but there is basically a linear increase in the explained variance by increasing dimensions
    in our dataset. It didn't help by cutting any dimensions.
    In the end, random forest, support vector machine, and gradient boosting classifier were best.
    '''

    data = dr.read('deid_full_data_cont.csv')

    data.drop(columns='id', axis=1, inplace=True)  # do not need id
    data['gender'].replace(to_replace=['Male', 'Female', ], value=[0, 1], inplace=True)  # make gender/smoke numeric
    data['bnp'].replace(to_replace=['a'], value=['nan'], inplace=True)  # typo in BNP data somewhere
    smoke_cat = data['smoke'].unique()
    data['smoke'].replace(to_replace=smoke_cat, value=np.arange(11), inplace=True)

    # exclude = ['bnp', 'a1c','chol'] # dropping highly missing data didn't make difference
    # data.drop(columns=exclude, axis=1, inplace=True)

    train = data.sample(frac=0.8, random_state=2)
    test = data.drop(train.index)

    train_data, train_Y = dr.split_hyperparams_target(train, 'stage')  # split data from label
    test_data, test_Y = dr.split_hyperparams_target(test, 'stage')

    final_col = train_data.columns  # saving column names for later

    simp_imp = SimpleImputer(strategy='mean').fit(train_data)  # impute missing values as mean
    train_imp = simp_imp.transform(train_data)
    test_imp = simp_imp.transform(test_data)


    scaler = StandardScaler().fit(train_imp) # scale to mean 0, std 1
    train_clean = scaler.transform(train_imp)
    test_clean = scaler.transform(test_imp)

    train_df = pd.DataFrame(train_clean)  # make things df again
    test_df = pd.DataFrame(test_clean)
    train_df.columns = final_col
    test_df.columns = final_col

    train_labels = [x[0] for x in train_Y.to_numpy()]  # convert to format sklearn likes
    test_labels = [x[0] for x in test_Y.to_numpy()]

    # Finally, training models!

    rf = RandomForestClassifier(random_state = 0, n_estimators=1000, max_features=0.5,oob_score=True)
    svm = SVC(kernel='poly', degree=3, coef0=1, C=5)
    gbt = GradientBoostingClassifier(max_depth=1, n_estimators=1000,learning_rate=0.5)
    voting_clf = VotingClassifier(estimators=[('rf',rf), ('svc',svm), ('gbt',gbt)],voting='hard')

    models = [rf, svm, gbt, voting_clf]

    for model in models:
        model.fit(train_df, train_labels)
        predicts = model.predict(test_df)

        predicts_tensor = torch.tensor(predicts)
        test_labels_tensor = torch.tensor(test_labels)
        conf_matrix = vu.confusion_matrix(predicts_tensor, test_labels_tensor)
        classes = {0: 'Not HF', 1: 'Stage C', 2: 'Stage D'}

        print(model)

        for each in classes:
            precision, recall, f1 = vu.precision_recall_F1(conf_matrix, each)
            print(f'{classes[each]}: Precision: {round(precision, 2)}, Recall: {round(recall, 2)}, F1: {round(f1, 2)}')

        print('----------------------------------------------------------------')


def univariate():
    '''
    function to test the univariate relationships between the continuous data and the target variables
    '''
    X, y = dr.read_cont(dropna=True)  # load data
    X = X.drop(columns=['id', 'gender', 'ace_arb', 'aldo', 'bb', 'ino', 'loop', 'arni', 'sglt2', 'stat', 'thia',
                        'xanthine', 'albumin', 'bnp', 'smoke'])  # drop non-continuous hyperparams

    train, valid, test = dr.generate_sets((X, y), splits=[70, 15, 15])

    # I want to drop the top/bottom 3 values for each parameter -> testing showed these ruined the plots
    train = tu.cut_topbottom(train, 100)

    tu.univariate(train, X.columns.to_list(), {0: "Stage 0", 1: "Stage 1", 2: "Stage 2"})


if __name__ == '__main__':
    # univariate()
    # compound_neural_net()
    neural_net()
    # vote_score()









