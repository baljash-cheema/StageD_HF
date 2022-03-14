import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class staged_dataset(Dataset):
    '''
    Subclass from torch.utils.data.Dataset.
    Needs to have __len__ and __getitem__ to qualify as a "Dataset" in PyTorch terms.
    Took all categorical data and did one-hot encoded it.
    '''

    def __init__(self):
        super().__init__()

        df = pd.read_csv('data_cat.csv')
        df.drop(['age', 'id'], axis=1, inplace=True)

        df2 = pd.get_dummies(df) # one-hot encode data

        train_val = df2.sample(frac=0.9, random_state=2) # 90% train/val, 10% test
        test = df2.drop(train_val.index)

        train = train_val.sample(frac=0.8, random_state=2) #80% train, 20% of original 90%
        val = train_val.drop(train.index)

        # all stored as dfs
        self.train_data = train.iloc[:, 1:]
        self.train_label = train.iloc[:, 0]
        self.val_data = val.iloc[:, 1:]
        self.val_label = val.iloc[:, 0]
        self.test_data = test.iloc[:, 1:]
        self.test_label = test.iloc[:, 0]
        self.full_data = df2.iloc[:,1:]
        self.full_label = df2.iloc[:,0]

    def __len__(self):
        return len(self.full_data)

    def __getitem__(self, item):
        return self.full_data.iloc[item], self.full_label.iloc[item] # returns items as df

if __name__ == "__main__":
    dataset = staged_dataset()


