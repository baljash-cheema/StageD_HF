import numpy as np
import pandas as pd
import torch
from dataset import staged_dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl

class staged_dataloader(pl.LightningDataModule):
    '''
    Dataloader expects a PyTorch dataset.
    Wraps an iterable around the dataset.
    Data stored as [...[data,label]...] in PyTorch tensors.
    '''

    def __init__(self, dataset, batch_size):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def setup(self,stage=None):
        # [...[data,label]...], convert df to numpy
        self.train = [[each[0], each[1]] for each in zip(self.dataset.train_data.to_numpy(), self.dataset.train_label.to_numpy())]
        self.val = [[each[0], each[1]] for each in zip(self.dataset.val_data.to_numpy(), self.dataset.val_label.to_numpy())]
        self.test = [[each[0], each[1]] for each in zip(self.dataset.test_data.to_numpy(), self.dataset.test_label.to_numpy())]

    # dataloaders convert numpy arrays to tensors
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False)

if __name__ == "__main__":
    dataset = staged_dataset()
    dataloader = staged_dataloader(dataset=dataset, batch_size=32)
