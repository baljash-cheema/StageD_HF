import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from dataset import staged_dataset
from dataloader import staged_dataloader
import torchmetrics
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers

class staged_ff(pl.LightningModule):
    '''
    This is the neural network itself.
    '''
    def __init__(self,size_in,size_out):
        super().__init__()
        self.input = nn.Linear(size_in,100)
        self.h1 = nn.Linear(100,100)
        self.h2 = nn.Linear(100,size_out)
        self.loss = nn.CrossEntropyLoss()

    def forward(self,x):
        x = F.leaky_relu(self.input(x))
        x = F.leaky_relu(self.h1(x))
        x = self.h2(x)
        return x

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        data,label = batch
        data = data.to(dtype=torch.float)
        label = label.to(dtype=torch.long)
        logits = self.forward(data)
        loss = self.loss(logits,label)
        accuracy = torchmetrics.functional.accuracy(logits,label)
        tensorboard_logs = {'acc': {'train': accuracy.detach()}, 'loss': {'train': loss.detach()}}
        self.log("training loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("training accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "log": tensorboard_logs}


    def validation_step(self, batch, batch_idx):
        data,label = batch
        data = data.to(dtype=torch.float)
        label = label.to(dtype=torch.long)
        logits = self.forward(data)
        loss = self.loss(logits,label)
        accuracy = torchmetrics.functional.accuracy(logits,label)
        tensorboard_logs = {'acc': {'val': accuracy.detach()}, 'loss': {'val': loss.detach()}}
        self.log("validation loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("validation accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        data,label = batch
        data = data.to(dtype=torch.float)
        label = label.to(dtype=torch.long)
        logits = self.forward(data)
        loss = self.loss(logits,label)
        accuracy = torchmetrics.functional.accuracy(logits,label)
        tensorboard_logs = {'acc': {'test': accuracy.detach()}, 'loss': {'test': loss.detach()}}
        self.log("test loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "log": tensorboard_logs}

if __name__ == "__main__":
    dataset = staged_dataset()
    dataloader = staged_dataloader(dataset=dataset, batch_size=32)
    neural_network = staged_ff(dataset.train_data.shape[1], 3)

    tb_logger = pl_loggers.TensorBoardLogger("./lightning_logs/", name="staged ff")

    trainer = pl.Trainer(logger=tb_logger, max_epochs=10)
    trainer.fit(neural_network, dataloader)

    result = trainer.test(neural_network, dataloader)
    print(result)
