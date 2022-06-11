import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from ignite.metrics import Accuracy

from models import get_model
from eval import DetectionEvaluator
from data import ImageDetectionDemoDataset
from util import constants as C
from .logger import TFLogger

import pdb

class DetectionTask(pl.LightningModule, TFLogger):
    """Standard interface for the trainer to interact with the model."""

    def __init__(self, params):
        super().__init__() #Initialize parent classes (pl.LightningModule params)
        self.save_hyperparameters(params) #Save hyperparameters to experiment directory, pytorch lightning function
        self.model = get_model(params) #Instantiates model from model folder
        self.evaluator = DetectionEvaluator()

    def training_step(self, batch, batch_nb): #Batch of data from train dataloader passed here
        losses = self.model.forward(batch)
        loss = torch.stack(list(losses.values())).mean()
        return loss

    def validation_step(self, batch, batch_nb): #Called once for every batch

        losses = self.model.forward(batch)
        loss = torch.stack(list(losses.values())).mean()
        preds = self.model.infer(batch)
        self.evaluator.process(batch, preds)
        return loss

    def validation_epoch_end(self, outputs): #outputs are loss tensors from validation step
        avg_loss = torch.stack(outputs).mean()
        self.log("val_loss", avg_loss)
        metrics = self.evaluator.evaluate()
        self.evaluator.reset()
        self.log_dict(metrics, prog_bar=True)

    def test_step(self, batch, batch_nb):
        preds = self.model.infer(batch)
        self.evaluator.process(batch, preds)

    def test_epoch_end(self, outputs):
        metrics = self.evaluator.evaluate()
        self.log_dict(metrics)
        return metrics

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=0.02)]

    def train_dataloader(self): #Called during init
        dataset = ImageDetectionDemoDataset() #For specific examples
        return DataLoader(dataset, shuffle=True, #For entire batch
                          batch_size=2, num_workers=8,
                          collate_fn=lambda x: x)

    def val_dataloader(self): #Called during init
        dataset = ImageDetectionDemoDataset() 
        return DataLoader(dataset, shuffle=False,
                batch_size=1, num_workers=8, collate_fn=lambda x: x)

    def test_dataloader(self): #Called during init
        dataset = ImageDetectionDemoDataset() 
        return DataLoader(dataset, shuffle=False,
                batch_size=1, num_workers=8, collate_fn=lambda x: x)

    #Process
    #1. Call Trainer.fit
    #2. Run validation step twice
    #3. Run validation epoch end as a dummy run
    #4. Call training step for all training batches, till end of epoch
    #5. Call validation step for all validation batches till validation batches exhausted
    #6. Compute  validation metric at validation epoch end, log it, save checkpoint if validation metrics improve
    #7. Return to training step 4 and continue

    #Test steps are only called when you call main.py