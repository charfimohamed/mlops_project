import torch
from pytorch_lightning import LightningModule
from torch import nn, optim
import timm
from torch.utils.data import Dataset,dataloader
import numpy as np
import os
import matplotlib as plt
import torch.nn.functional as F


class CatDogModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear((64**2)*3, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x=x.view(x.shape[0],-1)
        return F.softmax(self.model(x))

    #def training_step(self, batch, batch_idx):
    #    data, target = batch
    #    preds = self(data)
    #    loss = self.loss_fn(preds,target)
    #    return loss
#
    #def validation_step(self, batch, batch_idx):
    #    data, target = batch
    #    preds = self(data)
    #    preds = torch.argmax(preds,dim=1)
    #    correct = (preds==target).sum()
    #    val_accuracy = correct/len(target)
    #    return {'validation accuracy': val_accuracy}
#
    #def test_step(self, batch, batch_idx):
    #    data, target = batch
    #    preds = self(data)
    #    preds = torch.argmax(preds,dim=1)
    #    correct = (preds==target).sum()
    #    test_accuracy = correct/len(target)
    #    return {'test accuracy': test_accuracy}