import torch
from pytorch_lightning import LightningModule
from torch import nn, optim
import timm
from torch.utils.data import Dataset,Dataloade
import numpy as np
import os
import matplotlib as plt


class CatDogModel(LightningModule):
    def _init (self,num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.model = timm.create_model('inceptionresnet_v2', pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, out_features = num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self,x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.loss_fn(preds,target)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        preds = torch.argmax(preds,dim=1)
        correct = (preds==target).sum()
        val_accuracy = correct/len(target)
        return {'validation accuracy': val_accuracy}

    def test_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        preds = torch.argmax(preds,dim=1)
        correct = (preds==target).sum()
        test_accuracy = correct/len(target)
        return {'test accuracy': test_accuracy}

    def configure_optimizers (self):
        return optim.Adam(self.parameters(), lr=1e-3)