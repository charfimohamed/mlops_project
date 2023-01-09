import torch
from pytorch_lightning import LightningModule
from torch import nn, optim
import timm
from torch.utils.data import Dataset,dataloader
import numpy as np
import os
import matplotlib as plt


class CatDogModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('ens_adv_inception_resnet_v2', num_classes = 2 ,pretrained=True)

    def forward(self,x):
        return self.model(x)

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