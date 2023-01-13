import torch
from pytorch_lightning import LightningModule
from torch import nn, optim
from torch.utils.data import Dataset,dataloader
import numpy as np
import os
import matplotlib as plt
import torch.nn.functional as F
import torchvision

class CatDogModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model= torchvision.models.resnet50(pretrained=True)
        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_ftrs, 2)
        self.im_size = 128

    def forward(self, x):
        x = self.model(x)
        return x
