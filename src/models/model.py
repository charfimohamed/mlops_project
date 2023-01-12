import torch
from pytorch_lightning import LightningModule
from torch import nn, optim
import timm
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
        self.im_size = 64
        #self.conv1 =  nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1) #Shape= (32, 12, im_size, im_size)
        #self.norm1 = nn.BatchNorm2d(num_features=12) #Shape= (32, 12, im_size, im_size)
        #self.relu1 = nn.ReLU() #relu
#
        #self.pool1 = nn.MaxPool2d(kernel_size=2) #Reduce the image size be factor 2 Shape= (32, 12, im_size/2, im_size/2)
#
        #self.conv2 = nn.Conv2d(in_channels=12,out_channels=20,kernel_size=3,stride=1,padding=1) #Shape= (32,20, im_size/2, im_size/2)
        #self.relu2 = nn.ReLU() #Shape= (32,20,64,64)
#
        #self.conv3 = nn.Conv2d(in_channels=20,out_channels=32, kernel_size = 3, stride=1, padding=1) #Shape= (32,32,im_size/2,im_size/2)
        #self.norm3 = nn.BatchNorm2d(num_features=32) #Shape= (32, 32, im_size/2, im_size/2)
        #self.relu3 = nn.ReLU() #Shape= (32, 32, im_size/2, im_size/2)
#
        #self.fc1 = nn.Linear(in_features=32*((int(self.im_size/2))**2),out_features=1024)
        #self.fc2 = nn.Linear(in_features=1024,out_features=32)
        #self.fc3 = nn.Linear(in_features=32,out_features=2)
#
        #self.dropout = nn.Dropout2d(0.3)


    def forward(self, x):
        # make sure input tensor is flattened
        #x = self.conv1(x)
        #x = self.dropout(x)
        #x = self.norm1(x)
        #x = self.relu1(x)
        #x = self.pool1(x)
        #x = self.conv2(x)
        #x = self.dropout(x)
        #x = self.relu2(x)
        #x = self.conv3(x)
        #x = self.dropout(x)
        #x = self.norm3(x)
        #x = self.relu3(x)
        #x = x.view(x.shape[0], -1)
        #x = self.fc1(x)
        #x = self.fc2(x)
        #x = self.fc3(x)
        x = self.model(x)
        return x
