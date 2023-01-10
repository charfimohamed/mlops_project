import logging
import os
import sys
import timm
#import hydra
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim import sgd
from torch.utils.data import Dataset, DataLoader
from src.models.model import CatDogModel
from src.data import make_dataset
from src.data.make_dataset import CatDogDataset
import wandb
from pathlib import Path
from torchvision import transforms
from pytorch_lightning import Trainer
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import pprint
#from wandb.sweeps import HyperParameterSweep

#sys.path.append("..")
log = logging.getLogger(__name__)
#print = log.info

wandb.init(project="test-project", entity="group18_mlops")
wandb.config = {
  "lr": 0.001,
  "epochs": 10,
  "batch_size": 32,
  "dropout": 0.2
}

def compute_validation_metrics(model, dataloader, loss_fn):
    model.eval()
    total_loss = 0
    total_acc = 0
    nb_samples = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            total_acc += (preds == labels).sum().item()
            nb_samples += preds.shape[0]
    return total_loss/nb_samples , total_acc/nb_samples

#training_function

def train (batch_size = wandb.config["batch_size"], epochs = wandb.config["epochs"], lr = wandb.config["lr"]):
    ''' Trains a neural network from the TIMM framework'''
    #DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CatDogModel()
    #model.to(DEVICE)
    
    image_size = 64
    data_resize = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()])

    train_dataset = CatDogDataset(split="train", in_folder=Path("../data/raw"), out_folder=Path('../data/processed'), transform=data_resize)
    validation_dataset = CatDogDataset(split="validation", in_folder=Path("../data/raw"), out_folder=Path('../data/processed'), transform=data_resize)
   
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size = batch_size, shuffle=True)
    optimizer = Adam(model.parameters(),lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
   
    
    for epoch in range(wandb.config["epochs"]):
        print("epoch{i}".format(i=epoch))
        total_loss = 0
        for i, (images, labels) in enumerate(train_dataloader):
            output = model(images)
            print(f"step {i}")
            #print(labels)
            loss = loss_fn(output,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss, train_acc = compute_validation_metrics(model,train_dataloader,loss_fn)
        val_loss, val_acc = compute_validation_metrics(model,validation_dataloader,loss_fn)
        print("training accuracy = {i}".format(i=train_acc))
        print("validation accuracy = {i}".format(i=val_acc))

        wandb.log({
        'epoch': epoch, 
        'train_acc': train_acc,
        'validation_acc': val_acc,
        'train_loss': train_loss,
        'validation_loss': val_loss})
        #print('Average loss for epoch : {i}'.format(i=total_loss/len(train_loader)))
    
    return model  



if __name__ == "__main__":

    sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters': 
     {
        'batch_size': {'values': [16, 32, 64]},
        'epochs': {'values': [5, 10, 15]},
        'lr': {'max': 0.1, 'min': 0.0001},
        'dropout':{'values': [0.2, 0.4, 0.6]},
        'optimizer': {'values': ['adam', 'sgd']}

     }
    }
    pprint.pprint(sweep_configuration)

    # Create a sweep
    sweep_id = wandb.sweep(sweep_configuration, project="group18_mlops")
   

    
    
train()  # training function call
        
    # Run the sweep
wandb.agent(sweep_id, function="train()", count=4)

wandb.finish()