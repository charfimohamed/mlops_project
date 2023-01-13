import logging
import os
import sys
import timm
#import hydra
import wandb
import pprint
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim import SGD

from torch.utils.data import Dataset, DataLoader
from src.models.model import CatDogModel
from src.data import make_dataset
from src.data.make_dataset import CatDogDataset
from pathlib import Path
from torchvision import transforms
from pytorch_lightning import Trainer
import matplotlib.pyplot as plt
import numpy as np

#sys.path.append("..")
log = logging.getLogger(__name__)
#print = log.info


def save_checkpoint(model,epoch,best_accuracy):
    print("------> saving checkpoint <------")
    state = {
    'epoch': epoch + 1,
    'model' : model.state_dict(),
    'best accuracy': best_accuracy,
    }
    torch.save (state, 'model_best_checkpoint.pth')

def train_hp():
    wandb.init(project="test-project", entity="group18_mlops")
    train(batch_size=wandb.config.batch_size, epochs=5, lr=wandb.config.lr,optimizer_name=wandb.config.optimizer)

#training_function
def train (batch_size = 32, epochs = 5, lr = 0.001, optimizer_name='adam'):
    ''' Trains a neural network from the TIMM framework'''
    #DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = CatDogModel()
    #model.to(DEVICE)
    model = CatDogModel()
    image_size = model.im_size
    data_resize = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    train_dataset = CatDogDataset(split="train", in_folder=Path("../../data/raw"), out_folder=Path('../../data/processed'), transform=data_resize)
    validation_dataset = CatDogDataset(split="validation", in_folder=Path("../../data/raw"), out_folder=Path('../../data/processed'), transform=data_resize)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size = batch_size, shuffle=True)
    if(optimizer_name=='sgd'):
        optimizer = SGD(model.parameters(),lr=lr)
    if(optimizer_name=='adam'):
        optimizer = Adam(model.parameters(),lr=lr)

    loss_fn = torch.nn.CrossEntropyLoss()
    best_accuracy = 0.0 # accuracy of the best epoch to know what wights to save 
    for epoch in range(epochs):
        print(f"epoch : {epoch+1}/{epochs}")
        model.train()
        train_accuracy = 0.0 
        train_loss = 0.0 
        validation_accuracy = 0.0
        nb_train_samples = 0
        nb_valid_samples = 0
        for i,(images, labels) in enumerate(train_dataloader) :
            print(f" train step {i}")
            outputs = model(images)
            optimizer.zero_grad()
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, dim=1)
            train_loss+= loss
            train_accuracy+=torch.sum(preds==labels)
            nb_train_samples += preds.shape[0]
        train_loss = train_loss / nb_train_samples
        train_accuracy = train_accuracy /nb_train_samples
        print(f"Epoch {epoch+1}/{epochs}. Loss: {train_loss} . accuracy : {train_accuracy}")
        model.eval()
        for i,(images, labels) in enumerate(validation_dataloader) :
            print(f"validation step {i}")
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)
            validation_accuracy+=torch.sum(preds==labels)
            nb_valid_samples += preds.shape[0]
        validation_accuracy = validation_accuracy /nb_valid_samples
        print(f"validation accuracy : {validation_accuracy}")
        if(validation_accuracy>best_accuracy):
            best_accuracy=validation_accuracy
            save_checkpoint(model,epoch,best_accuracy)

        wandb.log({
        'train_acc': train_accuracy,
        'validation_acc': validation_accuracy,
        'best_accuracy':best_accuracy,
        'train_loss': train_loss,}) 
    return model  
           

if __name__ == "__main__":

    sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'best_accuracy'},
    'parameters': 
     {
        'batch_size': {'values': [16, 32, 64]},
        'lr': {'max': 0.001, 'min': 0.0001},
        'optimizer': {'values': ['adam', 'sgd']}

     }
    }
    pprint.pprint(sweep_configuration)

    # Create a sweep
    sweep_id = wandb.sweep(sweep_configuration, project="group18_mlops")
   
    
    #train()  # training function call
        
    # Run the sweep
    wandb.agent(sweep_id, function=train_hp, count=4)

    wandb.finish()