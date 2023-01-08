import logging
import os
import sys

#import hydra
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from model import CatDogModel
from src.data import make_dataset
from src.data.make_dataset import CatDogDataset
import wandb
from pathlib import Path
from torchvision import transforms


#sys.path.append("..")
#log = logging.getLogger(__name__)
#print = log.info


#training_function
def train (batch_size = 32, epochs = 10, lr = 0.03):
    ''' Trains a neural network from the TIMM framework'''

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = CatDogModel(num_classes =2)
    model.to(DEVICE)
    dataset = make_dataset(split = "train", )
    

    image_size = 224
    data_resize = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()])

    train_dataset = CatDogDataset(split="train", in_folder=Path("../data/raw"), out_folder=Path('../data/processed'), transform=data_resize)
    validation_dataset = CatDogDataset(split="validation", in_folder=Path("../data/raw"), out_folder=Path('../data/processed'), transform=data_resize)
   
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size = batch_size, shuffle=True)


    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_dataloader:
            
            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output,labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = model.training_step(batch = train_dataloader , batch_idx= epoch)
        val_acc = model.validation_step(batch = validation_dataloader , batch_idx= epoch)

        #wandb.log({
            #'epoch': epoch, 
            #'train_acc': train_acc,
            #'train_loss': train_loss, 
            #'val_acc': val_acc, 
            #'val_loss': val_loss
        #})

        #print('Average loss for epoch : {i}'.format(i=total_loss/len(train_loader)))
    
    return model  







if __name__ == "_main_":
    train()