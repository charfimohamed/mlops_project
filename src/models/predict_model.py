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
def test (batch_size = 32, epochs = 10):
    ''' tests the neural network after training'''

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CatDogModel(num_classes =2)
    model.to(DEVICE)
    dataset = make_dataset(split = "train", )
    

    image_size = 224
    data_resize = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()])

    test_dataset = CatDogDataset(split="test", in_folder=Path("../data/raw"), out_folder=Path('../data/processed'), transform=data_resize)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=True)


    for epoch in range(epochs):
        test_acc = model.test_step(batch = test_dataloader , batch_idx= epoch)
        print("accuracy = " + test_acc) 
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
    test()