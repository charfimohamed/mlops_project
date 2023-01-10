import logging
import os
import sys

#import hydra
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from src.data import make_dataset
from src.data.make_dataset import CatDogDataset
import wandb
from pathlib import Path
from torchvision import transforms
from src.models.train_model import *


#sys.path.append("..")
#log = logging.getLogger(__name__)
#print = log.info


#training_function
def test (batch_size = 32):
    ''' tests the neural network after training'''
    print("_____")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train()
    model.to(DEVICE)
    
    image_size = 128
    data_resize = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()])

    test_dataset = CatDogDataset(split="test", in_folder=Path("../data/raw"), out_folder=Path('../data/processed'), transform=data_resize)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=True)
    correct=0
    for i, (images, labels) in enumerate(test_dataloader):
        output = model(images)
        preds = torch.argmax(output,dim=1)
        correct = correct + (preds==labels).sum()

    test_accuracy = correct/(batch_size*len(labels))
    print(int(correct))
    print("accuracy = {i}".format(i=test_accuracy))

        #wandb.log({
            #'epoch': epoch, 
            #'train_acc': train_acc,
            #'train_loss': train_loss, 
            #'val_acc': val_acc, 
            #'val_loss': val_loss
        #})

        #print('Average loss for epoch : {i}'.format(i=total_loss/len(train_loader)))
    
    return model  

if __name__ == "__main__":
    test()