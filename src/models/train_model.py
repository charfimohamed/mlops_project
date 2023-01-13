import logging
import os
import sys
import timm
import hydra
import wandb
import pprint
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from src.models.model import CatDogModel
from src.data import make_dataset
from src.data.make_dataset import CatDogDataset
from pathlib import Path
from torchvision import transforms
from pytorch_lightning import Trainer
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
#import re
#import pstats
#from pstats import SortKey
import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity, record_function
import tensorboard
from torch.profiler import profile, tensorboard_trace_handler

#import cProfile
#cProfile.run('re.compile("foo|bar")')
#cProfile.run('re.compile("foo|bar")', 'restats')
#p = pstats.Stats('restats')
#p.strip_dirs().sort_stats(-1).print_stats()
#p.sort_stats(SortKey.NAME)
#p.print_stats()

#check which version of pytorch i have
print(torch.__version__)

#sys.path.append("..")
#save all printed output from the script
logs = logging.getLogger(__name__)
#print = log.info


@hydra.main(config_name="config.yaml")
def main(cfg: DictConfig):

#extracting the hyperparameters from the yaml file
    batch_size_values = cfg.hyperparameters.batch_size
    lr_values = cfg.hyperparameters.learning_rate
    optimizer_values = cfg.hyperparameters.optimizer
    
# Directing the hyperparameters to wandb
    sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'validation_acc'},
    'parameters': {
        'batch_size': {'values': list(batch_size_values['values'])},
        'lr': {'values': list(lr_values['values'])},
        'optimizer': {'values': list(optimizer_values['values'])}

     }
    }

    pprint.pprint(sweep_configuration)

    # Create a sweep
    sweep_id = wandb.sweep(sweep_configuration, project="group18_mlops")
   
    
    #train()  # training function call
        
    # Run the sweep
    wandb.agent(sweep_id, function=train_hp, count=4)

    wandb.finish()

def save_checkpoint(model,epoch,best_accuracy):
    logs.info("------> saving checkpoint <------")
    state = {
    'epoch': epoch + 1,
    'model' : model.state_dict(),
    'best accuracy': best_accuracy,
    }
    torch.save (state, 'model_best_checkpoint.pth')

def train_hp():
    logs.info("training hyperparameters")
    wandb.init(project="test-project", entity="group18_mlops")
    train(batch_size=wandb.config.batch_size, epochs=5, lr=wandb.config.lr)

#training_function
def train (batch_size =32, epochs =10, lr= 0.001):
    logs.info("Trains a neural network from the TIMM framework")
    #DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = CatDogModel()
    #model.to(DEVICE)
    model = CatDogModel()
    image_size = model.im_size
    data_resize = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    train_dataset = CatDogDataset(split="train", in_folder=Path("../data/raw"), out_folder=Path('../data/processed'), transform=data_resize)
    validation_dataset = CatDogDataset(split="validation", in_folder=Path("../data/raw"), out_folder=Path('../data/processed'), transform=data_resize)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size = batch_size, shuffle=True)
    optimizer = Adam(model.parameters(),lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    best_accuracy = 0.0 # accuracy of the best epoch to know what wights to save 
    for epoch in range(epochs):
        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA], record_shapes=True, schedule=torch.profiler.schedule(wait=0,warmup=0,active=10)) as prof: #,on_trace_ready=tensorboard_trace_handler("src/models/trace_prof")
            with record_function("model"):
                print(f"epoch : {epoch+1}/{epochs}")
                model.train()
                prof.step()
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
                'epoch': epoch, 
                'train_acc': train_accuracy,
                'validation_acc': validation_accuracy,
                'train_loss': train_loss,}) 

        # profiling with respect to the total CPU time      
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
        # to see if there is any correlation between the shape of the input and the cost of the operation
        #print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))
        # profiling with respect to the memory usage
        print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=15))
        #print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))

        prof.export_chrome_trace("src/models/trace.json")
    return model  
           

if __name__ == "__main__":
    #logs.info("main")

    #import cProfile
    #cProfile.run('main()','output_main.dat')

    #import pstats
    #from pstats import SortKey

    #p = pstats.Stats('output_main.dat')
    #p.sort_stats("time").print_stats()
    #p.sort_stats("calls").print_stats()


    main()

