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
from src.models.optimize_train_model import *
import torchvision
import matplotlib


#sys.path.append("..")
#log = logging.getLogger(__name__)
#print = log.info



#training_function
def test (batch_size:int = 32):
    class_labels = {0: "cat", 1: "dog"}
    ''' tests the neural network after training'''
    #print("_____")
    #DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load('model_best_checkpoint.pth')
    model=CatDogModel()
    checkpoint = torch.load('model_best_checkpoint.pth')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    #model.to(DEVICE)
    
    image_size = model.im_size
    data_resize = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    test_dataset = CatDogDataset(split="test", in_folder=Path("../data/raw"), out_folder=Path('../data/processed'), transform=data_resize)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=True)
    nb_test_samples=0
    test_accuracy = 0
    for i,(images, labels) in enumerate(test_dataloader) :
        print(f"test step {i}")
        outputs = model(images)
        _, preds = torch.max(outputs, dim=1)
        test_accuracy+=torch.sum(preds==labels)
        nb_test_samples += preds.shape[0]
    test_accuracy = test_accuracy /nb_test_samples
    print(f"test accuracy : {test_accuracy}")



    model.eval()

    samples, labels = next(iter(test_dataloader))
    outputs = model(samples)
    _, preds = torch.max(outputs, dim=1)

    # create a figure with a grid of subplots
    fig, axs = plt.subplots(4, 8, figsize=(20, 10))
    fig.set_size_inches(15,8)
    axs = axs.ravel()

    # loop through each image in the tensor
    for i in range(samples.shape[0]):
        # extract the current image
        img = samples[i]
        img = np.transpose(img, (1, 2, 0))
        # plot the image in the current subplot
        axs[i].imshow(img)
        # add a title to the image
        axs[i].set_title(f"predicted as {class_labels[int(preds[i])]}")
        # turn off axis labels
        axs[i].axis('off')

    # show the plot
    plt.show()

    #print(f"predictions are : _______ {preds}")
    #plt.figure(figsize=(16,32))
    #grid_imgs = torchvision.utils.make_grid(samples[:32])
    #np_grid_imgs = grid_imgs.numpy()
    #plt.title("24 samples of train data rezised to 224x224 pixels")
    #plt.imshow(np.transpose(np_grid_imgs, (1,2,0)))
    #plt.show()
    

    return model  

if __name__ == "__main__":
    test()