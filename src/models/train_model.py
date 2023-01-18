import logging
import os
from pathlib import Path

import torch
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.make_dataset import CatDogDataset
from src.models.model import CatDogModel

log = logging.getLogger(__name__)


def save_checkpoint(model: CatDogModel, best_accuracy: float):
    """saving the best model

    Parameters:
                   model (CatDogModel): the used model
                   best_accuracy (float): the best accuracy to be saved

    """
    print("------> saving checkpoint <------")
    state = {
        "model": model.state_dict(),
        "best accuracy": best_accuracy,
    }
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    torch.save(state, "checkpoints/model_best_checkpoint.pth")


def train(batch_size: int = 64, epochs: int = 5, lr: float = 0.0005, optimizer_name: str = "adam") -> CatDogModel:
    """
    Train the dataset

     Parameters:
                    batch_size (int): the size of the batch
                    epochs (int): the number of epochs
                    lr (float): the learning rate
                    optimizer_name (str): the name of the optimizer

            Returns:
                    model (CatDogModel): the training model
    """
    # checking whether a CUDA-enabled GPU is available, and if so, it sets the DEVICE to "cuda" otherwise, the DEVICE is set to "cpu".
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # creates an instance of the CatDogModel
    model = CatDogModel()
    # transfers the model from CPU to the device which is either GPU or CPU that was defined above
    model.to(DEVICE)
    image_size = model.im_size
    # add transformations to images and converting images into tensors
    data_resize = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    # processing the dataset
    train_dataset = CatDogDataset(
        split="train", in_folder=Path("../../data/raw"), out_folder=Path("../../data/processed"), transform=data_resize
    )
    validation_dataset = CatDogDataset(
        split="validation",
        in_folder=Path("../../data/raw"),
        out_folder=Path("../../data/processed"),
        transform=data_resize,
    )
    # loading the dataset
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    if optimizer_name == "sgd":
        optimizer = SGD(model.parameters(), lr=lr)
    if optimizer_name == "adam":
        optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    best_accuracy = 0.0  # accuracy of the best epoch to know what wights to save
    for epoch in range(epochs):
        print(f"epoch : {epoch+1}/{epochs}")
        model.train()
        train_accuracy = 0.0
        train_loss = 0.0
        validation_accuracy = 0.0
        for i, (images, labels) in enumerate(train_dataloader):
            print(f" train step {i}")
            outputs = model(images)
            optimizer.zero_grad()
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, dim=1)
            train_loss += loss
            train_accuracy += torch.sum(preds == labels)
        train_loss = train_loss / len(train_dataset)
        train_accuracy = train_accuracy / len(train_dataset)
        print(f"Epoch {epoch+1}/{epochs}. Loss: {train_loss} . accuracy : {train_accuracy}")
        model.eval()
        for i, (images, labels) in enumerate(validation_dataloader):
            print(f"validation step is {i}")
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)
            validation_accuracy += torch.sum(preds == labels)
        validation_accuracy = validation_accuracy / len(validation_dataset)
        print(f"validation accuracy is : {validation_accuracy}")
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            save_checkpoint(model, best_accuracy)
    return model


if __name__ == "__main__":
    train()
