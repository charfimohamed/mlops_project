import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.quantization
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.make_dataset import CatDogDataset
from src.models.model import CatDogModel


def test(batch_size: int = 32):
    """
    tests the neural network after training

     Parameters:
                    batch_size (int): the size of the batch

    """
    # checking whether a CUDA-enabled GPU is available, and if so, it sets the DEVICE to "cuda" otherwise, the DEVICE is set to "cpu".
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # creates an instance of the CatDogModel
    model = CatDogModel()
    # loading the best parameters for the model
    checkpoint = torch.load("checkpoints/model_best_checkpoint.pth")
    model.load_state_dict(checkpoint["model"])

    # apply dynamic quantization
    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    torch.quantization.prepare(model, inplace=True)
    model = torch.quantization.convert(model)

    model.eval()
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
    test_dataset = CatDogDataset(
        split="test", in_folder=Path("../../data/raw"), out_folder=Path("../../data/processed"), transform=data_resize
    )
    # loading the dataset
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    test_accuracy = 0

    start_time = time.time()

    for i, (images, labels) in enumerate(test_dataloader):
        print(f"test step {i}")
        outputs = model(images)
        _, preds = torch.max(outputs, dim=1)
        test_accuracy += torch.sum(preds == labels)
    test_accuracy = test_accuracy / len(test_dataset)

    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time)

    print(f"test accuracy : {test_accuracy}")
    model.eval()
    samples, labels = next(iter(test_dataloader))
    outputs = model(samples)
    _, preds = torch.max(outputs, dim=1)
    class_labels = {0: "cat", 1: "dog"}
    # create a figure with a grid of subplots
    fig, axs = plt.subplots(4, 8, figsize=(20, 10))
    fig.set_size_inches(15, 8)
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
        axs[i].axis("off")
    # show the plot
    plt.show()


if __name__ == "__main__":
    test()
