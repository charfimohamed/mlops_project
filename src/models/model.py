import torchvision
from torch import nn
import torch.quantization

class CatDogModel(nn.Module):
    """defines a neural network model"""

    def __init__(self):
        super().__init__()

        self.model = torchvision.models.resnet50(pretrained=True)
        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_ftrs, 2)
        self.im_size = 16

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("Expected input to a 4D tensor")
        x = self.model(x)
        return x


if __name__ == "__main__":
    # check that all model parameters are of type "torch.float32"
    model = CatDogModel()
    for n, p in model.named_parameters():
        print(n, ": ", p.dtype)