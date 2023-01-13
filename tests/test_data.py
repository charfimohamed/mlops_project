from pathlib import Path

import pytest
from torchvision import transforms

from src.data.make_dataset import CatDogDataset

project_dir = Path(__file__).resolve().parents[1]

train_dataset = CatDogDataset(
    split="train", in_folder=project_dir / "data" / "raw", out_folder=project_dir / "data" / "processed"
)

validation_dataset = CatDogDataset(
    split="validation", in_folder=project_dir / "data" / "raw", out_folder=project_dir / "data" / "processed"
)

test_dataset = CatDogDataset(
    split="test", in_folder=project_dir / "data" / "raw", out_folder=project_dir / "data" / "processed"
)


def test_trainset_size():
    assert len(train_dataset) == 2000, "Training dataset did not have the correct number of samples"


def test_validationset_size():
    assert len(validation_dataset) == 401, "Validation dataset did not have the correct number of samples"


def test_testset_size():
    assert len(test_dataset) == 401, "Testing dataset did not have the correct number of samples"


def test_testset_label_presence():
    labels = []
    for img, label in test_dataset:
        labels.append(label)
    assert set(labels) == set([0, 1]), "Testing dataset must include both cats and dogs"


def test_validationset_label_presence():
    labels = []
    for img, label in validation_dataset:
        labels.append(label)
    assert set(labels) == set([0, 1]), "Validation dataset must include both cats and dogs"


def test_trainset_label_presence():
    labels = []
    for img, label in train_dataset:
        labels.append(label)
    assert set(labels) == set([0, 1]), "Training dataset must include both cats and dogs"


@pytest.mark.parametrize("image_size, expected_shape", [(224, (3, 224, 224)), (100, (3, 100, 100))])
def test_resize_transform(image_size, expected_shape):
    data_resize = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])

    test_dataset = CatDogDataset(
        split="test",
        in_folder=project_dir / "data" / "raw",
        out_folder=project_dir / "data" / "processed",
        transform=data_resize,
    )

    for img, label in test_dataset:
        assert img.shape == expected_shape, "Images must be resized to the same shape"
