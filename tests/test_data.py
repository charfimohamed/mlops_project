import os.path
from pathlib import Path

import pytest
from torchvision import transforms

from src.data.make_dataset import CatDogDataset
from src.models.model import CatDogModel

project_dir = Path(__file__).resolve().parents[1]
model = CatDogModel()


@pytest.mark.skipif(not os.path.exists(project_dir), reason="Data files not found")
def test_trainset_size():
    train_dataset = CatDogDataset(split="train", in_folder=project_dir / "data" / "raw", out_folder=project_dir / "data" / "processed")
    assert len(train_dataset) == 2000, "Training dataset did not have the correct number of samples"


@pytest.mark.skipif(not os.path.exists(project_dir), reason="Data files not found")
def test_validationset_size():
    validation_dataset = CatDogDataset(split="validation", in_folder=project_dir / "data" / "raw", out_folder=project_dir / "data" / "processed")
    assert len(validation_dataset) == 401, "Validation dataset did not have the correct number of samples"


@pytest.mark.skipif(not os.path.exists(project_dir), reason="Data files not found")
def test_testset_size():
    test_dataset = CatDogDataset(split="test", in_folder=project_dir / "data" / "raw", out_folder=project_dir / "data" / "processed")
    assert len(test_dataset) == 401, "Testing dataset did not have the correct number of samples"


@pytest.mark.skipif(not os.path.exists(project_dir), reason="Data files not found")
def test_testset_label_presence():
    labels = []
    test_dataset = CatDogDataset(split="test", in_folder=project_dir / "data" / "raw", out_folder=project_dir / "data" / "processed")

    for img, label in test_dataset:
        labels.append(label)
    assert set(labels) == set([0, 1]), "Testing dataset must include both cats and dogs"


@pytest.mark.skipif(not os.path.exists(project_dir), reason="Data files not found")
def test_validationset_label_presence():
    labels = []
    validation_dataset = CatDogDataset(split="validation", in_folder=project_dir / "data" / "raw", out_folder=project_dir / "data" / "processed")
    for img, label in validation_dataset:
        labels.append(label)
    assert set(labels) == set([0, 1]), "Validation dataset must include both cats and dogs"


@pytest.mark.skipif(not os.path.exists(project_dir), reason="Data files not found")
def test_trainset_label_presence():
    train_dataset = CatDogDataset(split="train", in_folder=project_dir / "data" / "raw", out_folder=project_dir / "data" / "processed")

    labels = []
    for img, label in train_dataset:
        labels.append(label)
    assert set(labels) == set([0, 1]), "Training dataset must include both cats and dogs"


@pytest.mark.skipif(not os.path.exists(project_dir), reason="Data files not found")
@pytest.mark.parametrize("image_size, expected_shape", [(224, (3, 224, 224)), (model.im_size, (3, model.im_size, model.im_size))])
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
