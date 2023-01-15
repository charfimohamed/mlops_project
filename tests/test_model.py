from pathlib import Path

import pytest
import torch

from src.models.model import CatDogModel

project_dir = Path(__file__).resolve().parents[1]


def test_model_shape():
    model = CatDogModel()
    sample_input = torch.randn((32, 3, model.im_size, model.im_size))
    # Pass the input tensor through the model
    output = model(sample_input)
    # Assert that the output shape is (32,2)
    assert output.shape == (32, 2), "the model does not output the right shape"


def test_forward_with_invalid_input():
    model = CatDogModel()
    x = torch.randn(32, 16, 16)
    # 3D tensor
    with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
        model.forward(x)
