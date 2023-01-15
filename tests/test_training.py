import os
from pathlib import Path

import pytest
import torch

from src.models.model import CatDogModel

project_dir = Path(__file__).resolve().parents[1]
checkpoint_path = os.path.join(project_dir, 'src/models/checkpoints/model_best_checkpoint.pth')


@pytest.mark.skipif(not os.path.exists(checkpoint_path), reason="model weights are not found")
def test_saved_model_works():
    model = CatDogModel()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    sample_input = torch.randn((32, 3, model.im_size, model.im_size))
    # Pass the input tensor through the model
    output = model(sample_input)
    # Assert that the output shape is (32,2)
    assert output.shape == (32, 2), "the saved model does not output the right shape"


def test_training_accuracy():
    state = torch.load(checkpoint_path)
    best_accuracy = state['best accuracy']
    assert best_accuracy > 0.5, f"best_accuracy should be larger than 0.5 (more than chance) but it's {best_accuracy}"
