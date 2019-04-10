from torchtest import assert_vars_change
from torchtest import test_suite
import sys
import os
sys.path.append(os.path.abspath('../'))

import pytest
from src.dataloader import create_dataloaders
import torch.nn as nn
from src.models.Custom_CNN import Simple_CNN
import torch

use_cuda = torch.cuda.is_available()
img_size = 48
device = torch.device("cuda" if use_cuda else "cpu")

train_loader, test_loader, val_loader = create_dataloaders(
    batchsize=128, img_size=img_size, path="../../data/Classification"
)

for batch_idx, (data, target) in enumerate(train_loader):
    batch = [data, target]
    break

model = Simple_CNN().to(device)
loss = nn.CrossEntropyLoss()


def test_vars_change():
    assert_vars_change(
        model=model,
        loss_fn=loss,
        optim=torch.optim.Adam(model.parameters()),
        batch=batch,
    )


def test_nan_vals():
    test_suite(
        model=model,
        loss_fn=loss,
        optim=torch.optim.Adam(model.parameters()),
        batch=batch,
        test_nan_vals=True,
    )


def test_inf_vals():
    test_suite(
        model=model,
        loss_fn=loss,
        optim=torch.optim.Adam(model.parameters()),
        batch=batch,
        test_inf_vals=True,
    )


if __name__ == "__main__":
    test_vars_change()
    test_inf_vals()
    test_nan_vals()
