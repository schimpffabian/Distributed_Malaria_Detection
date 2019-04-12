from torchtest import assert_vars_change
from torchtest import test_suite

import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import torch.nn as nn
from src.models.Custom_CNN import Simple_CNN
import torch
import torch.utils.data as utils

use_cuda = torch.cuda.is_available()
img_size = 48
device = torch.device("cuda" if use_cuda else "cpu")

# Dummy Dataloader
dataset = utils.TensorDataset(torch.rand(100, 3, img_size, img_size), torch.rand(100).long())  # create your datset
dataloader = utils.DataLoader(dataset, batch_size=42, shuffle=True)

for batch_idx, (data, target) in enumerate(dataloader):
    batch = [data, target]
    break

model = Simple_CNN().float().to(device)
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
