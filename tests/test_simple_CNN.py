import torch.nn as nn
from src.models.Custom_CNN import Simple_CNN
import torch
import torch.utils.data as utils
import pytest
import sys
import os

from torchtest import assert_vars_change
from torchtest import test_suite

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../")


@pytest.fixture()
def model():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return Simple_CNN().float().to(device)


@pytest.fixture()
def loss():
    return nn.CrossEntropyLoss()


@pytest.fixture()
def loss_fn():
    return nn.CrossEntropyLoss()


@pytest.fixture()
def batch():
    img_size = 48

    # Dummy Dataloader
    dataset = utils.TensorDataset(
        torch.rand(100, 3, img_size, img_size), torch.rand(100).long()
    )  # create your datset
    dataloader = utils.DataLoader(dataset, batch_size=42, shuffle=True)

    for batch_idx, (data, target) in enumerate(dataloader):
        batch = [data, target]
        break
    return batch


@pytest.fixture()
def optim():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return torch.optim.Adam(Simple_CNN().float().to(device).parameters())


def test_vars_change(model, loss, batch):

    assert_vars_change(model, loss, torch.optim.Adam(model.parameters()), batch)


def test_nan_vals(model, loss_fn, batch, optim):

    test_suite(model, loss_fn, optim, batch, test_nan_vals=True)


def test_inf_vals(model, loss_fn, batch, optim):
    test_suite(model, loss_fn, optim, batch, test_inf_vals=True)


if __name__ == "__main__":
    test_vars_change()
    test_inf_vals()
    test_nan_vals()
