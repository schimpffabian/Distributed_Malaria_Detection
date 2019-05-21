import torch.nn as nn
from src.models.Custom_CNN import Simple_CNN_e2
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
    """

    :return:
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # device = torch.device("cpu")
    return Simple_CNN_e2(128).float().to(device)


@pytest.fixture()
def loss():
    """

    :return:
    """
    return nn.CrossEntropyLoss()


@pytest.fixture()
def loss_fn():
    """

    :return:
    """
    return nn.CrossEntropyLoss()


@pytest.fixture()
def batch():
    """

    :return:
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    img_size = 128

    # Dummy Dataloader
    dataset = utils.TensorDataset(
        torch.rand(100, 3, img_size, img_size), torch.rand(100).long()
    )  # create your datset
    dataloader = utils.DataLoader(dataset, batch_size=42, shuffle=True)

    for batch_idx, (data, target) in enumerate(dataloader):
        batch = [data.to(device), target.to(device)]
        break
    return batch


@pytest.fixture()
def optim():
    """

    :return:
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # device = torch.device("cpu")
    return torch.optim.Adam(Simple_CNN_e2(128).float().to(device).parameters())


def test_vars_change(model, loss, batch):
    """

    :param model:
    :param loss:
    :param batch:
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # device = torch.device("cpu")
    assert_vars_change(model, loss, torch.optim.Adam(model.parameters()), batch, device=device)


def test_nan_vals(model, loss_fn, batch, optim):
    """

    :param model:
    :param loss_fn:
    :param batch:
    :param optim:
    """
    test_suite(model, loss_fn, optim, batch, test_nan_vals=True)


def test_inf_vals(model, loss_fn, batch, optim):
    """

    :param model:
    :param loss_fn:
    :param batch:
    :param optim:
    """
    test_suite(model, loss_fn, optim, batch, test_inf_vals=True)
