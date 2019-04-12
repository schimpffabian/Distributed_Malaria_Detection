import sys, os

sys.path.append(os.path.join(".."))  # add the current directory
import src.dataloader as dataloader


def test_create_dataloaders():
    """
    The original dataset has evenly distributed classes - this should be represented in the training, test and
    validation data
    :return:
    """
    train_loader, test_loader, val_loader = dataloader.create_dataloaders()
