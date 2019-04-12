import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import src.dataloader as dataloader


def test_create_dataloaders():
    """
    The original dataset has evenly distributed classes - this should be represented in the training, test and
    validation data
    :return:
    """
    train_loader, test_loader, val_loader = dataloader.create_dataloaders()
