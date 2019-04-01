import src.dataloader as dataloader

train_loader, test_loader, val_loader = dataloader.create_dataloaders()


def test_create_dataloaders():
    """
    The original dataset has evenly distributed classes - this should be represented in the training, test and
    validation data
    :return:
    """

    # ToDo: Write test
