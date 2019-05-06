import torch
import torch.optim as optim
import torch.nn as nn
import os

try:
    from src.auxiliaries import train
    from src.auxiliaries import run_t
    from src.auxiliaries import initialize_model
    from src.dataloader import create_dataloaders
    from src.models.Custom_CNN import Simple_CNN
except ModuleNotFoundError:
    from auxiliaries import train
    from auxiliaries import run_t
    from auxiliaries import initialize_model
    from dataloader import create_dataloaders
    from models.Custom_CNN import Simple_CNN


def finetune_model(
    model_name="squeezenet",
    num_classes=2,
    batch_size=64,
    num_epochs=42,
    feature_extract=True,
    lr=1e-3,
):
    """

    :param model_name:
    :param num_classes:
    :param batch_size:
    :param num_epochs:
    :param feature_extract:
    :param lr:
    """
    model_ft, input_size = initialize_model(
        model_name, num_classes, feature_extract, use_pretrained=True
    )
    loss = nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, test_loader, val_loader = create_dataloaders(
        batchsize=batch_size, img_size=input_size
    )

    model = model_ft.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch, loss)
        run_t(model, device, test_loader, loss)

    torch.save(
        model.state_dict(), "./models/" + model_name + "_e" + str(num_epochs) + ".pt"
    )


def custom_classifier(model=Simple_CNN(), batch_size=64, num_epochs=42, img_size=48, lr=1e-3, use_gpu=True):
    """

    :param model:
    :param batch_size:  (int) batch size for training
    :param num_epochs: (int) number of training epochs
    :param img_size: size of input image
    :param lr: learning rate typically ~1e-3 - 1e-5
    :param use_gpu: (bool) use a GPU if available
    """
    # Training settings
    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda and use_gpu else "cpu")

    train_loader, test_loader, val_loader = create_dataloaders(
        batchsize=batch_size, img_size=img_size
    )

    model = model.to(device)

    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch, loss)
        run_t(model, device, test_loader)

    if os.name != 'nt':
        torch.save(
            model.state_dict(),
            "./models/custom_cnn_e" + str(num_epochs) + "_size_" + str(img_size) + ".pt",
        )
    else:
        torch.save(
            model.state_dict(),
            ".\\models\\custom_cnn_e" + str(num_epochs) + "_size_" + str(img_size) + ".pt",
        )


if __name__ == "__main__":
    train_custom_classifier()
    # finetune_model()
