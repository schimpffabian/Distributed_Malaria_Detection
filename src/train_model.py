import torch
import torch.optim as optim
import torch.nn as nn
from pathlib import Path

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
    use_gpu=True,
    name="",
):
    """

    :param str model_name: model to be loaded
    :param int num_classes: number of classes
    :param int batch_size: size of batch used for SGD
    :param int num_epochs: Number of times the network is updated on the data
    :param bool feature_extract: deactivate gradients
    :param float lr: learning rate
    :param bool use_gpu: Allow or prohibit use of GPU
    :param str name: Optional path to save the network parameters to
    """

    model_ft, input_size = initialize_model(
        model_name, num_classes, feature_extract, use_pretrained=True
    )
    loss = nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()

    if use_gpu:
        device = torch.device("cuda" if use_cuda else "cpu")
    else:
        device = "cpu"

    train_loader, test_loader = create_dataloaders(
        batchsize=batch_size, img_size=input_size
    )

    model = model_ft.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch, loss)
        accuracy = run_t(model, device, test_loader, loss)

    if name == "":
        save_name = Path("./models/" + model_name + "_e" + str(num_epochs) + ".pt")
    else:
        save_name = Path(name + ".pt")

    torch.save(model.state_dict(), save_name)
    return accuracy


def custom_classifier(
    model=Simple_CNN(img_size=128),
    batch_size=256,
    num_epochs=42,
    img_size=48,
    lr=1e-3,
    use_gpu=True,
    random_background=False,
    name="",
):
    """

    :param model:
    :param batch_size:  (int) batch size for training
    :param num_epochs: (int) number of training epochs
    :param img_size: size of input image
    :param lr: learning rate typically ~1e-3 - 1e-5
    :param use_gpu: (bool) use a GPU if available
    :param random_background: (bool) randomize uniform background
    :param name: (str) optionally name the saved state dicts
    """
    # Training settings
    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda and use_gpu else "cpu")
    train_loader, test_loader = create_dataloaders(
        batchsize=batch_size, img_size=img_size, random_background=random_background
    )

    model = model.to(device)

    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch, loss)
        accuracy = run_t(model, device, test_loader, loss)

    if random_background:
        background_extension = "_rand_backgr"
    else:
        background_extension = ""

    if name == "":
        save_name = Path(
            "./models/custom_cnn_e"
            + str(num_epochs)
            + "_size_"
            + str(img_size)
            + background_extension
            + ".pt"
        )
    else:
        save_name = Path(name + ".pt")

    torch.save(model.state_dict(), save_name)

    return accuracy


if __name__ == "__main__":
    custom_classifier()
    # finetune_model()
