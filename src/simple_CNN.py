from __future__ import print_function
from src.models.Custom_CNN import Simple_CNN
import torch
import torch.optim as optim
from src.auxiliaries import train, run_t, initialize_model
from src.dataloader import create_dataloaders
import torch.nn as nn

lr = 1e-3


def finetune_model():
    model_name = "squeezenet" # squeezenet, "resnet"
    num_classes = 2
    batch_size = 64
    num_epochs = 10
    feature_extract = True

    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    loss = nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, test_loader, val_loader = create_dataloaders(batchsize=batch_size, img_size=input_size)

    model = model_ft.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch, loss)
        run_t(model, device, test_loader, loss)

    torch.save(model.state_dict(), "./models/" + model_name + "_e" + str(num_epochs) + ".pt")


def main():
    epochs = 15
    img_size = 48

    # Training settings
    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, test_loader, val_loader = create_dataloaders(batchsize=128, img_size=img_size)

    model = Simple_CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        run_t(model, device, test_loader)

    torch.save(model.state_dict(), "./models/custom_cnn_e" + str(epochs) + "_size_" + str(img_size) + ".pt")


if __name__ == '__main__':
    finetune_model()
