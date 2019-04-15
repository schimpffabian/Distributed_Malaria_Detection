import syft as sy  # <-- NEW: import the Pysyft library
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils as utils
import numpy as np
from src.dataloader import create_dataset, split_dataset, get_data_augmentation
from src.models.Custom_CNN import Simple_CNN
from src.auxiliaries import run_t, train
from torch.utils.data import Dataset


def create_federated_dataset(
    path="../data/Classification",
    img_size=48,
    percentage_of_dataset=np.array([0.8, 0.2]),
):

    data_augmentation = get_data_augmentation(False, img_size)
    dataset = create_dataset(path=path, data_augmentation=data_augmentation)
    split_datasets = split_dataset(dataset, percentage_of_dataset)
    return split_datasets


def simple_federated_model():
    epochs = 2

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(42)
    hook = sy.TorchHook(
        torch
    )  # <-- NEW: hook PyTorch ie add extra functionalities to support Federated Learning
    katherienhospital = sy.VirtualWorker(
        hook, id="kh"
    )  # <-- NEW: define remote worker bob
    filderklinik = sy.VirtualWorker(hook, id="fikli")  # <-- NEW: and alice

    train_set, test_set = create_federated_dataset()

    train_dataset = DatasetFromSubset(train_set)
    train_set_federated = train_dataset.federate((katherienhospital, filderklinik))

    federated_train_loader = sy.FederatedDataLoader(
        federated_dataset=train_set_federated, batch_size=42
    )
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=42, shuffle=True)

    model = Simple_CNN()
    model.load_state_dict(torch.load("./models/custom_cnn_e10_size_48.pt"))
    model = model.float()

    device = torch.device("cuda" if use_cuda else "cpu")
    #optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    loss = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        run_t(model, device, test_loader, loss)
        train(
            model,
            device,
            federated_train_loader,
            optimizer,
            epoch,
            loss,
            federated=True,
        )
        run_t(model, device, test_loader, loss)


def test_input_federated_dataloader():
    from torchvision import datasets, transforms

    hook = sy.TorchHook(torch)  #
    bob = sy.VirtualWorker(hook, id="bob")  # <-- NEW: define remote worker bob
    alice = sy.VirtualWorker(hook, id="alice")  # <-- NEW: and alice

    test = datasets.MNIST(
        "../data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )

    federated_train_loader = sy.FederatedDataLoader(  # <-- this is now a FederatedDataLoader
        datasets.MNIST(
            "../data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ).federate((bob, alice)),
        # <-- NEW: we distribute the dataset across all the workers, it's now a FederatedDataset
        batch_size=42,
        shuffle=True,
    )

    pass


class DatasetFromSubset(Dataset):
    def __init__(self, subset):
        data, targets = self.subset_to_dataset(subset)
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.data[index, :], self.targets[index]

    @staticmethod
    def subset_to_dataset(subset):
        indices = subset.indices
        targets = subset.dataset.targets

        targets_subset = torch.tensor(
            [targets[ii] for ii in list(indices.data.numpy())]
        )

        # Empty definition
        concat_tensor = None

        dataloader = torch.utils.data.DataLoader(subset, batch_size=2000, shuffle=False)

        for ii, (data, target) in enumerate(dataloader):
            print(ii)

            if ii == 0:
                concat_tensor = data
            else:
                concat_tensor = torch.cat((concat_tensor, data), 0)

        return concat_tensor, targets_subset


if __name__ == "__main__":
    # test_input_federated_dataloader()
    simple_federated_model()

    """
    train_set_reformatted = train_set.dataset

    train_set_reformatted.imgs = [train_set_reformatted.imgs[ii] for ii in list(train_set.indices.data.numpy())]
    train_set_reformatted.data =  train_set_reformatted.imgs
    train_set.data = None
    train_set.targets = None
    train_set_reformatted.targets = [train_set_reformatted.targets[ii] for ii in list(train_set.indices.data.numpy())]
    print()
    """
