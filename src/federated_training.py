import torch
import torch.optim as optim
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import Dataset
from src.dataloader import create_dataset, split_dataset, get_data_augmentation, set_prop_dataset
from src.models.Custom_CNN import Simple_CNN_e2
from src.auxiliaries import run_t, train
import syft as sy
import timeit
from pathlib import Path


class DatasetFromSubset(Dataset):
    """
    Helper to convert subsets to datasets since PySyft only works with the ladder
    """

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
        """
        Method to turn the index tensor and original dataset in subsets into smaller datasets

        :param subset: Subset to transform
        :return: dataset
        """
        indices = subset.indices
        targets = subset.dataset.targets

        if isinstance(indices, list):
            pass
        elif isinstance(indices, torch.tensor):
            indices = list(indices.data.numpy())
        elif isinstance(indices, np.ndarray):
            indices = list(indices)

        else:
            print(type(indices))
            raise NotImplementedError

        targets_subset = torch.tensor(
            [targets[ii] for ii in indices]
        )

        # Empty definition
        concat_tensor = None

        dataloader = torch.utils.data.DataLoader(subset, batch_size=4000, shuffle=False)

        for ii, (data, target) in enumerate(dataloader):
            del target
            if ii == 0:
                concat_tensor = data
            else:
                concat_tensor = torch.cat((concat_tensor, data), 0)

        return concat_tensor, targets_subset


def create_federated_dataset(
    path=os.path.dirname(os.path.abspath(__file__)) + "/../data/Classification",
    img_size=128,
    percentage_of_dataset=np.array([0.75, 0.25]),
    balance=np.array([[0.5, 0.5], [0.5, 0.5]])
):
    """
    Helper function to create datasets that can be used for federated learning

    :param path: (str) path to folders containing images
    :param int img_size: size of image - underlying assumption of square images
    :param list percentage_of_dataset: list or numpy array with percentage of each split
    :param list balance: list of lists containing the wanted probabilities of each class in the given datasets
    :return: split datasets - subsets
    """

    data_augmentation = get_data_augmentation(random_background=False, img_size=img_size)
    dataset = create_dataset(path=path, data_augmentation=data_augmentation)

    targets = dataset.targets
    split_datasets = split_dataset(dataset, percentage_of_dataset)
    split_datasets = set_prop_dataset(split_datasets, targets, balance)
    return split_datasets


def imbalanced_distribution():
    """
    train a model with data federated over multiple workers
    :return:
    """
    # Parameters and general setup
    epochs = 1

    # use_cuda = torch.cuda.is_available()
    torch.manual_seed(42)
    hook = sy.TorchHook(torch)
    results = []

    # Setup virtual workers
    katherienhospital = sy.VirtualWorker(hook, id="kh")
    filderklinik = sy.VirtualWorker(hook, id="fikli")

    for identifier, distribution in enumerate([[0.9, 0.1], [0.7, 0.3], [0.5, 0.5]]):
        reverse_distribution = distribution[::-1]
        balance = [distribution, reverse_distribution, [0.5, 0.5]]
        percentage = [0.4, 0.4, 0.2]

        # Create datasets and dataloaders
        train_worker_kh, train_worker_fikli, test_set = create_federated_dataset(balance=balance, percentage_of_dataset=percentage)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1024, shuffle=True)

        # Create datasets
        train_dataset_kh = DatasetFromSubset(train_worker_kh)
        train_dataset_fikli = DatasetFromSubset(train_worker_fikli)

        # Seperate data and labels
        data_kh = train_dataset_kh.data
        target_kh = train_dataset_kh.targets

        data_fikli = train_dataset_fikli.data
        target_fikli = train_dataset_fikli.targets

        # Create pointers
        data_kh = data_kh.send(katherienhospital)
        target_kh = target_kh.send(katherienhospital)

        data_fikli = data_fikli.send(filderklinik)
        target_fikli = target_fikli.send(filderklinik)

        # Organize pointers to form a pseudo dataloader
        train_loader = [(data_kh, target_kh), (data_fikli, target_fikli)]

        for ii in range(3):
            # load model
            model = Simple_CNN_e2(img_size=128)
            model = model.float()

            # Setup optimization
            device = "cpu"  # device = torch.device("cuda" if use_cuda else "cpu") GPU not fully supported
            optimizer = optim.SGD(model.parameters(), lr=1e-1)  # optimizer = optim.Adam(model.parameters(), lr=1e-3)
            loss = nn.CrossEntropyLoss()

            for epoch in range(1, epochs + 1):
                train(
                    model,
                    device,
                    train_loader,
                    optimizer,
                    epoch,
                    loss,
                    federated=True,
                )
                accuracy = run_t(model, device, test_loader, loss)

                results.append([ii, identifier, epoch, accuracy])

            np.savetxt(Path("./logs/federated_learning_speed_distribution.csv"), results, delimiter=",",
                       header="run, identifier, epoch, accuracy")


if __name__ == "__main__":
    imbalanced_distribution()
