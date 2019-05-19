import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from src.dataloader import create_dataset, split_dataset, get_data_augmentation, set_prop_dataset, create_dataloaders
# from src.models.Custom_CNN import Simple_CNN_e1
from src.models.Custom_CNN import Simple_CNN_e2
from src.models.Custom_CNN import Simple_CNN2
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

        if type(indices) == list:
            pass
        elif type(indices) == torch.tensor:
            indices = list(indices.data.numpy())
        elif type(indices) == np.ndarray:
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
            if ii == 0:
                concat_tensor = data
            else:
                concat_tensor = torch.cat((concat_tensor, data), 0)

        return concat_tensor, targets_subset


def create_federated_dataset(
    path="../data/Classification",
    img_size=128,
    percentage_of_dataset=np.array([0.75, 0.25]),
    balance=np.array([[0.5, 0.5], [0.5, 0.5]])
):
    """
    :param path: (str) path to folders containing images
    :param int img_size: size of image - underlying assumption of square images
    :param list percentage_of_dataset: list or numpy array with percentage of each split
    :param list balance: list of lists containing the wanted probabilities of each class in the given datasets
    :return: split datasets - subsets
    """
    data_augmentation = get_data_augmentation(False, img_size)
    dataset = create_dataset(path=path, data_augmentation=data_augmentation)
    targets = dataset.targets
    split_datasets = split_dataset(dataset, percentage_of_dataset)
    split_datasets = set_prop_dataset(split_datasets, targets, balance)
    return split_datasets


def simple_federated_model():
    """
    train a model with data federated over multiple workers
    :return:
    """
    # Parameters and general setup
    epochs = 50

    # use_cuda = torch.cuda.is_available()
    torch.manual_seed(42)
    hook = sy.TorchHook(torch)
    results = []

    # Setup virtual workers
    katherienhospital = sy.VirtualWorker(hook, id="kh")
    filderklinik = sy.VirtualWorker(hook, id="fikli")
    kh_ruit = sy.VirtualWorker(hook, id="kh_ruit")
    marienhospital = sy.VirtualWorker(hook, id="marien")

    # Create datasets and dataloaders
    train_set, test_set = create_federated_dataset()
    train_dataset = DatasetFromSubset(train_set)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1024, shuffle=True)

    for ii in range(2):
        for num_workers in [2, 3, 4]:

            # load model
            model = Simple_CNN_e2(img_size=128)
            model = model.float()

            # Setup optimization
            device = "cpu"  # device = torch.device("cuda" if use_cuda else "cpu") GPU not fully supported
            optimizer = optim.SGD(model.parameters(), lr=1e-1)  # optimizer = optim.Adam(model.parameters(), lr=1e-3)
            loss = nn.CrossEntropyLoss()

            if num_workers == 1:
                # Doesn't work :(
                train_set_federated = train_dataset.federate((katherienhospital))
            elif num_workers == 2:
                train_set_federated = train_dataset.federate((katherienhospital, filderklinik))
            elif num_workers == 3:
                train_set_federated = train_dataset.federate((katherienhospital, filderklinik, kh_ruit))
            elif num_workers == 4:
                train_set_federated = train_dataset.federate((katherienhospital, filderklinik, kh_ruit, marienhospital))

            federated_train_loader = sy.FederatedDataLoader(
                federated_dataset=train_set_federated, batch_size=1024
            )

            start = timeit.default_timer()
            for epoch in range(1, epochs + 1):

                train(
                    model,
                    device,
                    federated_train_loader,
                    optimizer,
                    epoch,
                    loss,
                    federated=True,
                )
                accuracy = run_t(model, device, test_loader, loss)

            end = timeit.default_timer()
            results.append([ii, num_workers, end-start, accuracy])

            np.savetxt(Path("./logs/federated_learning_speed.csv"), results, delimiter=",",
                       header="run, num_workers,duration, accuracy")


def secure_evaluation(img_size=128):
    """
    https://blog.openmined.org/encrypted-deep-learning-classification-with-pysyft/
    """

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(42)
    hook = sy.TorchHook(torch)
    # client = sy.VirtualWorker(hook, id="client")
    katherienhospital = sy.VirtualWorker(
        hook, id="kh"
    )  # <-- NEW: define remote worker bob
    filderklinik = sy.VirtualWorker(hook, id="fikli")
    crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")

    train_set, test_set = create_federated_dataset(img_size=img_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=42, shuffle=True)

    model = Simple_CNN2(img_size)
    # model.load_state_dict(torch.load("./models/custom_cnn_e10_size_48.pt"))

    device = torch.device("cuda" if use_cuda else "cpu")
    # loss = F.nll_loss()
    loss = nn.NLLLoss()

    # Changes for secure evaluation
    private_test_loader = []
    for data, target in test_loader:
        pass
        private_test_loader.append(
            (
                data.fix_prec().share(
                    katherienhospital, filderklinik, crypto_provider=crypto_provider
                ),
                target.fix_prec().share(
                    katherienhospital, filderklinik, crypto_provider=crypto_provider
                ),
            )
        )

    model.fix_precision().share(
        katherienhospital, filderklinik, crypto_provider=crypto_provider
    )

    run_t(model, device, private_test_loader, loss, secure_evaluation=True)


def compare_optimizers():
    """
    train a model with data federated over multiple workers
    :return:
    """
    # Parameters and general setup
    epochs = 50
    torch.manual_seed(42)

    # Create datasets and dataloaders
    train_set, test_set = create_federated_dataset()
    train_dataset = DatasetFromSubset(train_set)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1024, shuffle=True)

    # Setup optimization
    device = "cpu"
    results_time = []
    results_acc = []

    for opt in range(3):
        for run in range(3):
            # load model
            model = Simple_CNN_e2(img_size=128)
            model = model.float()

            if opt == 0:
                optimizer = optim.Adam(model.parameters(), lr=1e-3)
            elif opt == 1:
                optimizer = optim.SGD(model.parameters(), lr=1e-3)
            elif opt == 2:
                optimizer = optim.SGD(model.parameters(), lr=1e-1)

            loss = nn.CrossEntropyLoss()

            start = timeit.default_timer()
            for epoch in range(1, epochs + 1):
                train(
                    model,
                    device,
                    train_loader,
                    optimizer,
                    epoch,
                    loss,
                    federated=False,
                )
                accuracy = run_t(model, device, test_loader, loss)
                results_acc.append([opt, run, epoch, accuracy])
                np.savetxt(Path("./logs/optimiser_acc_log.csv"), results_time, delimiter=",",
                           header="optimizer, run, epoch, accuracy")

            end = timeit.default_timer()
            results_time.append([opt, run , end - start, accuracy])
            np.savetxt(Path("./logs/optimiser_time_log.csv"), results_time, delimiter=",",
                       header="optimizer, run, time, accuracy")


if __name__ == "__main__":
    simple_federated_model()
    compare_optimizers()
