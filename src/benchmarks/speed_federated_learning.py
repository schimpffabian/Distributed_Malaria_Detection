import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import syft as sy  # <-- NEW: import the Pysyft library
import timeit
import numpy as np
from pathlib import Path
from src.models.Custom_CNN import Simple_CNN_e2
from src.federated_training import create_federated_dataset
from src.federated_training import DatasetFromSubset

hook = sy.TorchHook(torch)
bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")
mike = sy.VirtualWorker(hook, id="mike")
zoe = sy.VirtualWorker(hook, id="zoe")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Arguments:
    def __init__(self):
        self.batch_size = 1024
        self.test_batch_size = 1000
        self.epochs = 3
        self.lr = 0.01
        self.momentum = 0.5
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 30
        self.save_model = False


def train(args, model, device, federated_train_loader, optimizer, epoch, federate):
    """
    Training function used by PySyft in their examples
    :param Arguments args: Class providing (hyper)parameters
    :param model: PyTorch model
    :param str device: device to train on
    :param torch.utils.Dataloader federated_train_loader: dataloader
    :param torch.optim optimizer: optimizer used to update weights and biases
    :param epoch: Number of epochs to run
    :param bool federate: whether the execution should be federated
    :return: time passed in training loop
    """
    model.train()
    for batch_idx, (data, target) in enumerate(federated_train_loader):
        if federate:
            model.send(data.location)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if federate:
            model.get()
        if batch_idx % args.log_interval == 0:
            if federate:
                loss = loss.get()
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * args.batch_size,
                    len(federated_train_loader) * args.batch_size,
                    100.0 * batch_idx / len(federated_train_loader),
                    loss.item(),
                )
            )


def mnist(federate):
    args = Arguments()
    use_cuda = False
    # torch.manual_seed(args.seed)
    device = torch.device("cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    model = Net().to(device)
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr
    )  # TODO momentum is not supported at the moment

    if federate:
        train_loader = sy.FederatedDataLoader(
            datasets.MNIST(
                Path("../../data"),
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                ),
            ).federate((bob, alice)),
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs
        )

    else:
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                Path("../../data"),
                train=False,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                ),
            ),
            batch_size=args.test_batch_size,
            shuffle=True,
            **kwargs
        )

    start = timeit.default_timer()
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, federate)
    end = timeit.default_timer()
    return end - start


def malaria(federate, num_worker=2):
    """
    Function to benchmark different numbers of workers and compare it with regular execution

    :param bool federate: Whether to use federated training
    :param int num_worker:  Number of workers
    :return: time for training
    """
    image_size = 128
    args = Arguments()
    use_cuda = False  # not args.no_cuda and torch.cuda.is_available()
    # torch.manual_seed(args.seed)
    device = torch.device("cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    model = Simple_CNN_e2(128).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    train_set, test_set = create_federated_dataset()
    train_dataset = DatasetFromSubset(train_set)

    # Only difference in setup
    if federate:
        if num_worker == 2:
            workers = (bob, alice)
        elif num_worker == 3:
            workers = (bob, alice, mike)
        elif num_worker == 4:
            workers = (bob, alice, mike, zoe)
        else:
            raise NotImplementedError

        train_loader = sy.FederatedDataLoader(
            train_dataset.federate(workers),
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
        )

    # Train cycles
    start = timeit.default_timer()
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, federate)
    end = timeit.default_timer()

    return end - start


def eval_mnist():
    """
    Analyze runtimes of federated and regular execution on MNIST data
    """
    results = []

    for federate in [1, 0]:
        for run in range(3):
            duration = mnist(federate)
            results.append([federate, duration])

            np.savetxt(
                Path("../logs/federated_mnist.csv"),
                results,
                delimiter=",",
                header="federated, time",
            )


def eval_malaria_00():
    """
    Analyze runtimes of federated and regular execution on malaria data
    """
    results = []

    for federate in [0, 1]:
        for run in range(3):
            duration = malaria(federate)
            results.append([federate, duration])

            np.savetxt(
                Path("../logs/federated_malaria_runtime.csv"),
                results,
                delimiter=",",
                header="federated, time",
            )


def eval_malaria_01():
    """
    Analyze runtimes of federated training with different numbers of agents
    """
    results = []
    federate = True

    for num_worker in [2, 3, 4]:
        for run in range(3):
            duration = malaria(federate, num_worker=num_worker)
            results.append([num_worker, duration])

            np.savetxt(
                Path("../logs/federated_malaria_runtime_multiple_workers.csv"),
                results,
                delimiter=",",
                header="num_worker, time",
            )


if __name__ == "__main__":
    # eval_mnist()
    # eval_malaria_00()
    eval_malaria_01()
