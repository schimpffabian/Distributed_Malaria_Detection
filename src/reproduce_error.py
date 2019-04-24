import torch
import syft as sy
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F


class Simple_CNN(torch.nn.Module):
    """

    """

    def __init__(self):
        super(Simple_CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 20, 3, 3)
        self.conv2 = torch.nn.Conv2d(20, 50, 3, 3)

        self.fc1 = torch.nn.Linear(50, 500)
        self.fc2 = torch.nn.Linear(500, 2)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, 2, 2)
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2, 2)
        x = x.view(x.shape[0], -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.nn.functional.log_softmax(x, dim=1)


def reproduce_error():
    """

    :return:
    """
    hook = sy.TorchHook(torch)
    model = Simple_CNN()
    # client = sy.VirtualWorker(hook, id="client")
    bob = sy.VirtualWorker(hook, id="bob")
    alice = sy.VirtualWorker(hook, id="alice")
    crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")

    model.fix_precision().share(alice, bob, crypto_provider=crypto_provider)


class Arguments:
    def __init__(self):
        self.batch_size = 64
        self.test_batch_size = 200
        self.epochs = 10
        self.lr = 0.001  # learning rate
        self.log_interval = 100


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def testrun(args, model, test_loader):
    model.eval()
    n_correct_priv = 0
    n_total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            n_correct_priv += pred.eq(target.view_as(pred)).sum()
            n_total += args.test_batch_size

            n_correct = n_correct_priv.copy().get().float_precision().long().item()

            print(
                "Test set: Accuracy: {}/{} ({:.0f}%)".format(
                    n_correct, n_total, 100.0 * n_correct / n_total
                )
            )


def reproduce_example():
    """

    :return:
    """
    hook = sy.TorchHook(torch)
    bob = sy.VirtualWorker(hook, id="bob")
    alice = sy.VirtualWorker(hook, id="alice")
    crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")

    args = Arguments()

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=args.test_batch_size,
        shuffle=True,
    )

    private_test_loader = []
    for data, target in test_loader:
        private_test_loader.append(
            (
                data.fix_prec().share(alice, bob, crypto_provider=crypto_provider),
                target.fix_prec().share(alice, bob, crypto_provider=crypto_provider),
            )
        )

    model = Net()
    model.fix_precision().share(alice, bob, crypto_provider=crypto_provider)

    testrun(args, model, private_test_loader)


if __name__ == "__main__":
    # reproduce_error()
    reproduce_example()
