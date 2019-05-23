import torch.nn as nn
import torch.nn.functional as F
from numpy import floor
import torch


def output_shape(
    c_in, c_out, kernel_size, stride=1, padding=0, dilation=1, n=1, h_in=1, w_in=1
):
    """

    :param c_in:
    :param c_out:
    :param n:
    :param h_in:
    :param w_in:
    :param kernel_size:
    :param stride:
    :param padding:
    :param dilation:
    :return:
    """
    if not hasattr(kernel_size, "__len__"):
        kernel_size = (kernel_size, kernel_size)

    if not hasattr(stride, "__len__"):
        stride = (stride, stride)

    if not hasattr(padding, "__len__"):
        padding = (padding, padding)

    if not hasattr(dilation, "__len__"):
        dilation = (dilation, dilation)

    h_out = floor(
        (h_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / (stride[0])
        + 1
    )
    w_out = floor(
        (w_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / (stride[1])
        + 1
    )

    return n, c_out, h_out, w_out


class Simple_CNN(nn.Module):
    """
    Simple LeNet for image classification.
    It is designed to be trained in a reasonable time not to achieve SOTA performance
    """

    def __init__(self, img_size):
        super(Simple_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 50, 3, 3)
        self.conv2 = nn.Conv2d(50, 50, 3, 3)

        # determine output of convolutions
        x = torch.rand([1, 3, img_size, img_size])
        output = self.conv(x)
        output_size = output.shape[1]

        self.fc1 = nn.Linear(output_size, 800)
        self.fc2 = nn.Linear(800, 800)
        self.fc3 = nn.Linear(800, 2)

    def conv(self, x):
        x = self.conv1(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = x.view(x.shape[0], -1)
        return x

    def mlp(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = self.conv(x)
        x = self.mlp(x)

        return F.log_softmax(x, dim=1)


class Simple_CNN_e1(nn.Module):
    """
    Experiment 1

    Simple LeNet for image classification.
    It is designed to be trained in a reasonable time not to achieve SOTA performance
    """

    def __init__(self, img_size):
        super(Simple_CNN_e1, self).__init__()
        self.conv1 = nn.Conv2d(3, 50, 3, 3)
        self.conv2 = nn.Conv2d(50, 50, 3, 3)

        # determine output of convolutions
        x = torch.rand([1, 3, img_size, img_size])
        output = self.conv(x)
        output_size = output.shape[1]

        self.fc1 = nn.Linear(output_size, 800)
        self.fc2 = nn.Linear(800, 800)
        self.fc3 = nn.Linear(800, 2)

    def conv(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.shape[0], -1)
        return x

    def mlp(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = self.conv(x)
        x = self.mlp(x)

        return F.log_softmax(x, dim=1)


class Simple_CNN_e2(nn.Module):
    """
    Experiment 2

    Simple LeNet for image classification.
    It is designed to be trained in a reasonable time not to achieve SOTA performance
    """

    def __init__(self, img_size):
        super(Simple_CNN_e2, self).__init__()
        self.conv1 = nn.Conv2d(3, 50, 3, 3)
        self.conv2 = nn.Conv2d(50, 50, 3, 3)

        # determine output of convolutions
        x = torch.rand([1, 3, img_size, img_size])
        output = self.conv(x)
        output_size = output.shape[1]

        self.fc1 = nn.Linear(output_size, 800)
        self.fc2 = nn.Linear(800, 800)
        self.fc3 = nn.Linear(800, 2)

    def conv(self, x):
        x = self.conv1(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = x.view(x.shape[0], -1)
        return x

    def mlp(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = self.conv(x)
        x = self.mlp(x)

        return F.log_softmax(x, dim=1)


class Simple_CNN2(nn.Module):
    """
    Simple Network that is compatible with PySyft hooks at the time this is written
    """

    def __init__(self, img_size, num_neurons=500, num_layers=4):
        super(Simple_CNN2, self).__init__()
        self.img_size = img_size
        self.num_layers = num_layers
        self.fc1 = nn.Linear(img_size ** 2 * 3, num_neurons)
        self.fc2 = nn.Linear(num_neurons, num_neurons)
        self.fc3 = nn.Linear(num_neurons, 2)

    def forward(self, x):
        # Flatten input
        x = x.view(-1, 3 * self.img_size ** 2)
        x = self.fc1(x)
        x = F.relu(x)

        for ii in range(self.num_layers):
            x = self.fc2(x)
            x = F.relu(x)

        x = self.fc3(x)
        return x
