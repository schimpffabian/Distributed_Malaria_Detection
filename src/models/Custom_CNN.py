import torch.nn as nn
import torch.nn.functional as F
from numpy import floor


def output_shape(
    c_in, c_out, kernel_size, stride=1, padding=0, dilation=1, n=1, h_in=1, w_in=1
):
    """

    :param C_in:
    :param C_out:
    :param kernel_size:
    :param stride:
    :param padding:
    :param dilation:
    :param N:
    :param H_in:
    :param W_in:
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
    def __init__(self):
        super(Simple_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 3, 3)
        self.conv2 = nn.Conv2d(20, 50, 3, 3)

        self.fc1 = nn.Linear(50, 500)
        self.fc2 = nn.Linear(500, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
