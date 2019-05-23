import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, floor
from skimage.feature import blob_log
from src.auxiliaries import create_test_img
import timeit


def log_kernel(
    kernel_size, sigma_start=1.0, sigma_stop=10.0, num_sigma_steps=10, lin_sigma=True
):
    """
    Creates Kernels used for Laplacian of Gaussian convolutions

    :param int kernel_size: size of the kernel, should be odd
    :param float sigma_start: smallest kernel parameter
    :param float sigma_stop: biggest kernel parameter
    :param int num_sigma_steps: number of different kernels
    :param bool lin_sigma: use linear or logarithmic scaling between sigma_start and sigma_end
    :return: kernel
    """
    if kernel_size % 2 == 0:
        raise ValueError("please use odd kernel sizes")

    if lin_sigma is True:
        sigma_vec = np.linspace(sigma_start, sigma_stop, num_sigma_steps).tolist()
    elif lin_sigma is False:
        sigma_vec = np.logspace(sigma_start, sigma_stop, num_sigma_steps).tolist()
    else:
        raise NotImplementedError(
            "Please choose between: \nlinear scaling \t-> \t True \nlog scaling \t-> \t False"
        )
    # print(sigma_vec)
    x_vec = np.arange(kernel_size) - np.floor(kernel_size / 2)
    y_vec = np.arange(kernel_size) - np.floor(kernel_size / 2)

    kernel = np.zeros((num_sigma_steps, kernel_size, kernel_size))
    kernel_sum = []
    for sigma_idx, sigma in enumerate(sigma_vec):
        kernel_sum.append(0)
        for x_idx, x in enumerate(x_vec.tolist()):
            for y_idx, y in enumerate(y_vec.tolist()):
                # Old kernel
                # kernel[sigma_idx, y_idx, x_idx] = - (1 - (x**2 + y**2) / (2 * sigma**2)) / \
                #                                   (np.pi * sigma**2) * \
                #                                   np.exp(- (x**2 + y**2) / (2 * sigma**2))

                kernel[sigma_idx, x_idx, y_idx] = (
                    (x ** 2 + y ** 2 - 2 * sigma ** 2)
                    / (2 * np.pi * sigma ** 6)
                    * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
                )
                kernel_sum[sigma_idx] += kernel[sigma_idx, x_idx, y_idx]

    # Normalize kernels
    for sigma_idx, sigma in enumerate(sigma_vec):
        # kernel[sigma_idx, :, :] = kernel[sigma_idx, :, :] / kernel_sum[sigma_idx] #* -1
        kernel[sigma_idx, :, :] = kernel[sigma_idx, :, :] * sigma ** 2
        # print(kernel[sigma_idx, :, :].sum())

    return kernel.tolist(), sigma_vec


def pt_log(img, min_sigma=5, max_sigma=5, num_sigma=1, exclude_borders=True):
    """
    PyTorch implementation of Laplassian of Gaussian

    :param img: input image
    :param float sigma_start: smallest kernel parameter
    :param float sigma_stop: biggest kernel parameter
    :param int num_sigma_steps: number of different kernels
    :param bool exclude_borders: exclude extrema close to the image border
    :return: x-location, y-location, radius of blobs
    """
    num_filters = num_sigma
    sigma_start = min_sigma
    sigma_end = max_sigma

    kernel_size = sigma_end + 1  # 2*ceil(3*sigma_end)+1
    safe_dist = floor(kernel_size / 2)
    padding = int(np.floor(kernel_size / 2))
    neighborhood_size = kernel_size
    neighborhood_padding = int(np.floor(neighborhood_size / 2))

    x = torch.tensor(img)  # torch.rand(1, 1, 64, 64)
    noise = torch.rand(x.shape).double()
    x = x + noise

    while len(x.shape) < 4:
        x = x.unsqueeze(0)

    kernel, sigma_vec = log_kernel(kernel_size, sigma_start, sigma_end, num_filters)
    gauss_kernel = torch.Tensor(kernel).unsqueeze(1).double()

    start_pt = timeit.default_timer()
    x_gauss = F.conv2d(x, gauss_kernel, padding=padding)

    x_gauss = x_gauss * -1
    maxpooling = torch.nn.MaxPool3d(
        kernel_size=neighborhood_size, stride=1, padding=neighborhood_padding
    )
    x_max = maxpooling(x_gauss)

    # maxima = torch.eq(x_max, x_gauss)
    # indices = (torch.squeeze(maxima) == 1).nonzero()

    x_gauss_np = torch.squeeze(x_gauss).data.numpy()
    x_max_np = torch.squeeze(x_max).data.numpy()
    maxima_np = (x_gauss_np == x_max_np) * 1

    if num_filters > 1:
        sigma_idx, x, y = np.where(maxima_np == 1)
        # print(sigma_idx)
        sigma = [sigma_vec[i] for i in sigma_idx]
    elif num_filters == 1:
        x, y = np.where(maxima_np == 1)
        sigma = [sigma_vec[0] for i in range(len(x))]

    x_val = []
    y_val = []
    sigma_val = []

    if exclude_borders:
        for ii in range(len(x)):
            x_i = x[ii]
            if safe_dist < x_i < img.shape[0] - safe_dist:
                y_i = y[ii]
                if safe_dist < y_i < img.shape[1] - safe_dist:
                    sigma_i = sigma[ii]
                    x_val.append(x_i)
                    y_val.append(y_i)
                    sigma_val.append(sigma_i)

    else:
        x_val = x
        y_val = y
        sigma_val = sigma

    return x_val, y_val, sigma_val, safe_dist, start_pt


# https://discuss.pytorch.org/t/changing-the-weights-of-conv2d/22992/14
def benchmark_log(num_runs=1):
    """
    Compare different LoG from Sklearn with PyTorch implementation

    :param int num_runs: number of runs
    """

    time_pt_log = 0
    time_sk_log = 0

    for ii in range(num_runs):
        test_img, center_list, radius_list = create_test_img(
            size=(100, 100), num_points=4
        )
        del center_list
        del radius_list

        fig, axes = plt.subplots(3, 1, figsize=(3, 3))
        del fig
        ax = axes.ravel()
        ax[0].imshow(test_img)
        ax[1].imshow(test_img)
        ax[2].imshow(test_img)

        x, y, sigma, safe_dist, start_pt = pt_log(
            test_img, min_sigma=1, max_sigma=10, num_sigma=10, exclude_borders=False
        )
        end_pt = timeit.default_timer()
        r = np.array(sigma) * sqrt(2)
        time_pt_log += end_pt - start_pt

        start_sk = timeit.default_timer()
        blobs_log = blob_log(test_img, min_sigma=1, max_sigma=10, num_sigma=10)
        end_sk = timeit.default_timer()
        x_sk = blobs_log[:, 0]
        y_sk = blobs_log[:, 1]
        r_sk = blobs_log[:, 2] * sqrt(2) * sqrt(2)
        time_sk_log += end_sk - start_sk

        for ii in range(len(x)):
            x_i = x[ii]
            y_i = y[ii]
            r_i = r[ii]
            c = plt.Circle((y_i, x_i), r_i, color="r", linewidth=2, fill=False)
            ax[1].add_patch(c)

        for ii in range(len(x_sk)):
            x_i = x_sk[ii]
            y_i = y_sk[ii]
            r_i = r_sk[ii]
            c = plt.Circle((y_i, x_i), r_i, color="r", linewidth=2, fill=False)
            ax[2].add_patch(c)
        plt.show()

    print("Total time for sklearn log: ", time_sk_log / num_runs)
    print("Total time for PyTorch log: ", time_pt_log / num_runs)


if __name__ == "__main__":
    benchmark_log()
