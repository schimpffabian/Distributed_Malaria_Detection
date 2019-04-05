import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.image import imread
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import numpy as np
from math import sqrt, ceil


def preprocessing():
    # Test if image is grayscale

    pass


def log_kernel(
    kernel_size, sigma_start=1.0, sigma_stop=10.0, num_sigma_steps=10, lin_sigma=True
):
    if kernel_size % 2 == 0:
        ValueError("please use odd kernel sizes")

    if lin_sigma is True:
        sigma_vec = np.linspace(sigma_start, sigma_stop, num_sigma_steps).tolist()
    elif lin_sigma is False:
        sigma_vec = np.logspace(sigma_start, sigma_stop, num_sigma_steps).tolist()
    else:
        NotImplementedError(
            "Please choose between: \nlinear scaling \t-> \t True \nlog scaling \t-> \t False"
        )
    print(sigma_vec)
    x_vec = np.arange(kernel_size) - np.floor(kernel_size / 2)
    y_vec = np.arange(kernel_size) - np.floor(kernel_size / 2)

    kernel = np.zeros((num_sigma_steps, kernel_size, kernel_size))
    kernel_sum = []
    for sigma_idx, sigma in enumerate(sigma_vec):
        kernel_sum.append(0)
        for x_idx, x in enumerate(x_vec.tolist()):
            for y_idx, y in enumerate(y_vec.tolist()):
                """ Old kernel
                kernel[sigma_idx, y_idx, x_idx] = - (1 - (x**2 + y**2) / (2 * sigma**2)) / \
                                                  (np.pi * sigma**2) * \
                                                  np.exp(- (x**2 + y**2) / (2 * sigma**2))
                """

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
        print(kernel[sigma_idx, :, :].sum())

    return kernel.tolist(), sigma_vec


def log(img,):
    num_filters = 1
    sigma_start = 3
    sigma_end = 3
    kernel_size = 4 * sigma_end + 1  # 2*ceil(3*sigma_end)+1
    padding = int(np.floor(kernel_size / 2))
    neighborhood_size = kernel_size
    neighborhood_padding = int(np.floor(neighborhood_size / 2))

    x = torch.tensor(img)  # torch.rand(1, 1, 64, 64)

    while len(x.shape) < 4:
        x = x.unsqueeze(0)

    kernel, sigma_vec = log_kernel(kernel_size, sigma_start, sigma_end, num_filters)

    gauss_kernel = torch.Tensor(kernel).unsqueeze(1).double()
    x_gauss = F.conv2d(x, gauss_kernel, padding=padding)

    x_gauss = x_gauss * -1
    maxpooling = torch.nn.MaxPool3d(
        kernel_size=neighborhood_size, stride=1, padding=neighborhood_padding
    )
    x_max = maxpooling(x_gauss)
    x_min = maxpooling(x_gauss * -1)

    avg_pooling = torch.nn.AvgPool2d(
        kernel_size=neighborhood_size, stride=1, padding=neighborhood_padding
    )
    x_avg = avg_pooling(x_gauss)

    x_gauss_np = torch.squeeze(x_gauss).data.numpy()

    x_gauss = torch.squeeze(x_gauss).data.numpy()
    x_max = torch.squeeze(x_max).data.numpy()
    x_avg = torch.squeeze(x_avg).data.numpy()
    x_min = torch.squeeze(x_min).data.numpy()

    maxima = (x_gauss == x_max) * 1
    minima = (x_gauss == x_min) * 3

    real_maxima = maxima + minima

    plt.figure(0)
    fig0, axes0 = plt.subplots(1, 4, figsize=(3, 3))
    ax0 = axes0.ravel()
    ax0[0].imshow(img)
    ax0[1].imshow(x_gauss_np)
    ax0[2].imshow(x_max)
    ax0[3].imshow(x_min)

    if num_filters > 1:
        sigma_idx, x, y = np.where(real_maxima == 1)
        print(sigma_idx)
        sigma = [sigma_vec[i] for i in sigma_idx]
    elif num_filters == 1:
        x, y = np.where(real_maxima == 1)
        sigma = [sigma_vec[0] for i in range(len(x))]

    return x, y, sigma


def get_images(path):
    return [f for f in listdir(path) if isfile(join(path, f))]


def rgb2gray(rgb):
    transform_factor = np.array([0.2989, 0.5870, 0.1140]).reshape((3, 1))
    return rgb @ transform_factor


# https://discuss.pytorch.org/t/changing-the-weights-of-conv2d/22992/14
if __name__ == "__main__":
    path = "../data/Real_Application/"
    images = get_images(path)
    # for image in images:
    image = "malaria_0.jpg"
    img = imread(path + image)
    image_gray = rgb2gray(img)
    image_gray = image_gray.squeeze()

    image_gray_inverse = image_gray
    # image_gray_inverse[0:-1, 0:-1] = 150
    # image_gray_inverse[45:90, 150:200] = 0

    # create_test_image
    test_img = np.zeros([200, 200])

    center = np.array([int(test_img.shape[0] / 2), int(test_img.shape[1] / 2)])
    radius = 5

    for ii in range(test_img.shape[1]):
        for jj in range(test_img.shape[0]):
            if np.linalg.norm(np.array([ii, jj]) - center) < radius:
                test_img[ii, jj] = 255
            else:
                test_img[ii, jj] = 0
    image_gray_inverse = test_img

    plt.figure(1)
    fig, axes = plt.subplots(2, 1, figsize=(3, 3))
    ax = axes.ravel()
    ax[0].imshow(image_gray_inverse)
    ax[1].imshow(image_gray_inverse)

    x, y, sigma = log(image_gray_inverse)
    r = np.array(sigma) * sqrt(2)

    for ii in range(len(x)):
        x_i = x[ii]
        y_i = y[ii]
        r_i = r[ii]
        print(r_i)
        c = plt.Circle((y_i, x_i), r_i, color="r", linewidth=2, fill=False)
        ax[1].add_patch(c)

    plt.show()


"""
    from scipy.ndimage import gaussian_laplace
    x_gauss_scipy = gaussian_laplace(torch.squeeze(x).data.numpy(), sigma=sigma_start)
    x_max_scipy = maxpooling(torch.tensor(x_gauss_scipy).unsqueeze(0).unsqueeze(0).double())

    x_max_scipy = torch.squeeze(x_max_scipy).data.numpy()
    maxima_scipy = (x_gauss_scipy == x_max_scipy) * 1

    difference_gauss = x_gauss_np - x_gauss_scipy
    #difference_max = x_max - x_max_scipy

    plt.figure(1)
    fig, axes = plt.subplots(1, 4, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()


    ax[0].imshow(img, interpolation='nearest')
    ax[1].imshow(x_gauss_np, interpolation='nearest')
    ax[2].imshow(x_gauss_scipy, interpolation='nearest')
    ax[3].imshow(difference_gauss, interpolation='nearest')
    plt.show()

"""
