import sys
import os
import numpy as np
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.join(".."))
from src.dataloader import create_dataloaders
from src.auxiliaries import rgb2gray
from src.analysis.plot_config import params


def create_histogram(path, randomize_background):
    """
    Creates normalized histograms of the training images

    :param str path: path to use for saving raw data
    :param bool randomize_background: Randomize uniform background
    """
    num_episodes = 1
    batch_size = 1
    img_size = 32
    bins = int(256/2)
    torch.manual_seed(42)

    train_loader, test_loader, validation_loader = create_dataloaders(batch_size, img_size=img_size,
                                                                      random_background=randomize_background)
    del validation_loader

    num_run = 0
    bin_counts_cum = None
    for episode in range(num_episodes):
        print("\nRun \t %.0f" % episode)
        for data, target in train_loader:
            del target

            # Reshape data for grayscale conversion
            data = data.view((-1, img_size, img_size, 3))
            data_train = data.detach().cpu().numpy()
            data_gray_train = rgb2gray(data_train)

            bin_counts = np.histogram(data_gray_train, bins)

            if bin_counts_cum is None:
                bin_counts_cum = bin_counts[0]
            else:
                bin_counts_cum += bin_counts[0]

            num_run += 1
            if num_run % 1000 == 0:
                print(num_run)

    normalized_hist = bin_counts_cum / num_run / (img_size*img_size)
    np.savetxt(Path(path), normalized_hist, delimiter=",")


def load_histogram(load_path):
    """
    Loads histograms that were previously created

    :param str load_path: path to raw data
    """
    # Load data
    data = np.genfromtxt(Path(load_path))

    # Plotting settings
    params['figure.figsize'] = [8, 4]
    matplotlib.rcParams.update(params)
    width = 0.5

    # Set up figure
    fig, ax = plt.subplots()
    del fig

    # Plot
    ind = np.arange(256/2)*2
    ax.bar(ind, data, width)

    # Refine plot
    plt.ylabel("Normalized probability")
    plt.xlabel("Gray value")
    plt.ylim((0, 0.03))
    plt.grid()
    plt.tight_layout()
    plt.show()


def compare_histograms(random_path, standard_path):
    """
    Compare randomized background with standard

    :param random_path:
    :param standard_path:
    :return:
    """
# Load data
    random_data = np.genfromtxt(Path(random_path))
    standard_data = np.genfromtxt(Path(standard_path))

    # Plotting settings
    params['figure.figsize'] = [8, 4]
    matplotlib.rcParams.update(params)
    width = 0.4

    # Set up figure
    fig, ax = plt.subplots()
    del fig

    # Plot
    ind = np.arange(256/2)*2
    ax.bar(ind+0.2, random_data, width, label="Randomized background")
    ax.bar(ind-0.2, standard_data, width, label="Standard background")

    # Refine plot
    plt.ylabel("Normalized probability")
    plt.xlabel("Gray value")
    plt.ylim((0, 0.03))
    plt.grid()
    plt.tight_layout()
    plt.legend()
    plt.show()


def main():
    """
    Main function for evaluating the input distribution
    """
    random_path = "./logs/normalized_histogram_random.csv"
    standard_path = "./logs/normalized_histogram_standard.csv"

    # create_histogram(random_path, True)
    # create_histogram(standard_path, False)

    # load_histogram(standard_path)
    compare_histograms(random_path, standard_path)


if __name__ == "__main__":
    main()
