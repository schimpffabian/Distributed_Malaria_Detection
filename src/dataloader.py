import torch
import os
from pathlib import Path
import numpy as np
from torchvision.datasets.folder import ImageFolder
import torchvision.transforms as transforms

# from torch.utils.data import Dataset


def randomize_background(x, dark_background=True):
    """
    Since images in the malaria datasets have default backgrounds around the cells and LIME
    indicates that NNs use the backgrounds shape to make predictions

    :param x: torch.tensor
    :param dark_background: Bool value of background True - 0,False - 255
    :return: Image with ranomized input
    """

    if len(x.shape) != 3:
        raise AttributeError("Make sure arrays fed to function have the right shape")

    for ii in range(x.shape[1]):
        for jj in range(x.shape[2]):
            if dark_background:
                if x[0][ii][jj] == 0 and x[1][ii][jj] == 0 and x[2][ii][jj] == 0:
                    x[0][ii][jj] = torch.rand(1)
                    x[1][ii][jj] = torch.rand(1)
                    x[2][ii][jj] = torch.rand(1)
            else:
                if x[0][ii][jj] == 255 and x[1][ii][jj] == 255 and x[2][ii][jj] == 255:
                    x[0][ii][jj] = torch.rand(1)
                    x[1][ii][jj] = torch.rand(1)
                    x[2][ii][jj] = torch.rand(1)
    return x


def get_data_augmentation(random_background, img_size, dark_background=True):
    """
    Standard composed transformations for data augmentation

    :param random_background: Bool - Randomize background
    :param img_size: Scales input images to square img_size
    :param dark_background: Bool - True - background has val 0 otherwise val 255
    :return:
    """
    # Define data augmentation and transforms
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if random_background:
        data_augmentation = transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size),
                transforms.RandomRotation((30)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: randomize_background(x)),
            ]
        )
    elif not random_background:
        data_augmentation = transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size),
                transforms.RandomRotation((30)),
                transforms.ToTensor(),
            ]
        )
    else:
        NotImplementedError

    return data_augmentation


def get_labels_and_class_counts(labels_list):
    """
    Calculates the counts of all unique classes.

    :param labels_list: list or ndarray with labels
    :return:
    """
    labels = np.array(labels_list)
    _, class_counts = np.unique(labels, return_counts=True)

    return labels, class_counts


def resample(target_list, imbal_class_prop):
    """
    Function adapted from ptrblck's PyTorch fork
    https://github.com/ptrblck/tutorials/blob/imbalanced_tutorial/intermediate_source/imbalanced_data_tutorial.py#L297
    Resample the indices to create an artificially imbalanced dataset.

    :param target_list
    :param imbal_class_prop
    :return: indices to satisfy the probabilities given in imbal_class_prop
    """
    targets, class_counts = get_labels_and_class_counts(target_list)
    nb_classes = len(imbal_class_prop)

    # Get class indices for resampling
    class_indices = [np.where(targets == i)[0] for i in range(nb_classes)]

    # Reduce class count by proportion
    imbal_class_counts = [
        int(count * prop) for count, prop in zip(class_counts, imbal_class_prop)
    ]

    # Get class indices for reduced class count
    idxs = []
    for c in range(nb_classes):
        imbal_class_count = imbal_class_counts[c]
        idxs.append(class_indices[c][:imbal_class_count])
    idxs = np.hstack(idxs)

    return idxs


def create_dataset(path, data_augmentation):
    """
    Convenience function for this project

    :param path: str path to root directory containing folders with samples for each class
    :param data_augmentation: torch transforms object
    :return: torch dataset
    """
    dataset = ImageFolder(root=path, transform=data_augmentation)
    return dataset


def split_dataset(dataset, percentage_of_dataset):
    """
    Separate dataset into parts with percentages specified in percentage_of_dataset

    :param dataset: torch Dataset to be split
    :param percentage_of_dataset: list or numpy array with percentage of each split
    :return: torch subsets
    """
    if np.array(percentage_of_dataset).sum() != 1:
        Warning("The provided percentages don't add up to 1")
        percentage_of_dataset = (
            np.array(percentage_of_dataset) / np.array(percentage_of_dataset).sum()
        )

    size_set = []

    for ii in range(len(percentage_of_dataset) - 1):
        size_set.append(int(percentage_of_dataset[ii] * len(dataset)))

    # Add last element so that all samples are used
    size_set.append(int(len(dataset) - np.array(size_set).sum()))

    split_datasets = torch.utils.data.random_split(dataset, size_set)
    return split_datasets


def set_prop_dataset(datasets, targets, balance):
    """
    Creates datasets with a given balance of classes

    :param datasets: subsets of datasets
    :param targets: lables of originial e.g. not split dataset
    :param balance: list of lists containing the wanted probabilities of each class in the given datasets
    :return: modified datasets
    """
    new_datasets = []
    for ii, dataset in enumerate(datasets):
        dataset_targets = [targets[ii] for ii in dataset.indices]
        idx = resample(dataset_targets, balance[ii])
        dataset.indices = list(dataset.indices[idx])
        new_datasets.append(dataset)
    return new_datasets


def create_dataloaders(
    batchsize=28,
    img_size=128,
    path=Path("../data/Classification"),
    random_background=False,
    num_workers=0,
    percentage_of_dataset=np.array([0.75, 0.2, 0.05]),
    balance=np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]),
):
    """
    Convenience function to setup dataloaders for experiments

    :param batchsize: (int) size of batch
    :param img_size: (int) size of image - underlying assumption of square images
    :param path: (str) path to folders containing images
    :param random_background: (bool) whether to randomize uniform backgrounds or not
    :param num_workers: (int) number of processes used to move data from RAM to GPU memory
    :param percentage_of_dataset: (tuple) percentages of each split
    :param balance: list of lists containing the wanted probabilities of each class in the given datasets
    :return: torch.utils.dataloader objects
    """
    data_augmentation = get_data_augmentation(random_background, img_size)

    # Load all data, create dataset
    dataset = create_dataset(path, data_augmentation)
    targets = dataset.targets

    # Split dataset into training, test and validation set
    # This is pretty clumsy but due to backward compatibility of PyTorch
    # https://github.com/pytorch/pytorch/pull/12068
    split_datasets = split_dataset(dataset, percentage_of_dataset)

    # Set probability of targets in each dataset
    split_datasets = set_prop_dataset(split_datasets, targets, balance)

    # Create dataloaders and set probability of classes in each dataset
    dataloaders = []
    for ii, dataset in enumerate(split_datasets):
        dataloaders.append(
            torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=batchsize,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True
            )
        )

    return tuple(dataloaders)


if __name__ == "__main__":
    train_loader, test_loader, val_loader = create_dataloaders()
