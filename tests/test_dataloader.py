import sys
import os
import torch
import torch.utils.data as utils

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataloader import randomize_background
from src.dataloader import get_labels_and_class_counts
from src.dataloader import split_dataset
from src.dataloader import set_prop_dataset

def test_randomize_background():
    """
    test randomize_background function
    """
    # Test 1 Change random background dark background
    start_img_1 = torch.rand(3, 4, 4)
    start_img_1[:, 0:2, 0:2] = 0

    randomized_img_1 = randomize_background(start_img_1)
    assert randomized_img_1[0, 0, 0].item() > 0

    # Test 2 Change random background bright background
    start_img_2 = torch.rand(3, 4, 4)
    start_img_2[:, 0:2, 0:2] = 255

    randomized_img_1 = randomize_background(start_img_1, dark_background=False)
    assert randomized_img_1[0, 0, 0].item() < 255

    # Test 3 wrong format
    img_false_format = torch.rand(4, 4)
    try:
        randomize_background(img_false_format)
        assert False
    except AttributeError:
        assert True


def test_get_labels_and_class_counts():
    """
    Test function get_labels_and_class_counts
    """
    # Test 1
    labels = [1, 1, 1, 0, 0, 0]
    _, counts = get_labels_and_class_counts(labels)

    assert counts[0] == 3
    assert counts[1] == 3
    assert len(counts) == 2

    # Test 2
    labels_2 = [1, 1, 2, 0, 0, 0]
    _, counts_2 = get_labels_and_class_counts(labels_2)
    assert counts_2[0] == 3
    assert counts_2[1] == 2
    assert counts_2[2] == 1
    assert len(counts_2) == 3


def test_split_dataset():
    """
    test random splits of dataset
    """
    # Dummy Dataset

    dataset = utils.TensorDataset(torch.rand(100, 3, 42, 42), torch.rand(100))

    percentages = [0.4, 0.5, 0.1]

    datasets = split_dataset(dataset, percentages)

    samples_in_split = []
    for ii in range(len(datasets)):
        samples_in_split.append(len(datasets[ii].indices))

    assert samples_in_split[0] == percentages[0] * len(dataset)
    assert samples_in_split[1] == percentages[1] * len(dataset)
    assert samples_in_split[2] == percentages[2] * len(dataset)
    assert len(percentages) == len(datasets)


def test_resample():
    """
    List of
    """


def test_set_prop_dataset():
    """
    make sure that datasets can be split as intended
    """
    torch.manual_seed(42)

    labels = torch.zeros([300])
    labels[0:150] = 1

    dataset_original = utils.TensorDataset(torch.rand(300, 3, 42, 42), labels)
    percentages = [0.33, 0.33, 0.334]

    datasets = split_dataset(dataset_original, percentages)

    balances = [[0.5, 0.5],
                [0.4, 0.6],
                [0.01, 0.99]]

    modified_datasets = set_prop_dataset(datasets, labels, balances)
    print("")
    for ii, dataset in enumerate(modified_datasets):
        indices = dataset.indices
        dataset_labels = labels[indices]
        print(abs(torch.sum(dataset_labels).item() / len(dataset_labels) - balances[ii][1]))
        assert abs(torch.sum(dataset_labels).item() / len(dataset_labels) - balances[ii][1]) < 0.05


if __name__ == "__main__":
    test_randomize_background()
    test_get_labels_and_class_counts()
    test_split_dataset()
