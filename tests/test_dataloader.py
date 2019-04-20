import sys
import os
import torch
import torch.utils.data as utils

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataloader import randomize_background
from src.dataloader import get_labels_and_class_counts
from src.dataloader import split_dataset


def test_randomize_background():
    """
    test randomize_background function
    :return:
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
    :return:
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


if __name__ == "__main__":
    test_randomize_background()
    test_get_labels_and_class_counts()
    test_split_dataset()
