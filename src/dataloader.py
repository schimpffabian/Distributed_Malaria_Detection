import torch
import numpy as np
from torchvision.datasets.folder import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import Dataset


def randomize_background(x):
    for ii in range(x.shape[1]):
        for jj in range(x.shape[2]):
            if x[0][ii][jj] == 0 and x[1][ii][jj] == 0 and x[2][ii][jj] == 0:
                x[0][ii][jj] = torch.rand(1)
                x[1][ii][jj] = torch.rand(1)
                x[2][ii][jj] = torch.rand(1)
    return x


def get_data_augmentation(random_background, img_size):
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
    """
    labels = np.array(labels_list)
    _, class_counts = np.unique(labels, return_counts=True)

    return labels, class_counts


def resample(target_list, imbal_class_prop):
    """
    Function adapted from ptrblck's PyTorch fork
    https://github.com/ptrblck/tutorials/blob/imbalanced_tutorial/intermediate_source/imbalanced_data_tutorial.py#L297
    Resample the indices to create an artificially imbalanced dataset.
    :return:
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


def create_dataloaders(
    batchsize=28,
    img_size=28,
    path="../data/Classification",
    random_background=False,
    num_workers=0,
    percentage_of_dataset=np.array([0.75, 0.2, 0.05]),
    balance=np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]),
):
    data_augmentation = get_data_augmentation(random_background, img_size)

    # Load all data, create dataset
    dataset = ImageFolder(root=path, transform=data_augmentation)
    targets = dataset.targets

    # Split dataset into training, test and validation set
    # This is pretty clumsy but due to backward compatibility of PyTorch https://github.com/pytorch/pytorch/pull/12068

    size_set = []
    for ii in range(len(percentage_of_dataset) - 1):
        size_set.append(int(percentage_of_dataset[ii] * len(dataset)))
    size_set.append(int(len(dataset) - np.array(size_set).sum()))
    split_datasets = torch.utils.data.random_split(dataset, size_set)

    # Create dataloaders and set probability of classes in each dataset
    dataloaders = []
    for ii, dataset in enumerate(split_datasets):
        dataset_targets = [targets[ii] for ii in dataset.indices]
        idx = resample(dataset_targets, balance[ii])
        dataset.indices = list(dataset.indices[idx])
        dataloaders.append(
            torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=batchsize,
                shuffle=True,
                num_workers=num_workers,
            )
        )

    return tuple(dataloaders)


if __name__ == "__main__":
    train_loader, test_loader, val_loader = create_dataloaders()

"""       
   size_train = int(0.75 * len(dataset))
   size_test = int(0.2 * len(dataset))
   size_val = int(len(dataset) - size_train - size_test)

   train_set, test_set, val_set = torch.utils.data.random_split(
       dataset, [size_train, size_test, size_val]
   )

   val_set_targets = [targets[ii] for ii in val_set.indices]
   print(len(val_set_targets))
   idx = resample(val_set_targets, np.array([0.2, 0.8]))
   val_set.indices = list(val_set.indices[idx])
   val_set_targets = [targets[ii] for ii in val_set.indices]
   print(len(val_set_targets))

   # Create dataloaders
   train_loader = torch.utils.data.DataLoader(
       dataset=train_set, batch_size=batchsize, shuffle=True, num_workers=num_workers
   )

   test_loader = torch.utils.data.DataLoader(
       dataset=test_set, batch_size=batchsize, shuffle=True, num_workers=num_workers
   )

   val_loader = torch.utils.data.DataLoader(
       dataset=val_set, batch_size=batchsize, shuffle=True, num_workers=num_workers
   )


   return train_loader, test_loader, val_loader
"""
