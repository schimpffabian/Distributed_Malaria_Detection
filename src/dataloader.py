import torch
from torchvision.datasets.folder import ImageFolder
import torchvision.transforms as transforms

from torch.utils.data import Dataset


def create_dataloaders(batchsize=28, img_size=28):
    # Define data augmentation and transforms
    data_augmentation = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size),
            transforms.RandomRotation((30)),
            transforms.ToTensor(),
        ]
    )
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # Load all data, create dataset
    dataset = ImageFolder(root="../data/Classification", transform=data_augmentation)

    # Split dataset into training, test and validation set
    # This is pretty clumsy but due to backward compatibility of PyTorch https://github.com/pytorch/pytorch/pull/12068
    size_train = int(0.75 * len(dataset))
    size_test = int(0.2 * len(dataset))
    size_val = int(len(dataset) - size_train - size_test)
    train_set, test_set, val_set = torch.utils.data.random_split(
        dataset, [size_train, size_test, size_val]
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=batchsize, shuffle=True, num_workers=0
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=batchsize, shuffle=True, num_workers=0
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_set, batch_size=batchsize, shuffle=True, num_workers=0
    )

    return train_loader, test_loader, val_loader


if __name__ == "__main__":
    train_loader, test_loader, val_loader = create_dataloaders()
