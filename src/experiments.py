import torch


def get_max():
    a = [[[1, 3, 1], [1, 4, 1], [1, 2, 1]]]
    a_tensor = torch.tensor(a)

    print((a_tensor == torch.max(a_tensor)).nonzero())


if __name__ == "__main__":
    get_max()
