import torch
import torch.nn as nn
import torch.optim as optim
import timeit
import numpy as np
from pathlib import Path
from src.federated_training import create_federated_dataset
from src.federated_training import DatasetFromSubset
from src.auxiliaries import train
from src.auxiliaries import run_t
from src.models.Custom_CNN import Simple_CNN_e2


def compare_optimizers():
    """
    train a model with data federated over multiple workers.
    Comparison to federated training since this suffered from poor optimizer performance
    """
    # Parameters and general setup
    epochs = 100
    torch.manual_seed(42)

    # Create datasets and dataloaders
    train_set, test_set = create_federated_dataset()
    train_dataset = DatasetFromSubset(train_set)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1024, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1024, shuffle=True)

    # Setup optimization
    device = "cpu"
    results_time = []
    results_acc = []

    for opt in range(3):
        for run in range(3):
            # load model
            model = Simple_CNN_e2(img_size=128)
            model = model.float()

            """ Run 2
            if opt == 0:
                optimizer = optim.Adam(model.parameters(), lr=1e-3)
            elif opt == 1:
                optimizer = optim.SGD(model.parameters(), lr=1e-3)
            elif opt == 2:
                optimizer = optim.SGD(model.parameters(), lr=1e-1)
            """

            if opt == 0:
                optimizer = optim.SGD(model.parameters(), lr=0.5)
            elif opt == 1:
                optimizer = optim.SGD(model.parameters(), lr=1e-1)
            elif opt == 2:
                optimizer = optim.SGD(model.parameters(), lr=1e-2)

            loss = nn.CrossEntropyLoss()

            start = timeit.default_timer()
            for epoch in range(1, epochs + 1):
                train(
                    model, device, train_loader, optimizer, epoch, loss, federated=False
                )
                accuracy = run_t(model, device, test_loader, loss)
                results_acc.append([opt, run, epoch, accuracy])
                np.savetxt(
                    Path("../logs/optimizer_acc_log_3.csv"),
                    results_acc,
                    delimiter=",",
                    header="optimizer, run, epoch, accuracy",
                )

            end = timeit.default_timer()
            results_time.append([opt, run, end - start, accuracy])
            np.savetxt(
                Path("../logs/optimizer_time_log_3.csv"),
                results_time,
                delimiter=",",
                header="optimizer, run, time, accuracy",
            )


if __name__ == "__main__":
    compare_optimizers()
