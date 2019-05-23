import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
from src.analysis.plot_config import params


def main():
    # Plotting settings
    params["figure.figsize"] = [8, 4]
    matplotlib.rcParams.update(params)
    width = 0.5
    labels = [
        "Regular",
        "Federated 2 worker",
        "Federated 3 worker",
        "Federated 4 worker",
    ]

    # Load data
    regular_federated = np.genfromtxt(
        Path("../logs/federated_malaria_runtime.csv"), delimiter=",", skip_header=1
    )
    multi_worker = np.genfromtxt(
        Path("../logs/federated_malaria_runtime_multiple_workers.csv"),
        delimiter=",",
        skip_header=1,
    )

    # Create lists to store results
    regular = []
    federated_2 = []
    federated_3 = []
    federated_4 = []

    mean_list = []
    std_list = []

    # Fill lists
    for row in range(regular_federated.shape[0]):
        if regular_federated[row, 0] == 1:
            federated_2.append(regular_federated[row, 1])
        else:
            regular.append(regular_federated[row, 1])

    for row in range(multi_worker.shape[0]):
        if multi_worker[row, 0] == 2:
            federated_2.append(multi_worker[row, 1])
        elif multi_worker[row, 0] == 3:
            federated_3.append(multi_worker[row, 1])
        elif multi_worker[row, 0] == 4:
            federated_4.append(multi_worker[row, 1])

    time_data = [regular, federated_2, federated_3, federated_4]

    # Analyse time data
    for data in time_data:
        mean_list.append(np.mean(data))
        std_list.append(np.std(data))

    # Uniform plotting
    ind = np.arange(4)
    plt.bar(ind, mean_list, width, yerr=std_list)

    plt.xticks(ind, labels)
    plt.ylabel("Training time [s]")

    # plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
