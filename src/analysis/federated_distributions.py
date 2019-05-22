import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from src.analysis.plot_config import params


def main():
    params['figure.figsize'] = [8, 4]
    matplotlib.rcParams.update(params)

    # Define lists for plotting
    distribution = []
    epochs = []
    accuracies = []

    last_run = -1
    epoch_temp = []
    accuracy_temp = []

    # Load data
    name_str = "../logs/federated_learning_speed_distribution_yoda.csv"

    data = np.genfromtxt(name_str, delimiter=",", skip_header=1)

    # Define labels and colors
    if name_str == "../logs/federated_learning_speed_distribution_yoda.csv":
        data = np.genfromtxt(name_str, delimiter=";", skip_header=1)
        labels = ["50 / 50",
                  "70 / 30",
                  "90 / 10"
                  ]
        colors = ["b", "r", "g"]

    elif name_str == "../logs/federated_learning_speed_distribution_yoda2.csv":
        labels = ["50 / 50",
                  "60 / 40",
                  ]
        colors = ["b", "r"]
    else:
        raise NotImplementedError

    first = True

    # separate different runs with different optimizers
    for ii in range(data.shape[0]):
        if data[ii, 0] != last_run:

            if not first:
                epochs.append(epoch_temp)
                distribution.append(int(data[ii-1, 1]))
                accuracies.append(accuracy_temp)

            else:
                first = False

            last_run = int(data[ii, 0])
            epoch_temp = []
            accuracy_temp = []

        epoch_temp.append(data[ii, 2])
        accuracy_temp.append(data[ii, 3])

    epochs.append(epoch_temp)
    distribution.append(int(data[ii, 1]))
    accuracies.append(accuracy_temp)

    # plot
    used_labels = []
    for index in range(len(distribution)):
        if labels[distribution[index]] in used_labels:
            plt.plot(epochs[index], accuracies[index], colors[distribution[index]])
        else:
            plt.plot(epochs[index], accuracies[index], colors[distribution[index]], label=labels[distribution[index]])
            used_labels.append(labels[distribution[index]])

    plt.legend()
    plt.grid()
    if name_str == "../logs/federated_learning_speed_distribution_yoda.csv":
        plt.xlim((0, 150))

    elif name_str == "../logs/federated_learning_speed_distribution_yoda2.csv":
        plt.xlim((0, 300))
    else:
        raise NotImplementedError

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy [\%]")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()