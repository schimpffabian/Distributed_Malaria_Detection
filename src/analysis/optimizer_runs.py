import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from src.analysis.plot_config import params


def main():
    params['figure.figsize'] = [9, 4]
    matplotlib.rcParams.update(params)

    # Define labels and colors
    """
    labels = ["Adam \tlr=1e-3",
              "SGD  \tlr=1e-3",
              "SGD  \tlr=1e-1"]
    """
    labels = ["SGD \tlr=0.5",
              "SGD  \tlr=1e-1",
              "SGD  \tlr=1e-2"]

    colors = ["g", "b", "r"]

    # Define lists for plotting
    optimizer = []
    epochs = []
    accuracies = []

    last_run = -1
    epoch_temp = []
    accuracy_temp = []

    # Load data
    data = np.genfromtxt("../logs/optimizer_acc_log_3.csv", delimiter=",", skip_header=1)
    first = True

    # separate different runs with different optimizers
    for ii in range(data.shape[0]):
        if data[ii, 1] != last_run:

            if not first:
                epochs.append(epoch_temp)
                optimizer.append(int(data[ii-1, 0]))
                accuracies.append(accuracy_temp)

            else:
                first = False

            last_run = int(data[ii, 1])
            epoch_temp = []
            accuracy_temp = []

        epoch_temp.append(data[ii, 2])
        accuracy_temp.append(data[ii, 3])

    epochs.append(epoch_temp)
    optimizer.append(int(data[ii, 0]))
    accuracies.append(accuracy_temp)

    # plot
    used_labels = []
    for index in range(len(optimizer)):
        if labels[optimizer[index]] in used_labels:
            plt.plot(epochs[index], accuracies[index], colors[optimizer[index]])
        else:
            plt.plot(epochs[index], accuracies[index], colors[optimizer[index]], label=labels[optimizer[index]])
            used_labels.append(labels[optimizer[index]])

    plt.legend()
    plt.grid()
    plt.xlim((0, 100))
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()