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
        "Simple Model CPU",
        "Simple Model GPU",
        "Squeezenet CPU",
        "Squeezenet GPU",
    ]

    # Load data
    simple_model = np.genfromtxt(
        Path("../logs/results_experiment_2.csv"), delimiter=",", skip_header=1
    )
    squeezenet = np.genfromtxt(
        Path("../logs/results_experiment_5.csv"), delimiter=",", skip_header=1
    )

    # Create lists to store results
    simple_time_gpu = []
    simple_time_cpu = []
    squeezenet_time_gpu = []
    squeezenet_time_cpu = []

    mean_list = []
    std_list = []

    # Fill lists
    for row in range(simple_model.shape[0]):
        if simple_model[row, 4] == 1:
            simple_time_gpu.append(simple_model[row, 6])
        else:
            simple_time_cpu.append(simple_model[row, 6])

    for row in range(squeezenet.shape[0]):
        if squeezenet[row, 4] == 1:
            squeezenet_time_gpu.append(squeezenet[row, 6])
        else:
            squeezenet_time_cpu.append(squeezenet[row, 6])

    time_data = [simple_time_cpu, simple_time_gpu, squeezenet_time_cpu, simple_time_gpu]

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
