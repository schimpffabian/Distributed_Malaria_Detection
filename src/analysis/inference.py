import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from src.analysis.plot_config import params


def main():
    # Set plotting parameters
    params["figure.figsize"] = [9, 4]
    matplotlib.rcParams.update(params)
    width = 0.5

    # Load data
    data = np.genfromtxt(
        "../logs/inference_speedup_squeezenet.csv", skip_header=1, delimiter=","
    )
    # data = np.genfromtxt("../logs/inference_speedup_custom_e2.csv", skip_header=1, delimiter=",")
    # Define lists for later plotting
    tracing_gpu = []
    tracing_cpu = []
    gpu = []
    cpu = []

    mean_list = []
    std_list = []

    # Labels
    labels = ["CPU", "CPU + Tracing", "GPU", "GPU + Tracing"]

    # Fill matrices
    for ii in range(data.shape[0]):
        if data[ii, 0] == 0:
            if data[ii, 1] == 0:
                cpu.append(data[ii, 2])
            else:
                tracing_cpu.append(data[ii, 2])
        else:
            if data[ii, 1] == 0:
                gpu.append(data[ii, 2])
            else:
                tracing_gpu.append(data[ii, 2])

    for matrix in [cpu, tracing_cpu, gpu, tracing_gpu]:
        mean_list.append(np.mean(matrix))
        std_list.append(np.std(matrix))

    # Uniform plotting
    ind = np.arange(4)
    plt.bar(ind, mean_list, width, yerr=std_list)

    plt.xticks(ind, labels)
    plt.ylabel("Inference time [s]")

    # plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
