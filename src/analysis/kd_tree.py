import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from src.analysis.plot_config import params


def main():
    # Plotting settings
    params['figure.figsize'] = [8, 4]
    matplotlib.rcParams.update(params)
    width = 0.5

    # Load data
    # data = np.genfromtxt("../logs/speedup_kd_tree.csv", delimiter=",", skip_header=1)
    data = np.genfromtxt("../logs/speedup_kd_tree_fake_big.csv", delimiter=",", skip_header=1)

    data = data[data[:, 0].argsort()]

    # Get unique values
    unique_values = np.unique(data[:, 0])
    ind = np.arange(len(unique_values))*2
    ticks = [str(ii) for ii in unique_values]

    # Create lists for plots
    mean_build = []
    std_build = []

    mean_query = []
    std_query = []

    mean_naive = []
    std_naive = []

    fig, ax = plt.subplots()
    del fig

    for unique_val in unique_values:
        indices = list(np.where(data[:, 0] == unique_val)[0])
        selected_slice = data[indices, :]

        # select relevant rows
        time_build_kd = selected_slice[:, 1]
        time_query_kd = selected_slice[:, 2]
        naive_knn = selected_slice[:, 3]

        # Append lists for plotting
        mean_build.append(np.mean(time_build_kd))
        std_build.append(np.std(time_build_kd))

        mean_query.append(np.mean(time_query_kd))
        std_query.append(np.std(time_query_kd))

        mean_naive.append(np.mean(naive_knn))
        std_naive.append(np.std(naive_knn))

    ax.bar(ind+0.25, mean_naive, width, yerr=std_naive, label="Naive approach", bottom=0.000001)
    ax.bar(ind-0.25, mean_build, width, yerr=std_build, label="Build KD Tree",  bottom=0.000001)
    ax.bar(ind-0.25, mean_query, width,
                 bottom=mean_build, yerr=std_query, label="Query KD Tree")

    plt.xticks(ind, ticks)
    ax.set_yscale('log')
    plt.ylim((10**(-5), 100))
    plt.ylabel("Time [s]")
    plt.xlabel("Length of lists")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()