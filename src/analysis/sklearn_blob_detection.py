import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from src.analysis.plot_config import params


def main():
    """
    Simple analysis script to plot True Positive and Detection Rate over Time
    """
    # Set plotting parameters
    params["figure.figsize"] = [8, 5]
    matplotlib.rcParams.update(params)

    # different algorithms
    name_list = ["log", "dog", "doh", "sbd"]
    label_list = [
        "Laplassian of Gaussian",
        "Difference of Gaussian",
        "Difference of Hessian",
        "OpenCV SBD",
    ]
    colors = ["b", "g", "y", "m"]

    # Create lists to store results for plotting
    time_mean = []
    time_std = []

    # Load runtimes
    runtimes = np.genfromtxt(
        "../logs/runtime_blob_detection.csv", delimiter=",", skip_header=1
    )

    for col in range(runtimes.shape[1]):
        run_dur = runtimes[:, col]
        time_mean.append(np.mean(run_dur))
        time_std.append(np.std(run_dur))

    # Load data from runs
    for index, name in enumerate(name_list):
        # Load results
        load_str = "../logs/" + name + "_acc_blob_detection.csv"
        data = np.genfromtxt(load_str, delimiter=",", skip_header=1)

        # Seperate columns
        true_pos = data[:, 0]
        detected_centers = data[:, 2]

        true_pos_mean = np.mean(true_pos)
        true_pos_std = np.std(true_pos)

        detected_centers_mean = np.mean(detected_centers)
        detected_centers_std = np.std(detected_centers)

        ax1 = plt.subplot(2, 1, 1)
        box1 = ax1.get_position()
        ax1.set_position([box1.x0, box1.y0, box1.width * 0.9, box1.height])
        ax1.set_xscale("log", nonposx="clip")
        plt.plot(
            time_mean[index],
            true_pos_mean,
            "o" + colors[index],
            label=label_list[index],
        )
        plt.errorbar(
            time_mean[index],
            true_pos_mean,
            yerr=true_pos_std,
            xerr=time_std[index],
            color=colors[index],
        )
        # ax1.legend()

        ax2 = plt.subplot(2, 1, 2)
        box2 = ax2.get_position()
        ax2.set_position([box2.x0, box2.y0, box2.width * 0.9, box2.height])
        ax2.set_xscale("log", nonposx="clip")
        plt.plot(
            time_mean[index],
            detected_centers_mean,
            "o" + colors[index],
            label=label_list[index],
        )
        plt.errorbar(
            time_mean[index],
            detected_centers_mean,
            yerr=detected_centers_std,
            xerr=time_std[index],
            color=colors[index],
        )
        ax2.legend(loc="center left", bbox_to_anchor=(1, 1.15))

    ax1.grid()
    ax1.set_ylabel("True Positive Rate")
    ax1.set_xlabel("Runtime [s]")

    ax2.grid()
    ax2.set_ylabel("Detection Rate")
    ax2.set_xlabel("Runtime [s]")

    plt.show()


if __name__ == "__main__":
    main()
