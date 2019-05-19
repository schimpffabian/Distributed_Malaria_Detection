import os
import sys
from scipy.spatial import KDTree
import sys
from skimage.feature import blob_log
import timeit
import numpy as np
from pathlib import Path

sys.path.append(os.path.join("..", ".."))
from src.auxiliaries import create_test_img


def build_kd_tree(center_list):
    """
    seperate function to build kd trees

    :param center_list:
    :return:
    """

    center_kd_tree = KDTree(center_list)
    return center_kd_tree


def query_kd_tree(kd_tree, blob_list):
    """

    :param kd_tree:
    :param blob_list:
    :return:
    """
    sys.setrecursionlimit(999999)
    for blob in blob_list:
        dist, idx = kd_tree.query(blob[0:2])


def naive_knn(blob_list, center_list):
    """

    :param blob_list:
    :param center_list:
    :return:
    """
    blob_list = np.array(blob_list)
    center_list = np.array(center_list)

    for blob in blob_list:
        for center in center_list:
            dist = np.linalg.norm(center-blob[0:2])


def main():
    """
    Compare runtimes
    """
    num_runs = 16
    num_points = [10, 33, 100, 333]
    time_build_kd = []
    time_query_kd = []
    time_naive_nn = []
    num_points_list = []

    for ii in range(num_runs):
        print("Run %.0f" % ii)
        index = ii % (len(num_points))
        num_points_list.append(num_points[index])

        # Create test image
        image, center_list, radius_list = create_test_img(
            size=(200, 200), num_points=num_points[index], radius_min=3, radius_max=9, random_seed=ii
        )

        # Run LoG
        blobs_log = blob_log(image, min_sigma=1, max_sigma=3, num_sigma=3, overlap=1)

        # Build Kd Tree
        start = timeit.default_timer()
        kd_tree = build_kd_tree(center_list)
        stop = timeit.default_timer()
        time_build_kd.append(stop-start)

        # Run KD Tree
        start = timeit.default_timer()
        query_kd_tree(kd_tree, blobs_log)
        stop = timeit.default_timer()
        time_query_kd.append(stop - start)

        # Naive approach
        start = timeit.default_timer()
        naive_knn(blobs_log, center_list)
        stop = timeit.default_timer()
        time_naive_nn.append(stop - start)

    # Save results
    save_list = np.array([num_points_list, time_build_kd, time_query_kd, time_naive_nn]).T
    np.savetxt(Path("../logs/speedup_kd_tree.csv"), save_list, delimiter=",",
               header=",num_points,time_build_kd,time_query_kd,time_naive_nn")


if __name__ == "__main__":
    main()