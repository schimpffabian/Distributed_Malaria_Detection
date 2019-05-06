"""
author: fasc
"""
import os
import sys
import numpy as np

try:
    from src.train_model import custom_classifier
except ModuleNotFoundError:
    from train_model import custom_classifier

import timeit


def gpu_speedup():
    """
    Compares training time for a net with and without using a GPU
    """
    duration_gpu = []
    duration_cpu = []

    for ii in range(5):
        # GPU used
        start = timeit.default_timer()
        custom_classifier(use_gpu=True)
        end = timeit.default_timer()
        duration_gpu.append(end-start)

        # GPU not used
        start = timeit.default_timer()
        custom_classifier(use_gpu=False)
        end = timeit.default_timer()
        duration_cpu.append(end - start)

    np.savetxt("duration_gpu.csv", np.array(duration_gpu), delimiter=",")
    np.savetxt("duration_cpu.csv", np.array(duration_cpu), delimiter=",")


if __name__ == "__main__":
    gpu_speedup()