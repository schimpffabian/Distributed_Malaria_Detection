"""
author: fasc
"""
import os
import sys
import numpy as np
from pathlib import Path
import timeit

sys.path.append(os.path.join("..", ".."))

from src.train_model import custom_classifier
from src.train_model import finetune_model
# from src.models.Custom_CNN import Simple_CNN_e1
from src.models.Custom_CNN import Simple_CNN_e2


def gpu_speedup_custom():
    """
    Compares training time for a net with and without using a GPU
    """
    num_epochs = 100

    results = []
    counter = 0
    header = "counter, batch_size, img_size, random_background, gpu_used, ii, end-start, accuracy"

    for batch_size in [1024]:
        for img_size in [128]:
            for random_background in [0, 1]:
                for gpu_used in [1]:
                    for ii in range(1):
                        print("\n %1.0f \n" % counter)
                        start = timeit.default_timer()

                        name = "../state_dicts/custom_cnn_e4_" + str(int(counter))

                        accuracy = custom_classifier(Simple_CNN_e2(img_size=img_size),
                                                     num_epochs=num_epochs,
                                                     batch_size=batch_size,
                                                     use_gpu=gpu_used,
                                                     img_size=img_size,
                                                     random_background=random_background,
                                                     name=name)

                        end = timeit.default_timer()

                        results.append([counter,
                                        batch_size,
                                        img_size,
                                        random_background,
                                        gpu_used,
                                        ii,
                                        end-start,
                                        accuracy])

                        counter += 1

                        # Log results
                        np.savetxt(Path("../logs/results_experiment_4.csv"),
                                   np.array(results),
                                   delimiter=",",
                                   header=header)


def gpu_speedup_pretrained():
    """
    Compares training time for a net with and without using a GPU
    """

    results = []
    counter = 0
    header = "counter, batch_size, img_size, random_background, gpu_used, ii, end-start, accuracy"

    for batch_size in [256]:
        for img_size in [224]:
            for random_background in [0]:
                for gpu_used in [1, 0]:
                    for ii in range(3):
                        print("\n %1.0f \n" % counter)
                        start = timeit.default_timer()

                        name = "../state_dicts/squeezenet_e5_" + str(int(counter))

                        accuracy = finetune_model(batch_size=batch_size,
                                                  num_epochs=3,
                                                  use_gpu=gpu_used)

                        end = timeit.default_timer()

                        results.append([counter,
                                        batch_size,
                                        img_size,
                                        random_background,
                                        gpu_used,
                                        ii,
                                        end-start,
                                        accuracy])

                        counter += 1

                        # Log results
                        np.savetxt(Path("../logs/results_experiment_5.csv"),
                                   np.array(results),
                                   delimiter=",",
                                   header=header)


if __name__ == "__main__":
    gpu_speedup_pretrained()
