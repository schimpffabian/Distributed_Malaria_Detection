import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from src.auxiliaries import create_test_img

# from src.pytorch_log import log

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../")


def test_1():
    test_img, center, radius = create_test_img()
    assert True


if __name__ == "__main__":
    # create_test_image
    test_img = np.zeros([200, 200])

    center = np.array([int(test_img.shape[0] / 2), int(test_img.shape[1] / 2)])
    radius = 15

    for ii in range(test_img.shape[1]):
        for jj in range(test_img.shape[0]):
            if np.linalg.norm(np.array([ii, jj]) - center) < radius:
                test_img[ii, jj] = 255

    plt.imshow(test_img)
    plt.show()
