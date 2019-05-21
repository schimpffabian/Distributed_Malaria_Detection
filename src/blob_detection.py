from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import blob_dog, blob_log, blob_doh
from math import sqrt
from skimage.transform import resize

try:
    from src.models.Custom_CNN import Simple_CNN_e2
    from src.auxiliaries import get_images, rgb2gray
except ModuleNotFoundError:
    from models.Custom_CNN import Simple_CNN_e2
    from auxiliaries import get_images, rgb2gray
import torch
from torch.jit import trace

IMG_SIZE = 128


def get_cell_image(x, y, r, img):
    """
    Receives x, y and the respective radius of a a blob in an image, returns rectangular image with
    blob inside

    :param float x: x coordinate of blob
    :param float y: y coordinate of blob
    :param float r: radius of blob
    :param img: image containing the blob
    :return: section of the image containing the blob
    """
    img_shape = img.shape
    cons_r = np.ceil(r)  # conservative radius is ceiled to encapsulate blob fully

    if x - cons_r >= 0:
        x_start = x - cons_r
    else:
        x_start = 0

    if x + cons_r <= img_shape[0]:
        x_end = x + cons_r
    else:
        x_end = img_shape[0]

    if y - cons_r >= 0:
        y_start = y - cons_r
    else:
        y_start = 0

    if y + cons_r <= img_shape[1]:
        y_end = y + cons_r
    else:
        y_end = img_shape[1]

    x_start = int(x_start)
    x_end = int(x_end)
    y_start = int(y_start)
    y_end = int(y_end)

    # grayscale
    if len(img_shape) == 2:
        return img[x_start:x_end, y_start:y_end]
    # RGB, HSV etc
    if len(img_shape) == 3:
        return img[x_start:x_end, y_start:y_end, :]


def create_blob_sequence(image):
    """
    Apply multiple blob detection algorithms to compare them
    :param image: image to be analysed
    :return: squenece containing the blobs, colors and titles
    """
    blobs_log = blob_log(image, min_sigma=15, max_sigma=40, num_sigma=10, overlap=1)

    # Compute radii in the 3rd column.
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

    blobs_dog = blob_dog(image, min_sigma=5, max_sigma=40, overlap=1)
    blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

    blobs_doh = blob_doh(image, min_sigma=5, max_sigma=40, overlap=1)

    blobs_list = [blobs_log, blobs_dog, blobs_doh]
    colors = ["yellow", "green", "black"]
    titles = [
        "Laplacian of Gaussian",
        "Difference of Gaussian",
        "Determinant of Hessian",
    ]
    sequence = zip(blobs_list, colors, titles)

    return sequence


#  @profile
def classify_cell_image(cell_image, model):
    """

    :param cell_image:
    :param model:
    :return:
    """
    cell_resized = resize(cell_image, (IMG_SIZE, IMG_SIZE), anti_aliasing=False)
    cell_torch = torch.from_numpy(cell_resized).reshape((1, 3, IMG_SIZE, IMG_SIZE))
    cell_torch = cell_torch.double()
    prediction = model(cell_torch)

    return prediction.data.numpy().argmax()


def load_model(path, tracing=False):
    """
    function to load a network for classification
    :param str path: state dicts of previous training
    :param bool tracing: turn tracing on or off
    :return: model
    """
    model = Simple_CNN_e2(128)
    model.load_state_dict(torch.load(path))
    model = model.double()

    if tracing:
        sample_input = torch.rand((1, 3, IMG_SIZE, IMG_SIZE)).double()
        traced_model = trace(model, example_inputs=sample_input)
        return traced_model
    else:
        return model


def main():
    """
    Demonstration of the combination of object detection and classification
    """

    path = "../data/Real_Application/"
    images = get_images(path)

    model = load_model(path="./state_dicts/custom_cnn_e4_0.pt", tracing=False)

    for image in images:
        # image = "malaria_0.jpg"
        img = imread(path + image)
        image_gray = rgb2gray(img)
        image_gray = image_gray.squeeze()

        image_gray_inverse = 255 - image_gray

        sequence = create_blob_sequence(image_gray_inverse)

        fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
        ax = axes.ravel()

        for idx, (blobs, color, title) in enumerate(sequence):
            ax[idx].set_title(title)
            ax[idx].imshow(image_gray_inverse, interpolation="nearest")
            for blob in blobs:
                y, x, r = blob
                cell_img = get_cell_image(x, y, r, img)
                try:

                    label = classify_cell_image(cell_img, model)

                except ValueError:
                    label = 1
                if label == 1:
                    c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
                else:
                    c = plt.Circle((x, y), r, color="r", linewidth=2, fill=False)
                ax[idx].add_patch(c)
            ax[idx].set_axis_off()

        #break
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
