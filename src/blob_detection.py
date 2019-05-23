from matplotlib.image import imread
from src.analysis.plot_config import params
import line_profiler
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import blob_dog, blob_log, blob_doh
from math import sqrt
from skimage.transform import resize
import timeit

try:
    from src.models.Custom_CNN import Simple_CNN_e2
    from src.auxiliaries import get_images, rgb2gray
except ModuleNotFoundError:
    from models.Custom_CNN import Simple_CNN_e2
    from auxiliaries import get_images, rgb2gray
import torch
from torch.jit import trace

IMG_SIZE = 128


#  @profile
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
    colors = ["yellow", "green", "black", "blue"]
    titles = [
        "Laplacian of Gaussian",
        "Difference of Gaussian",
        "Determinant of Hessian",
    ]
    sequence = zip(blobs_list, colors, titles)

    return sequence


#  @profile
def classify_cell_image(cell_image, model, anti_aliasing=False):
    """
    Feeds cell_image to model and returns prediction

    :param cell_image: image to be classified
    :param model: PyTorch model used for classification
    :param bool anti_aliasing: turn anti aliasing on
    :return: prediction
    """
    cell_resized = resize(cell_image, (IMG_SIZE, IMG_SIZE), anti_aliasing=anti_aliasing)
    cell_torch = torch.from_numpy(cell_resized).reshape((1, 3, IMG_SIZE, IMG_SIZE))
    cell_torch = cell_torch.double()
    prediction = model(cell_torch)

    return prediction.data.numpy().argmax()


def load_model(path, tracing=False, img_size=128):
    """
    function to load a network for classification

    :param str path: state dicts of previous training
    :param bool tracing: turn tracing on or off
    :param int img_size: input size needed for model initialization
    :return: model
    """
    model = Simple_CNN_e2(img_size)
    model.load_state_dict(torch.load(path))
    model = model.double()

    if tracing:
        sample_input = torch.rand((1, 3, IMG_SIZE, IMG_SIZE)).double()
        traced_model = trace(model, example_inputs=sample_input)
        return traced_model
    else:
        return model


def compare_blob_detection_algorithms(path, model):
    """
    Loads images in path, apply different blob detection algorithms, classify detected blobs

    :param str path: path to images that should be analyzed
    :param model: PyTorch model to use for classification
    """
    # Plotting settings
    params["figure.figsize"] = [8, 4]
    matplotlib.rcParams.update(params)

    images = get_images(path)
    for image in images:
        img = imread(path + image)
        image_gray = rgb2gray(img)
        image_gray = image_gray.squeeze()

        image_gray_inverse = 255 - image_gray

        sequence = create_blob_sequence(image_gray_inverse)

        fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
        del fig
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

        plt.tight_layout()
        plt.show()


#  @profile
def dog_blob_detection(image):
    """
    Difference of Gaussian Blob detection - Mere wrapper for profiler

    :param np.ndarray image: grayscale image to analyze
    :return: list of blobs [y, x, r]
    """
    blobs = blob_dog(image, min_sigma=5, max_sigma=40, overlap=1)
    blobs[:, 2] = blobs[:, 2] * sqrt(2)

    return blobs


# @profile
def analyze_blobs(blobs, model, img):
    """
    Apply classification model to every blob

    :param np.ndarray blobs: [nx3] array with blobs [y, x, r]
    :param model: model used for classification
    :param img: full image to crop input from
    :return: list of labels
    """
    labels = []
    for row in range(blobs.shape[0]):
        y = blobs[row, 0]
        x = blobs[row, 1]
        r = blobs[row, 2]
        cell_img = get_cell_image(x, y, r, img)

        try:
            label = classify_cell_image(cell_img, model)
        except ValueError:
            label = 1

        labels.append(label)

    return labels


# @profile
def analyse_image(path, model):
    """
    Loads images in path, apply different blob detection algorithms, classify detected blobs

    :param str path: path to images that should be analyzed
    :param model: PyTorch model to use for classification
    """

    images = get_images(path)
    for image in images:
        img = imread(path + image)
        image_gray = rgb2gray(img)
        image_gray = image_gray.squeeze()

        image_gray_inverse = 255 - image_gray

        start = timeit.default_timer()

        blobs = dog_blob_detection(image_gray_inverse)
        analyze_blobs(blobs, model, img)

        end = timeit.default_timer()
        print("It took %.3f s to analyze the picture" % (end - start))


def main():
    """
    Demonstration of the combination of object detection and classification
    """

    path = "../data/Real_Application/"

    model = load_model(path="./state_dicts/custom_cnn_e4_0.pt", tracing=True)
    model_traced = load_model(path="./state_dicts/custom_cnn_e4_0.pt", tracing=True)

    # Uncomment application to run

    # compare_blob_detection_algorithms(path, model)
    analyse_image(path, model)
    profile_analyse_image()
    # analyse_image(path, model_traced)


def profile_analyse_image():
    """
    Profile the analysis function
    """

    path = "../data/Real_Application/"

    model = load_model(path="./state_dicts/custom_cnn_e4_0.pt", tracing=True)
    lp = line_profiler.LineProfiler()
    lp_wrapper = lp(analyse_image)
    lp_wrapper(path, model)
    lp.print_stats()


if __name__ == "__main__":
    main()
