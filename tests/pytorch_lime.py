from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from matplotlib.image import imread
import sys
import os

try:
    from src.models.Custom_CNN import Simple_CNN_e2
    from src.auxiliaries import get_images
    from src.auxiliaries import initialize_model
except ModuleNotFoundError:
    sys.path.append(os.path.join("..", ".."))
    # from src.models.Custom_CNN import Simple_CNN
    from src.auxiliaries import get_images
    from src.models.Custom_CNN import Simple_CNN_e2
    from src.auxiliaries import initialize_model
import torch
from lime import lime_image
from scipy import misc


class Model:
    """

    """

    def __init__(self, net, device, input_size):
        self.net = net.double().to(device)
        self.device = device
        self.input_size = input_size
        self.softmax = torch.nn.Softmax()

    def predict(self, x):
        """

        :param x:
        :return:
        """
        self.net.eval()
        x = (
            torch.from_numpy(x)
            .view(-1, 3, self.input_size, self.input_size)
            .double()
            .to(self.device)
        )
        output = self.net(x)
        return output.detach().cpu().numpy()

    def predict_prob(self, x):
        """

        :param x:
        :return:
        """
        self.net.eval()
        x = (
            torch.from_numpy(x)
            .view(-1, 3, self.input_size, self.input_size)
            .double()
            .to(self.device)
        )
        output = self.net(x)
        output = self.softmax(output)
        return output.detach().cpu().numpy()


def demonstrate_randomization():
    """

    :return:
    """
    # Reproduce
    # https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20Image%20Classification%20Keras.ipynb

    use_cuda = torch.cuda.is_available()

    use_pretrained_squeezenet = False
    path = "../data/Classification/Parasitized/"

    device = torch.device("cuda" if use_cuda else "cpu")

    if use_pretrained_squeezenet:
        model, input_size = initialize_model("squeezenet", 2, True, use_pretrained=True)
        model.load_state_dict(torch.load("../src/models/squeezenet_e10.pt"))
    else:
        input_size = 128
        model_random = Simple_CNN_e2(img_size=input_size)
        model_random.load_state_dict(torch.load("../src/state_dicts/custom_cnn_e4_1.pt"))

        model_standard = Simple_CNN_e2(img_size=input_size)
        model_standard.load_state_dict(torch.load("../src/state_dicts/custom_cnn_e4_0.pt"))

    images = get_images(path)
    image = imread(path + images[100])
    image_resized = misc.imresize(image, (input_size, input_size))

    plt.subplot(1, 3, 1)
    plt.imshow(image_resized)
    plt.title("Original image")

    plt.subplot(1, 3, 2)
    plt.imshow(image_resized)

    explainer_standard = lime_image.LimeImageExplainer()
    Model_Lime_API_standard = Model(model_standard, device, input_size)
    explanation_standard = explainer_standard.explain_instance(
        image_resized,
        Model_Lime_API_standard.predict_prob,
        hide_color=None,
        # num_samples=300,
    )
    temp, mask = explanation_standard.get_image_and_mask(0, positive_only=False, num_features=20, hide_rest=False)

    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask), alpha=0.35)
    plt.title("Explanation Default")

    plt.subplot(1, 3, 3)
    plt.imshow(image_resized)

    explainer_random = lime_image.LimeImageExplainer()
    Model_Lime_API_random = Model(model_random, device, input_size)
    explanation_random = explainer_random.explain_instance(
        image_resized,
        Model_Lime_API_random.predict_prob,
        hide_color=None,
        # num_samples=300,
    )
    temp, mask = explanation_random.get_image_and_mask(0, positive_only=False, num_features=20, hide_rest=False)

    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask), alpha=0.35)
    plt.title("Explanation Random")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demonstrate_randomization()
