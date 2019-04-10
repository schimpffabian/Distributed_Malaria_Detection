from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from matplotlib.image import imread
import sys, os

try:
    from src.models.Custom_CNN import Simple_CNN
    from src.auxiliaries import get_images, initialize_model
except:
    sys.path.append(os.path.join("..", ".."))  # add the current directory
    from src.models.Custom_CNN import Simple_CNN
    from src.auxiliaries import get_images, initialize_model
import torch
import torch.nn as F

import lime
from lime import lime_image
from scipy import misc


class Model:
    def __init__(self, net, device, input_size):
        self.net = net.double().to(device)
        self.device = device
        self.input_size = input_size
        self.softmax = torch.nn.Softmax()

    def predict(self, x):
        self.net.eval()
        x = (
            torch.from_numpy(x)
            .view(-1, 3, input_size, input_size)
            .double()
            .to(self.device)
        )
        output = self.net(x)
        return output.detach().cpu().numpy()

    def predict_prob(self, x):
        self.net.eval()
        x = (
            torch.from_numpy(x)
            .view(-1, 3, input_size, input_size)
            .double()
            .to(self.device)
        )
        output = self.net(x)
        output = self.softmax(output)
        return output.detach().cpu().numpy()


if __name__ == "__main__":
    # Reproduce
    # https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20Image%20Classification%20Keras.ipynb

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model, input_size = initialize_model("squeezenet", 2, True, use_pretrained=True)
    model.load_state_dict(torch.load("../models/squeezenet_e10.pt"))

    path = "../../data/Classification/Parasitized/"
    images = get_images(path)

    Model_Lime_API = Model(model, device, input_size)
    explainer = lime_image.LimeImageExplainer()

    for ii in range(45):
        image = imread(path + images[ii])
        image_resized = misc.imresize(image, (input_size, input_size))
        print(Model_Lime_API.predict(image_resized))
    print(image_resized[0:10, 0:10])
    explanation = explainer.explain_instance(
        image_resized,
        Model_Lime_API.predict_prob,
        hide_color=None,
        # num_samples=300,
    )

    temp, mask = explanation.get_image_and_mask(
        1, positive_only=False, num_features=20, hide_rest=False
    )

    plt.imshow(image_resized)
    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask), alpha=0.5)

    plt.show()
