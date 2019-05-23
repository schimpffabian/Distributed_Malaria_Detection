"""
Auxiliary functions
"""

import torch
import torch.nn as nn
from torchvision import models
from os import listdir
from os.path import isfile, join
import numpy as np

log_interval = 3


def set_parameter_requires_grad(model, feature_extracting):
    """
    Function for model finetuning to freeze feature extractor and retrain last layers

    :param model: PyTorch model child of torch.nn.Module
    :param bool feature_extracting: True disables gradients
    """
    #  Source: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.htmlS
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    """
    Source: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    Initialize these variables which will be set in this if statement. Each of these
    variables is model specific.

    :param str model_name: model to be loaded
    :param int num_classes: number of classes
    :param bool feature_extract: deactivate gradients
    :param bool use_pretrained: load pretrained weights
    :return: pretrained model, input_size
    """
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(
            512, num_classes, kernel_size=(1, 1), stride=(1, 1)
        )
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3, Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def train(
    model,
    device,
    train_loader,
    optimizer,
    epoch,
    loss,
    federated=False,
    random_background=False,
):
    """
    Training function for NNs

    :param torch.nn.Module model: PyTorch model child of
    :param str device: device to run the model on
    :param torch.utils.data.DataLoader train_loader: Dataloader
    :param torch.optim optimizer: Optimizer from
    :param int epoch:  number of epoches to train
    :param loss: Loss function from
    :param bool federated: federated training

    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if federated:
            pass
            # model_backup = copy.deepcopy(model)
        try:
            if federated:
                model.send(data.location)
        except KeyError:
            print("\n\n\n\nKey Error occured\n\n\n\n")
            break
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        prediction = model(data)

        output = loss(prediction, target)

        output.backward()
        optimizer.step()
        try:
            optimizer.step()
        except TypeError:
            print("Type Error occured")
            # model = model_backup
            break
        if federated:
            model.get()
        if batch_idx % log_interval == 0:
            if federated:
                output = output.get()

            if hasattr(train_loader, "dataset"):
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        output.item(),
                    )
                )


def run_t(model, device, test_loader, loss, secure_evaluation=False):
    """
    Test function for NNs

    :param model: PyTorch model child of torch.nn.Module
    :param str device: device to run the model on
    :param torch.utils.data.DataLoader() test_loader: Dataloader
    :param loss: Loss function from torch.nn
    """
    model.eval()
    test_loss = 0
    correct = 0
    correct_encoded = 0
    num_predictions = 0

    with torch.no_grad():
        for data, target in test_loader:
            batch_size = len(target)
            num_predictions += batch_size

            if not secure_evaluation:
                data, target = data.to(device), target.to(device)

            prediction = model(data)

            if not secure_evaluation:
                test_loss += loss(prediction, target)

                pred = prediction.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability

                correct += pred.eq(target.view_as(pred)).sum().item()

                if hasattr(test_loader, "dataset"):
                    print(
                        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
                            test_loss / num_predictions,
                            correct,
                            num_predictions,
                            100.0 * correct / num_predictions,
                        )
                    )

            else:
                pred = prediction.argmax(dim=1)
                correct_encoded += pred.eq(target.view_as(pred)).sum()
                correct_decoded = (
                    correct_encoded.copy().get().float_precision().long().item()
                )

                print(
                    "Test set: Accuracy: {}/{} ({:.0f}%)".format(
                        correct_decoded,
                        num_predictions,
                        100.0 * correct_decoded / num_predictions,
                    )
                )

        if secure_evaluation:
            return 100 * correct_decoded / num_predictions
        else:
            return 100 * correct / len(test_loader.dataset)


def get_images(path):
    """
    Return all images in path as list of strings

    :param str path: path to image directory
    :return: list of images in path
    """
    return [f for f in listdir(path) if isfile(join(path, f))]


def rgb2gray(rgb):
    """
    Converts RGB images to black and white

    :param ndarray rgb: RGB image - shape: (n, m, 3)
    :return: image - shape: (n, m, 1)
    """
    transform_factor = np.array([0.2989, 0.5870, 0.1140]).reshape((3, 1))
    return rgb @ transform_factor


def create_test_img(
    size=(200, 200), num_points=100, radius_min=1, radius_max=10, random_seed=42
):
    """
    Creates randomly distributed bright dots on black background

    :param tuple size: dimensions of test image
    :param int num_points:  number of bright spots
    :param int radius_min:  minimum radius for bright spots
    :param int radius_max:  maximum radius for bright spots
    :param int random_seed: random seed
    :return: test_img (ndarray), center_list (list), radius_list (list)
    """

    test_img = np.zeros(list(size))
    radius_list = []
    center_list = []
    np.random.seed(random_seed)

    for point_nr in range(num_points):
        if point_nr != 0 and point_nr % 1000 == 0:
            print("Point %.0f added" % point_nr)
        center_0 = np.random.randint(low=0, high=int(test_img.shape[0]))
        center_1 = np.random.randint(low=0, high=int(test_img.shape[1]))

        center = [center_0, center_1]
        center_list.append(center)

        radius = np.random.randint(low=radius_min, high=radius_max)
        radius_list.append(radius)

        for ii in range(test_img.shape[0]):
            for jj in range(test_img.shape[1]):
                if np.linalg.norm(np.array([ii, jj]) - center) < radius:
                    test_img[ii, jj] = 255

    return test_img, center_list, radius_list
