import torch
from pathlib import Path
import timeit
import copy
from torch.jit import trace
import numpy as np
import os
import sys

sys.path.append(os.path.join("..", ".."))
# from src.models.Custom_CNN import Simple_CNN_e1
from src.models.Custom_CNN import Simple_CNN_e2
from src.dataloader import create_dataloaders
from src.auxiliaries import initialize_model


def benchmark_inference_custom_model():
    """
    Benchmark inference duration with JIT and GPU execution
    """
    num_runs = 25

    batch_size = 1024
    img_size = 128
    random_background = 0

    path = Path("../state_dicts/custom_cnn_e4_0.pt")
    results = []

    # load model
    original_model = Simple_CNN_e2(img_size=img_size)
    original_model.load_state_dict(torch.load(path))

    # get dataloader
    train_loader, test_loader, val_loader = create_dataloaders(
        batchsize=batch_size, img_size=img_size, random_background=random_background
    )

    for use_gpu in [0, 1]:
        for use_tracing in [0, 1]:

            print(use_gpu, use_tracing)

            # Specify execution device
            if use_gpu:
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            # Clone original model
            execution_model = copy.deepcopy(original_model)
            original_model.train()

            if use_tracing:
                for (data, target) in val_loader:
                    example_input = data
                    break
                execution_model = trace(execution_model, example_inputs=example_input)

                if use_gpu:
                    execution_model.cuda()
            else:
                # Move model to GPU
                execution_model = execution_model.to(device)

            for ii in range(num_runs):
                start = timeit.default_timer()
                for (data, target) in val_loader:
                    data = data.to(device)
                    prediction = execution_model(data)
                end = timeit.default_timer()
                results.append([use_gpu, use_tracing, (end - start) / num_runs])
                np.savetxt(
                    Path("../logs/inference_speedup_custom_e2.csv"),
                    results,
                    delimiter=",",
                    header="use_gpu,use_tracing,duration",
                )


def benchmark_inference_squeezenet():
    """
    Benchmark inference duration with JIT and GPU execution
    """
    num_runs = 10

    batch_size = 32

    random_background = 0

    path = Path("../state_dicts/squeezenet_e3.pt")
    results = []

    # load model
    original_model, img_size = initialize_model(
        "squeezenet", 2, True, use_pretrained=True
    )
    original_model.load_state_dict(torch.load(path))

    # get dataloader
    train_loader, test_loader, val_loader = create_dataloaders(
        batchsize=batch_size, img_size=img_size, random_background=random_background
    )

    for use_gpu in [1, 0]:
        for use_tracing in [1, 0]:

            print(use_gpu, use_tracing)

            # Specify execution device
            if use_gpu:
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            # Clone original model
            execution_model = copy.deepcopy(original_model)
            execution_model.train()

            if use_tracing:
                for (data, target) in val_loader:
                    example_input = data
                    break
                execution_model = trace(
                    execution_model, example_inputs=example_input, check_trace=False
                )

                if use_gpu:
                    execution_model.cuda()
            else:
                # Move model to GPU
                execution_model = execution_model.to(device)

            for ii in range(num_runs):
                start = timeit.default_timer()
                for (data, target) in val_loader:
                    data = data.to(device)
                    prediction = execution_model(data)
                end = timeit.default_timer()
                results.append([use_gpu, use_tracing, (end - start) / num_runs])
                np.savetxt(
                    Path("../logs/inference_speedup_squeezenet.csv"),
                    results,
                    delimiter=",",
                    header="use_gpu,use_tracing,duration",
                )


if __name__ == "__main__":
    benchmark_inference_squeezenet()
