import torch
from pathlib import Path
import timeit
import copy
from torch.jit import trace

try:
    from src.models.Custom_CNN import Simple_CNN_e1
    from src.models.Custom_CNN import Simple_CNN_e2
    from src.dataloader import create_dataloaders
except ModuleNotFoundError:
    from dataloader import create_dataloaders
    from models.Custom_CNN import Simple_CNN_e1
    from models.Custom_CNN import Simple_CNN_e2


def benchmark_inference():
    """
    Benchmark inference duration with JIT and GPU execution
    """

    batch_size = 1024
    img_size = 128
    random_background = 0

    path = Path("./state_dicts/custom_cnn_e4_0.pt")
    header = "counter, batch_size, img_size, random_background, gpu_used, ii, end-start, accuracy"

    # load model
    original_model = Simple_CNN_e2(img_size=img_size)
    original_model.load_state_dict(torch.load(path))

    # get dataloader
    train_loader, test_loader, val_loader = create_dataloaders(
        batchsize=batch_size, img_size=img_size, random_background=random_background
    )

    for use_gpu in [0, 1]:
        for use_tracing in [0, 1]:


            # Specify execution device
            if use_gpu:
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')

            # Clone original model
            execution_model = copy.deepcopy(original_model)
            execution_model = execution_model.to(device)

            if use_tracing:
                for (data, target) in val_loader:
                    example_input = data.to(device)
                    break
                execution_model = trace(execution_model, example_inputs=example_input)


            start = timeit.default_timer()
            for (data, target) in val_loader:
                pass



if __name__ == "__main__":
    benchmark_inference()
