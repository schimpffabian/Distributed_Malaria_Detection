import torch
import torch.nn as nn
import syft as sy
from src.federated_training import create_federated_dataset
from src.auxiliaries import run_t
from src.models.Custom_CNN import Simple_CNN2


def secure_evaluation(img_size=128):
    """
    https://blog.openmined.org/encrypted-deep-learning-classification-with-pysyft/
    """

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(42)
    hook = sy.TorchHook(torch)
    # client = sy.VirtualWorker(hook, id="client")
    katherienhospital = sy.VirtualWorker(
        hook, id="kh"
    )  # <-- NEW: define remote worker bob
    filderklinik = sy.VirtualWorker(hook, id="fikli")
    crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")

    train_set, test_set = create_federated_dataset(img_size=img_size)
    del train_set
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=42, shuffle=True)

    model = Simple_CNN2(img_size)
    # model.load_state_dict(torch.load("./models/custom_cnn_e10_size_48.pt"))

    device = torch.device("cuda" if use_cuda else "cpu")
    # loss = F.nll_loss()
    loss = nn.NLLLoss()

    # Changes for secure evaluation
    private_test_loader = []
    for data, target in test_loader:

        private_test_loader.append(
            (
                data.fix_prec().share(
                    katherienhospital, filderklinik, crypto_provider=crypto_provider
                ),
                target.fix_prec().share(
                    katherienhospital, filderklinik, crypto_provider=crypto_provider
                ),
            )
        )

    model.fix_precision().share(
        katherienhospital, filderklinik, crypto_provider=crypto_provider
    )

    run_t(model, device, private_test_loader, loss, secure_evaluation=True)


if __name__ == "__main__":
    secure_evaluation()