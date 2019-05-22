import sys
import os
sys.path.append(os.path.join(".."))

from src.dataloader import create_dataloaders
from src.auxiliaries import rgb2gray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np


def main():
    num_episodes = 5
    batch_size = 2000
    img_size = 64

    train_loader, test_loader, validation_loader = create_dataloaders(batch_size, img_size=img_size)
    del validation_loader

    classifier = LogisticRegression(random_state=42, warm_start=True, solver="liblinear")

    for episode in range(num_episodes):
        print("\nRun \t %.0f" % episode)
        for data, target in train_loader:
            # Reshape data for grayscale conversion
            data = data.view((-1, img_size, img_size, 3))
            data_train = data.detach().cpu().numpy()
            data_gray_train = rgb2gray(data_train)
            data_gray_train = data_gray_train.reshape((-1, img_size * img_size))
            target_train = target.detach().cpu().numpy()

            # Train classifier
            classifier.fit(data_gray_train, target_train)

        # Eval accuracy
        results = []
        for batch_id, (data, target) in enumerate(test_loader):
            data = data.view((-1, img_size, img_size, 3))

            data_test = data.detach().cpu().numpy()
            data_gray_test = rgb2gray(data_test)
            data_gray_test = data_gray_test.reshape((-1, img_size * img_size))
            target_test = target.detach().cpu().numpy()

            predictions = classifier.predict(data_gray_test)
            score = accuracy_score(target_test, predictions)
            results.append(score)
            print("Batch %.0f, Accuracy: %.2f" % (batch_id, score))

    print("\n\nMean accuracy: %.2f" % np.mean(results))


if __name__ == "__main__":
    main()