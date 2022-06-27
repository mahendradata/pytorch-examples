#!/usr/bin/env python3

import torch
import models
from torch import nn, optim
from utils import getFashionMNIST, train, predict


def run(model):
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get dataset
    _, test_data, train_loader, test_loader = getFashionMNIST()

    # *** Preparing the model ***
    model = model.to(device)
    # print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    # *** Training ***
    epochs = 5
    train(train_loader, test_loader, model, loss_fn, optimizer, epochs, device)

    # *** Predict test ***
    x, y = test_data[0][0], test_data[0][1]
    predict(model, x, y, device)


if __name__ == '__main__':
    print("BasicNN")
    run(models.BasicNN())
    print("RestNet18")
    run(models.RestNet18())
    