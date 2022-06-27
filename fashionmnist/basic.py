#!/usr/bin/env python3
# Ref: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

import torch
from torch import nn, optim
from utils import getFashionMNIST, train, eval



# Define model
class BasicNN(nn.Module):
    def __init__(self):
        super(BasicNN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits



if __name__ == '__main__':
    # Get dataset
    train_data, test_data, train_loader, test_loader = getFashionMNIST()

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # *** Preparing the model ***
    model = BasicNN()
    model = model.to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    # *** Training ***

    epochs = 5
    for t in range(epochs):
        print(f'Epoch {t+1}\n-------------------------------')
        train(train_loader, model, loss_fn, optimizer, device)
        eval(test_loader, model, loss_fn, device)
    print('Done!')

    model.eval()
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        x = x.to(device)  # Move to specified device
        pred = model(x)
        predicted, actual = pred[0].argmax(0), y
        print(f'Predicted: "{predicted}", Actual: "{actual}"')