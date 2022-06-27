#!/usr/bin/env python3
# Ref: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

from torch import nn
from torchvision import models


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


class RestNet18(nn.Module):
    def __init__(self):
        super(RestNet18, self).__init__()
        self.conv = nn.Conv2d(1, 3, 1)  # Tricky way to transform 1 channel images to 3 channel images
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 10)

    def forward(self, x):
        logits = self.conv(x)
        logits = self.model(logits)
        return logits