"""
The LeNet-5 architecture:

    INPUT: 32 x 32
            v
    C1: 6 @ 28 x 28
            v
    S2: 6 @ 14 x 14 (sub-sampling)
            v
    C3: 16 @ 10 x 10
            v
    S4: 16 @ 5 x 5 (sub-sampling)
            v
    C5: 120 @ 1 x 1
            v
    F6: 84 Linear
            v
    OUTPUT: 10 Gaussian
"""

import torch
from torch.nn import Sequential, Module


class C1S2(Module):
    def __init__(self):
        super().__init__()
        self.m = Sequential(
            torch.nn.Conv2d(1, 6, kernel_size=(5, 5)),      # Number of channels = number of feature maps
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        )

    def forward(self, x):
        return self.m(x)


class C3S4(Module):
    def __init__(self):
        super().__init__()
        self.m = Sequential(
            torch.nn.Conv2d(6, 16, kernel_size=(5, 5)),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        )

    def forward(self, x):
        return self.m(x)


class C5(Module):
    def __init__(self):
        super().__init__()
        self.m = Sequential(
            torch.nn.Conv2d(16, 120, kernel_size=(5, 5)),
            torch.nn.ReLU()
        )

    def forward(self, x):
        return self.m(x)


class F6(Module):
    def __init__(self):
        super().__init__()
        self.m = Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(120, 84),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        return self.m(x)


class PyLeNet5(Module):
    def __init__(self):
        super().__init__()
        self.model = Sequential(
            C1S2(),
            C3S4(),
            C5(),
            F6(),
            torch.nn.Linear(84, 10)     # OUTPUT layer
        )

    def forward(self, x):
        return self.model(x)
