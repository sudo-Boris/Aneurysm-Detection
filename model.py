import torch
from torch import nn


def init_model():
    return SimpleCNN().cuda()


class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        self.conv1 = nn.Sequential(
            torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1
            ),
            torch.nn.Sigmoid(),
        )

    def __call__(self, x):
        return self.conv1(x)
