from torch import nn


class Residual(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x
