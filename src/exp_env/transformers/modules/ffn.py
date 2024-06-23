import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation: nn.Module = nn.SiLU()) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.activation = activation
        self.linear2 = nn.Linear(out_features, in_features)
        self.linear3 = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.linear1(x))*self.linear3(x)
        x = self.linear2(x)
        return x
