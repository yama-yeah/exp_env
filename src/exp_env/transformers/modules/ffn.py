import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.checkpoint

class FeedForward(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation: nn.Module = nn.SiLU(),use_checkpoint=False) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.activation = activation
        self.linear2 = nn.Linear(out_features, in_features)
        self.linear3 = nn.Linear(in_features, out_features)
        self.use_checkpoint = use_checkpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # y = self.activation(self.linear1(x))
        if self.use_checkpoint:
            x = torch.utils.checkpoint.checkpoint(self._forward, x,use_reentrant=False)*self.linear3(x)
        else:
            x = self._forward(x)*self.linear3(x)
        x = self.linear2(x)
        return x
    
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.linear1(x))
        return x
