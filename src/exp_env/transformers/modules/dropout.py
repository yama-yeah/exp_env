import torch
from torch import nn

class Dropouts(nn.Module):
    def __init__(self, dp_rate: float = 0.1,how_many: int=1):
        super().__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(dp_rate) for _ in range(how_many)])
        self.how_many=how_many

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logist = torch.sum(torch.stack([dropout(x) for dropout in self.dropouts]), dim=0)/self.how_many
        return logist