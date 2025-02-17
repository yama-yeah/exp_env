from exp_env.reservoir.delay_esn import DelayESNModule
import torch
import torch.nn as nn

from exp_env.reservoir.esn import ESNModule
from exp_env.transformers.modules.dropout import Dropouts
from exp_env.transformers.modules.ffn import FeedForward


class ESNEncoderLayer(nn.Module):
    def __init__(self,dimension_of_query: int,hidden_dim: int,layer_norm_eps: float,dp_rate: float = 0.1,how_many_dropout=1,how_many_delay=1):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(dimension_of_query, eps=layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(dimension_of_query, eps=layer_norm_eps)
        self.esn = ESNModule(dimension_of_query, dimension_of_query)
        self.ffn = FeedForward(dimension_of_query, hidden_dim)
        self.pe = None # RotaryPositionalEmbedding(hidden_dim// how_many_heads, 256)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        y = self.esn(x, one_step_mode=False, auto_reset=True)
        z = self.layer_norm1(y+x)
        
        y = self.ffn(z)
        x = self.layer_norm2(x+y+z)
        return x
    